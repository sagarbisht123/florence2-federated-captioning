# -*- coding: utf-8 -*-
"""
Florence-2 Federated Learning for Detailed Captioning
Model : microsoft/Florence-2-large-ft (LoRA + fp16, NO quantization)
GPU   : Tesla V100 16 GB  — fp16 + LoRA fits comfortably without bitsandbytes
Task  : <DETAILED_CAPTION>
FL    : 3 clients, 20 rounds, FedProx, FedAvg aggregation
Eval  : BLEU / METEOR / ROUGE-L / CIDEr
Output: <output_dir>/run_YYYY-MM-DD_HH-MM-SS/
          ├── run_config.json
          ├── model_checkpoints/best/
          └── scores/
               ├── round_metrics.csv   (live update every round)
               └── final_results.json
"""

import os
import json
import gc
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler,
)
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider

# ========================= NLTK DOWNLOADS =========================
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)

# ========================= ARGUMENTS =========================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/fed_input_data")
OUTPUT_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",   type=str, default=DATA_DIR,
                    help="Path to fed_input_data/")
parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                    help="Base path — each run creates output_dir/run_YYYY-MM-DD_HH-MM-SS/")
args = parser.parse_args()

DATA_DIR = args.data_dir

# ── Timestamped run directory ─────────────────────────────────────────────────
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR    = os.path.join(args.output_dir, f"run_{RUN_TIMESTAMP}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n{'='*70}")
print(f"Run directory : {OUTPUT_DIR}")
print(f"{'='*70}\n")

# ========================= CONFIGURATION =========================
MODEL_NAME   = "microsoft/Florence-2-base"
CAPTION_TASK = "<DETAILED_CAPTION>"

RESIZE_IMAGES       = False
RESIZE_SCALE_FACTOR = 0.5

# LoRA
LORA_R       = 8     # down from 16
LORA_ALPHA   = 16    # keep 2x of r — standard practice
LORA_DROPOUT = 0.05  # slightly lower dropout for smaller r

# FL
NUM_CLIENTS   = 3
ROUNDS        = 20
LOCAL_EPOCHS  = 2
LEARNING_RATE = 1e-4
BATCH_SIZE    = 2        # fp16 on V100 16 GB — safe at 2
PATIENCE      = 6
MU            = 0.01     # FedProx

# --- Memory Optimization ---
GRADIENT_CHECKPOINTING    = True  # saves VRAM by recomputing activations
GRADIENT_ACCUMULATION     = 4     # accumulate gradients over N steps
                                  # effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION

# Device  (NO quantization — fp16 only, no bitsandbytes needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE      = torch.device("cuda:0")
torch_dtype = torch.float16
torch.cuda.set_device(0)

print(f"Device      : {DEVICE}  ({torch.cuda.get_device_name(0)})")
print(f"Model       : {MODEL_NAME}")
print(f"Task        : {CAPTION_TASK}")
print(f"Precision   : fp16  (no quantization — no bitsandbytes)")
print(f"Batch size  : {BATCH_SIZE}")

# ========================= OUTPUT DIRS =========================
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "model_checkpoints")
SCORES_DIR     = os.path.join(OUTPUT_DIR, "scores")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SCORES_DIR,     exist_ok=True)

# ── Persist run config ────────────────────────────────────────────────────────
run_config = {
    "run_timestamp" : RUN_TIMESTAMP,
    "model"         : MODEL_NAME,
    "task"          : CAPTION_TASK,
    "precision"     : "fp16_no_quantization",
    "lora_r"        : LORA_R,
    "lora_alpha"    : LORA_ALPHA,
    "lora_dropout"  : LORA_DROPOUT,
    "num_clients"   : NUM_CLIENTS,
    "rounds"        : ROUNDS,
    "local_epochs"  : LOCAL_EPOCHS,
    "learning_rate" : LEARNING_RATE,
    "batch_size"    : BATCH_SIZE,
    "patience"      : PATIENCE,
    "mu_fedprox"    : MU,
    "data_dir"      : DATA_DIR,
    "output_dir"    : OUTPUT_DIR,
}
with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2)
print(f"Config saved → {os.path.join(OUTPUT_DIR, 'run_config.json')}\n")

# ========================= HELPERS =========================
def gpu_mem_str():
    if torch.cuda.is_available():
        alloc    = torch.cuda.memory_allocated(DEVICE) / 1024**3
        reserved = torch.cuda.memory_reserved(DEVICE)  / 1024**3
        return f"{alloc:.1f}/{reserved:.1f}GB"
    return "N/A"

# ========================= DATA =========================
print("="*70)
print("LOADING FEDERATED DATA")
print("="*70)

CLIENT_NAMES = ["client_01_data", "client_02_data", "client_03_data"]
TEST_NAME    = "test_data"

def resize_images(image_dir, output_dir, scale):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(image_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        new_size = (int(img.width * scale), int(img.height * scale))
        img.resize(new_size, Image.LANCZOS).save(os.path.join(output_dir, fname))
    print(f"Resized → {output_dir}")

class JSONLDataset:
    def __init__(self, jsonl_path, image_dir):
        self.image_dir = image_dir
        self.entries   = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e          = self.entries[idx]
        image_field = e["image"]

        # Primary: try the path as stored (works when absolute path is valid)
        primary_path = os.path.join(self.image_dir, image_field)
        if os.path.isfile(primary_path):
            return Image.open(primary_path).convert("RGB"), e

        # Fallback: strip to bare filename and look in self.image_dir
        # (handles the case where dataset was moved to a different machine)
        filename     = os.path.basename(image_field)
        fallback_path = os.path.join(self.image_dir, filename)
        if os.path.isfile(fallback_path):
            return Image.open(fallback_path).convert("RGB"), e

        raise FileNotFoundError(
            f"Image not found.\n"
            f"  Tried (primary)  : {primary_path}\n"
            f"  Tried (fallback) : {fallback_path}\n"
            f"  Check that images exist in: {self.image_dir}"
        )

class CaptionDataset(Dataset):
    def __init__(self, jsonl_path, image_dir, task_prefix):
        self.dataset     = JSONLDataset(jsonl_path, image_dir)
        self.task_prefix = task_prefix

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, data = self.dataset[idx]
        return self.task_prefix, data["suffix"], image

def get_active_image_dir(base_dir):
    if not RESIZE_IMAGES:
        return base_dir
    resized = base_dir.rstrip("/") + "_resized"
    if not os.path.exists(resized):
        resize_images(base_dir, resized, RESIZE_SCALE_FACTOR)
    return resized

# Load client datasets
client_datasets = []
client_sizes    = []
for cid, name in enumerate(CLIENT_NAMES):
    path     = os.path.join(DATA_DIR, name)
    img_dir  = get_active_image_dir(os.path.join(path, "images"))
    ann_path = os.path.join(path, "annotations", "annotations.jsonl")
    ds       = CaptionDataset(ann_path, img_dir, CAPTION_TASK)
    client_datasets.append(ds)
    client_sizes.append(len(ds))
    print(f"  Client {cid+1} ({name}): {len(ds)} samples")

# Load test dataset
test_path    = os.path.join(DATA_DIR, TEST_NAME)
test_img_dir = get_active_image_dir(os.path.join(test_path, "images"))
test_ann     = os.path.join(test_path, "annotations", "annotations.jsonl")
test_dataset = CaptionDataset(test_ann, test_img_dir, CAPTION_TASK)
print(f"  Test set  : {len(test_dataset)} samples\n")

# ========================= MODEL + LoRA =========================
print("="*70)
print("LOADING FLORENCE-2 + LoRA  (fp16, no quantization)")
print("="*70)

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ── fp16 only — BitsAndBytesConfig intentionally removed ─────────────────────
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    device_map={"": DEVICE},
    trust_remote_code=True,
)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                    "linear", "Conv2d", "lm_head", "fc2"],
    task_type="CAUSAL_LM",
    bias="none",
    inference_mode=False,
    use_rslora=True,
    init_lora_weights="gaussian",
)

global_model = get_peft_model(base_model, lora_config)
global_model.enable_input_require_grads()
global_model.gradient_checkpointing_enable()
global_model.print_trainable_parameters()
print(f"GPU after model load : {gpu_mem_str()}")
print("LoRA applied — only adapter weights are trainable\n")

# ========================= COLLATE =========================
def collate_fn(batch):
    prefixes, captions, images = zip(*batch)
    inputs = processor(
        text=list(prefixes),
        images=list(images),
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)
    return inputs, captions

# ========================= LOCAL TRAINING =========================
def local_train(model, dataset, epochs, lr, global_ref_params,
                round_num, client_id):
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    optimizer    = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    
    # account for accumulation steps in total training steps
    # max(..., 1) guards against tiny datasets where len(loader) < GRADIENT_ACCUMULATION
    num_steps    = max(1, epochs * (len(loader) // GRADIENT_ACCUMULATION))
    lr_scheduler = get_scheduler("linear", optimizer,
                                 num_warmup_steps=min(2, num_steps),
                                 num_training_steps=num_steps)
    model.train()
    global_step = 0
    optimizer.zero_grad()  # moved outside loop for accumulation

    for epoch in range(epochs):
        epoch_loss  = 0.0
        epoch_prox  = 0.0

        bar = tqdm(
            loader,
            desc=f"  [R{round_num:02d}|C{client_id}] Epoch {epoch+1}/{epochs}",
            leave=True,
            dynamic_ncols=True,
        )

        for step, (inputs, captions) in enumerate(bar, 1):
            inputs = {
                k: v.to(dtype=torch_dtype) if v.dtype.is_floating_point else v
                for k, v in inputs.items()
            }
            labels = processor.tokenizer(
                list(captions), padding=True,
                return_tensors="pt",
                return_token_type_ids=False,
            ).input_ids.to(DEVICE)

            outputs  = model(**inputs, labels=labels)
            ce_loss  = outputs.loss

            # FedProx proximal term
            prox_term = torch.tensor(0.0, device=DEVICE)
            for name, param in model.named_parameters():
                if param.requires_grad and name in global_ref_params:
                    diff = param.float() - global_ref_params[name].to(param.device).float()
                    prox_term += (diff ** 2).sum()

            loss = ce_loss + (MU / 2.0) * prox_term

            # scale loss by accumulation steps
            loss = loss / GRADIENT_ACCUMULATION
            loss.backward()

            epoch_loss += ce_loss.item()
            epoch_prox += prox_term.item()

            # only update weights every GRADIENT_ACCUMULATION steps
            if step % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            bar.set_postfix(
                ce     = f"{ce_loss.item():.4f}",
                avg_ce = f"{epoch_loss / step:.4f}",
                prox   = f"{prox_term.item():.4f}",
                lr     = f"{lr_scheduler.get_last_lr()[0]:.2e}",
                gpu    = gpu_mem_str(),
                step   = f"{global_step}/{num_steps}",
            )

        # handle remaining steps that didn't complete a full accumulation
        if step % GRADIENT_ACCUMULATION != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        print(
            f"    ↳ Epoch {epoch+1} done | "
            f"avg CE={epoch_loss/len(loader):.4f} | "
            f"avg prox={epoch_prox/len(loader):.4f} | "
            f"GPU {gpu_mem_str()}"
        )

        
# ========================= EVALUATION =========================
def generate_predictions(model, dataset, processor, max_samples=200):
    model.eval()
    preds, refs = [], []
    n = min(len(dataset), max_samples)

    bar = tqdm(range(n), desc="  Generating predictions", dynamic_ncols=True)
    with torch.no_grad():
        for i in bar:
            prefix, gt, img = dataset[i]
            inputs    = processor(text=prefix, images=img,
                                  return_tensors="pt").to(DEVICE)
            generated = model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=3,
                length_penalty=1.0,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            pred = processor.decode(generated[0], skip_special_tokens=True).strip()
            preds.append(pred)
            refs.append(gt)

            if i % 50 == 0:
                torch.cuda.empty_cache()

            bar.set_postfix(sample=f"{i+1}/{n}", gpu=gpu_mem_str())

    return preds, refs

def evaluate_captions(preds, refs):
    smooth  = SmoothingFunction().method1
    rouge   = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    pt      = [nltk.word_tokenize(p.lower()) for p in preds]
    rt      = [[nltk.word_tokenize(r.lower())] for r in refs]   # corpus_bleu format: list of list-of-refs
    rt_flat = [nltk.word_tokenize(r.lower()) for r in refs]     # meteor_score format: flat token list per ref
    bleu    = corpus_bleu(rt, pt, smoothing_function=smooth)
    meteor  = float(np.mean([meteor_score([rt_flat[i]], pt[i]) for i in range(len(preds))]))
    rouge_l = float(np.mean([
        rouge.score(preds[i], refs[i])["rougeL"].fmeasure for i in range(len(preds))
    ]))
    cider_scorer = Cider()
    gts  = {i: [refs[i]]  for i in range(len(refs))}
    res  = {i: [preds[i]] for i in range(len(preds))}
    cs, _ = cider_scorer.compute_score(gts, res)
    return {"BLEU": bleu, "METEOR": meteor, "ROUGE-L": rouge_l, "CIDEr": float(cs)}

# ========================= MAIN FL LOOP =========================
print("\n" + "="*70)
print("STARTING FLORENCE-2 FEDERATED TRAINING")
print(f"Run ID : run_{RUN_TIMESTAMP}")
print("="*70)

BEST_CIDER       = -1.0
BEST_ROUND       = -1
patience_counter = 0
all_metrics      = []

for round_num in range(ROUNDS):
    progress = round_num / max(1, ROUNDS - 1)
    round_lr = LEARNING_RATE * max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))

    print(f"\n{'='*70}")
    print(f"Round {round_num+1}/{ROUNDS}  |  LR: {round_lr:.2e}  |  "
          f"Best CIDEr: {BEST_CIDER:.4f}  |  GPU: {gpu_mem_str()}")
    print("="*70)

    client_updates = []

    for cid, client_ds in enumerate(client_datasets):
        print(f"\n  ── Client {cid+1}/{NUM_CLIENTS}  "
              f"({CLIENT_NAMES[cid]}, {len(client_ds)} samples) ──")

        # Snapshot global LoRA weights on CPU
        original_trainable = {
            name: param.clone().cpu()
            for name, param in global_model.named_parameters()
            if param.requires_grad
        }

        local_train(
            global_model, client_ds, LOCAL_EPOCHS, round_lr,
            original_trainable,
            round_num=round_num + 1,
            client_id=cid + 1,
        )

        # Collect updated weights
        new_trainable = {
            name: param.clone().cpu()
            for name, param in global_model.named_parameters()
            if param.requires_grad
        }
        client_updates.append((new_trainable, len(client_ds)))

        # Reset to global before next client
        for name, param in global_model.named_parameters():
            if param.requires_grad:
                param.data.copy_(original_trainable[name].to(param.device))

        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Client {cid+1} done | GPU after cleanup: {gpu_mem_str()}")

    # ── FedAvg ───────────────────────────────────────────────────────────────
    print("\n  ── FedAvg aggregation ──")
    total_size = sum(size for _, size in client_updates)
    first_keys = list(client_updates[0][0].keys())

    agg_bar = tqdm(first_keys, desc="  Aggregating weights",
                   dynamic_ncols=True, leave=False)
    for key in agg_bar:
        avg_tensor = sum(
            state[key] * (size / total_size) for state, size in client_updates
        )
        for name, param in global_model.named_parameters():
            if name == key:
                param.data.copy_(avg_tensor.to(param.device))
                break
        agg_bar.set_postfix(param=key[-35:])
    print(f"  Aggregation done | GPU: {gpu_mem_str()}")

    # ── Evaluation ───────────────────────────────────────────────────────────
    print("\n  ── Evaluation on test set ──")
    preds, refs = generate_predictions(
        global_model, test_dataset, processor, max_samples=200
    )
    metrics = evaluate_captions(preds, refs)
    all_metrics.append({"round": round_num + 1, **metrics})

    # Live CSV
    pd.DataFrame(all_metrics).to_csv(
        os.path.join(SCORES_DIR, "round_metrics.csv"), index=False
    )

    print(f"\n  ── Round {round_num+1} Results ──")
    print(f"  BLEU    = {metrics['BLEU']:.4f}")
    print(f"  METEOR  = {metrics['METEOR']:.4f}")
    print(f"  ROUGE-L = {metrics['ROUGE-L']:.4f}")
    print(f"  CIDEr   = {metrics['CIDEr']:.4f}")

    # ── Save best ────────────────────────────────────────────────────────────
    if metrics["CIDEr"] > BEST_CIDER:
        BEST_CIDER       = metrics["CIDEr"]
        BEST_ROUND       = round_num + 1
        patience_counter = 0
        best_path        = os.path.join(CHECKPOINT_DIR, "best")
        os.makedirs(best_path, exist_ok=True)
        global_model.save_pretrained(best_path)
        processor.save_pretrained(best_path)
        print(f"\n  *** NEW BEST  CIDEr={BEST_CIDER:.4f} "
              f"(round {BEST_ROUND}) → {best_path}")
    else:
        patience_counter += 1
        print(f"\n  No improvement ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print("  Early stopping triggered.")
            break

    gc.collect()
    torch.cuda.empty_cache()

# ========================= FINAL EVALUATION =========================
print("\n" + "="*70)
print("FINAL EVALUATION  (full test set)")
print("="*70)

print("Loading best checkpoint...")
base_final  = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    device_map={"": DEVICE},
    trust_remote_code=True,
)
best_path   = os.path.join(CHECKPOINT_DIR, "best")
final_model = PeftModel.from_pretrained(base_final, best_path, is_trainable=False)

preds, refs   = generate_predictions(
    final_model, test_dataset, processor, max_samples=len(test_dataset)
)
final_metrics = evaluate_captions(preds, refs)

print(f"\n  BLEU    = {final_metrics['BLEU']:.4f}")
print(f"  METEOR  = {final_metrics['METEOR']:.4f}")
print(f"  ROUGE-L = {final_metrics['ROUGE-L']:.4f}")
print(f"  CIDEr   = {final_metrics['CIDEr']:.4f}")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "run_id"           : f"run_{RUN_TIMESTAMP}",
    "model"            : MODEL_NAME,
    "task"             : CAPTION_TASK,
    "precision"        : "fp16_no_quantization",
    "best_round"       : BEST_ROUND,
    "best_cider"       : BEST_CIDER,
    "final_metrics"    : final_metrics,
    "all_round_metrics": all_metrics,
}
with open(os.path.join(SCORES_DIR, "final_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print("TRAINING COMPLETE!")
print(f"Run ID          : run_{RUN_TIMESTAMP}")
print(f"Run directory   : {OUTPUT_DIR}")
print(f"Best checkpoint : {os.path.join(CHECKPOINT_DIR, 'best')}")
print(f"Scores          : {SCORES_DIR}")
print(f"  • round_metrics.csv")
print(f"  • final_results.json")
print(f"  • run_config.json")
print(f"Best CIDEr      : {BEST_CIDER:.4f}  (round {BEST_ROUND})")
print("="*70)


"""
python florence_fed_captioning.py \
  --data_dir  /scratch/dharmendra.rs.phy23.itbhu/datasets/florence_data/fed_input_data \
  --output_dir /home/dharmendra.rs.phy23.itbhu/asmit_2/IIT_H_internship/Florence/Output
"""