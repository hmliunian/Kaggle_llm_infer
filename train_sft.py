"""
Nemotron-3-Nano-30B-A3B LoRA SFT Training Script
M2-M5: Full SFT training with stratified validation and per-family evaluation
"""

import os
import re
import subprocess
import math
import random
from pathlib import Path
from collections import Counter, defaultdict

import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemotron_cache_compat import patch_nemotron_h_cache_compat

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get("MODEL_PATH", "/data2/yaoxuran/models")
TRAIN_CSV = os.environ.get("TRAIN_CSV", str(BASE_DIR / "train.csv"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(BASE_DIR / "output_adapter"))
SUBMISSION_DIR = os.environ.get("SUBMISSION_DIR", str(BASE_DIR))

# Training hyperparams
LORA_RANK = 32
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
MAX_SEQ_LEN = 512  # sequences are short (~127 tokens avg, 267 max)
BATCH_SIZE = 4  # increase to improve GPU utilization
GRAD_ACCUM_STEPS = 8  # effective batch = 4*8 = 32 (same as before)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03

# Validation
VAL_RATIO = 0.1
VAL_MAX_SAMPLES = 20  # max samples to evaluate (generation is slow without KV cache)
EVAL_EVERY_STEPS = 100  # evaluate every N optimizer steps
SEED = 42

# GPU selection
GPU_ID = 6

SYSTEM_PROMPT = (
    "You are a precise reasoning model. "
    "Infer the hidden rule from the examples and answer the final query. "
    "Always put only the final answer inside \\boxed{}."
)

NUMERIC_FAMILIES = {"gravity_formula", "unit_conversion"}


# ─── Task Family Classifier ──────────────────────────────────────────────────
def classify_prompt(prompt: str) -> str:
    s = prompt.lower()
    if "secret bit manipulation" in s:
        return "bit_manipulation"
    if "d = 0.5*g*t^2" in s or "falling distance" in s:
        return "gravity_formula"
    if "convert the following measurement" in s or ("measurement" in s and "becomes" in s):
        return "unit_conversion"
    if "decrypt" in s:
        return "text_decryption"
    if "wonderland numeral" in s or "write the number" in s:
        return "numeral_system"
    if "determine the result for" in s:
        return "equation_symbol_transformation"
    return "unknown"


# ─── Answer Extraction ────────────────────────────────────────────────────────
def extract_boxed(text: str):
    matches = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def numeric_equal(pred, gold, rel_tol=1e-3, abs_tol=1e-4):
    try:
        p = float(pred)
        g = float(gold)
    except Exception:
        return False
    return abs(p - g) <= max(abs_tol, rel_tol * max(abs(g), 1.0))


# ─── Data Formatting ─────────────────────────────────────────────────────────
def format_example(tokenizer, prompt, answer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"The final answer is \\boxed{{{answer}}}."},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
    except Exception:
        return (
            f"System: {SYSTEM_PROMPT}\n\n"
            f"User:\n{prompt}\n\n"
            f"Assistant:\nThe final answer is \\boxed{{{answer}}}."
        )


def format_inference_prompt(tokenizer, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return f"System: {SYSTEM_PROMPT}\n\nUser:\n{prompt}\n\nAssistant:\n"


# ─── Stratified Split ────────────────────────────────────────────────────────
def stratified_split(df, val_ratio=0.1, seed=42):
    """Split dataframe into train/val stratified by task family."""
    rng = random.Random(seed)
    rows = df.to_dicts()

    # Group by family
    family_groups = defaultdict(list)
    for row in rows:
        family = classify_prompt(row["prompt"])
        family_groups[family].append(row)

    train_rows, val_rows = [], []
    for family, group in family_groups.items():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_ratio))
        val_rows.extend(group[:n_val])
        train_rows.extend(group[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    train_df = pl.DataFrame(train_rows)
    val_df = pl.DataFrame(val_rows)
    return train_df, val_df


# ─── Dataset ─────────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=2048):
        self.rows = df.to_dicts()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = format_example(self.tokenizer, row["prompt"], row["answer"])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc


def collate_fn(features, tokenizer):
    # Separate labels before padding (tokenizer.pad doesn't handle labels)
    labels = [f.pop("labels") for f in features]
    batch = tokenizer.pad(features, padding=True, return_tensors="pt")
    # Pad labels manually with -100
    max_len = batch["input_ids"].shape[1]
    padded_labels = []
    for lab in labels:
        pad_len = max_len - len(lab)
        padded_labels.append(lab + [-100] * pad_len)
    batch["labels"] = torch.tensor(padded_labels)
    return batch


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(model, tokenizer, val_df, max_samples=100, step_label=""):
    model.eval()
    # Disable gradient checkpointing for generation
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    samples = val_df.to_dicts()
    if len(samples) > max_samples:
        rng = random.Random(SEED)
        samples = rng.sample(samples, max_samples)

    correct = 0
    total = 0
    family_correct = defaultdict(int)
    family_total = defaultdict(int)
    boxed_count = 0

    first_device = next(model.parameters()).device

    for row in tqdm(samples, desc=f"Eval {step_label}", leave=False):
        prompt_text = format_inference_prompt(tokenizer, row["prompt"])
        enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        enc = {k: v.to(first_device) for k, v in enc.items()}

        with torch.no_grad():
            input_len = enc["input_ids"].shape[1]
            out = model.generate(
                **enc,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        decoded = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        pred = extract_boxed(decoded)
        gold = str(row["answer"]).strip()
        family = classify_prompt(row["prompt"])

        family_total[family] += 1
        total += 1

        if pred is not None:
            boxed_count += 1
            if family in NUMERIC_FAMILIES:
                is_correct = numeric_equal(pred, gold)
            else:
                is_correct = pred.strip() == gold
        else:
            is_correct = False

        if is_correct:
            correct += 1
            family_correct[family] += 1

    acc = correct / max(total, 1)
    boxed_rate = boxed_count / max(total, 1)

    print(f"\n{'='*60}")
    print(f"Evaluation @ {step_label}")
    print(f"  Overall accuracy: {correct}/{total} = {acc:.4f}")
    print(f"  Boxed answer rate: {boxed_count}/{total} = {boxed_rate:.4f}")
    for fam in sorted(family_total.keys()):
        fc = family_correct[fam]
        ft = family_total[fam]
        print(f"  {fam}: {fc}/{ft} = {fc/max(ft,1):.4f}")
    print(f"{'='*60}\n")

    model.train()
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    return acc


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    # Optimize CUDA memory and compute
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"[1/8] Loading train.csv ...")
    full_df = pl.read_csv(TRAIN_CSV)
    print(f"  Total rows: {len(full_df)}")

    # Task family distribution
    families = [classify_prompt(r["prompt"]) for r in full_df.to_dicts()]
    dist = Counter(families)
    print(f"  Task family distribution:")
    for k, v in sorted(dist.items()):
        print(f"    {k}: {v}")

    print(f"\n[2/8] Stratified train/val split ({1-VAL_RATIO:.0%}/{VAL_RATIO:.0%}) ...")
    train_df, val_df = stratified_split(full_df, val_ratio=VAL_RATIO, seed=SEED)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}")

    # Val family distribution
    val_families = [classify_prompt(r["prompt"]) for r in val_df.to_dicts()]
    val_dist = Counter(val_families)
    print(f"  Val family distribution:")
    for k, v in sorted(val_dist.items()):
        print(f"    {k}: {v}")

    print(f"\n[3/8] Loading tokenizer from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Show a sample formatted example
    sample_row = train_df.to_dicts()[0]
    sample_text = format_example(tokenizer, sample_row["prompt"], sample_row["answer"])
    print(f"\n  Sample formatted text (first 500 chars):")
    print(f"  {sample_text[:500]}")
    print()

    print(f"[4/8] Loading model from {MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print(f"  Model loaded.")

    if patch_nemotron_h_cache_compat(model):
        print("  Patched Nemotron-H generation cache compatibility.")

    print(f"\n[5/8] Attaching LoRA adapter (rank={LORA_RANK}) ...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"\n[6/8] Preparing dataset ...")
    train_dataset = SFTDataset(train_df, tokenizer, max_length=MAX_SEQ_LEN)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda feats: collate_fn(feats, tokenizer),
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Baseline eval (now feasible with fixed cache + transformers 4.55)
    print(f"\n[6.5/8] Baseline evaluation (before training) ...")
    evaluate(model, tokenizer, val_df, max_samples=VAL_MAX_SAMPLES, step_label="step=0 (baseline)")

    print(f"\n[7/8] Training ...")
    model.train()
    # Gradient checkpointing: disable for now to reduce overhead
    # (model fits in 80GB with batch_size=4, seq_len=512)
    # If OOM, re-enable and reduce batch_size
    print("  Gradient checkpointing disabled (to reduce overhead).")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    # LR scheduler: linear warmup + cosine decay
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    print(f"  Total optimizer steps: {total_steps}, warmup: {warmup_steps}")

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    global_step = 0
    best_acc = 0.0
    optimizer.zero_grad()
    running_loss = 0.0

    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for step, batch in enumerate(pbar):
            first_device = next(model.parameters()).device
            batch = {k: v.to(first_device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            running_loss += loss.item()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                avg_loss = running_loss
                running_loss = 0.0

                current_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    step=global_step,
                )

                # Periodic evaluation
                if global_step % EVAL_EVERY_STEPS == 0:
                    acc = evaluate(
                        model, tokenizer, val_df,
                        max_samples=VAL_MAX_SAMPLES,
                        step_label=f"step={global_step}",
                    )
                    if acc > best_acc:
                        best_acc = acc
                        # Save best checkpoint
                        best_dir = OUTPUT_DIR + "_best"
                        os.makedirs(best_dir, exist_ok=True)
                        model.save_pretrained(best_dir)
                        print(f"  New best accuracy: {best_acc:.4f}, saved to {best_dir}")
                    model.train()

        # End-of-epoch evaluation
        acc = evaluate(
            model, tokenizer, val_df,
            max_samples=VAL_MAX_SAMPLES,
            step_label=f"epoch={epoch}, step={global_step}",
        )
        if acc > best_acc:
            best_acc = acc
            best_dir = OUTPUT_DIR + "_best"
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            print(f"  New best accuracy: {best_acc:.4f}, saved to {best_dir}")

    print(f"\n  Training complete. Total optimizer steps: {global_step}")
    print(f"  Best validation accuracy: {best_acc:.4f}")

    # ─── Save final adapter ───
    print(f"\n[8/8] Saving final adapter to {OUTPUT_DIR} ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)

    # Use best checkpoint for submission if it exists
    best_dir = OUTPUT_DIR + "_best"
    submit_dir = best_dir if os.path.exists(best_dir) else OUTPUT_DIR
    print(f"  Using {submit_dir} for submission (best checkpoint)")

    # Verify files
    adapter_config = os.path.join(submit_dir, "adapter_config.json")
    adapter_weights = os.path.join(submit_dir, "adapter_model.safetensors")
    assert os.path.exists(adapter_config), f"Missing {adapter_config}"
    assert os.path.exists(adapter_weights), f"Missing {adapter_weights}"
    print(f"  adapter_config.json: {os.path.getsize(adapter_config)} bytes")
    print(f"  adapter_model.safetensors: {os.path.getsize(adapter_weights) / 1e6:.1f} MB")

    # Package submission.zip
    zip_path = os.path.join(SUBMISSION_DIR, "submission.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    subprocess.run(
        f"cd {submit_dir} && zip -r {zip_path} adapter_config.json adapter_model.safetensors",
        shell=True, check=True,
    )
    print(f"  submission.zip: {os.path.getsize(zip_path) / 1e6:.1f} MB")

    # List zip contents
    result = subprocess.run(f"unzip -l {zip_path}", shell=True, capture_output=True, text=True)
    print(f"\n  submission.zip contents:")
    print(result.stdout)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE — Best val accuracy: {best_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
