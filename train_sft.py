"""
Nemotron-3-Nano-30B-A3B LoRA SFT Training Script
M2-M5: Full SFT training with stratified validation and per-family evaluation
"""

import os
import re
import csv
import json
import subprocess
import math
import random
import shutil
import itertools
from contextlib import nullcontext
from pathlib import Path
from collections import Counter, defaultdict

import polars as pl
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from nemotron_cache_compat import patch_nemotron_h_cache_compat


def env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def log_main(*args, **kwargs):
    if IS_MAIN_PROCESS:
        print(*args, **kwargs)


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def barrier():
    if IS_DISTRIBUTED:
        dist.barrier()


def broadcast_float(value: float, device):
    if not IS_DISTRIBUTED:
        return value
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    dist.broadcast(tensor, src=0)
    return float(tensor.item())


def average_float(value: float, device):
    if not IS_DISTRIBUTED:
        return value
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return float(tensor.item())

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get("MODEL_PATH", "/data2/yaoxuran/models")
TRAIN_CSV = os.environ.get("TRAIN_CSV", str(BASE_DIR / "data" / "train.csv"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(BASE_DIR / "outputs" / "adapter"))
SUBMISSION_DIR = os.environ.get("SUBMISSION_DIR", str(BASE_DIR / "outputs" / "artifacts"))
SMOKE_MODE = env_bool("SMOKE_MODE", False)
CHECKPOINT_DIR = os.environ.get(
    "CHECKPOINT_DIR",
    str(Path(OUTPUT_DIR).resolve().parent / "checkpoints"),
)
CHECKPOINT_EVERY_STEPS = env_int("CHECKPOINT_EVERY_STEPS", 1 if SMOKE_MODE else 100)
KEEP_LAST_CHECKPOINTS = env_int("KEEP_LAST_CHECKPOINTS", 2)
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME_FROM_CHECKPOINT", "").strip()

# Training hyperparams
LORA_RANK = env_int("LORA_RANK", 32)
LORA_ALPHA = env_int("LORA_ALPHA", 16)
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
MAX_SEQ_LEN = env_int("MAX_SEQ_LEN", 512)  # sequences are short (~127 tokens avg, 267 max)
BATCH_SIZE = env_int("BATCH_SIZE", 1)
GRAD_ACCUM_STEPS = env_int("GRAD_ACCUM_STEPS", 1 if SMOKE_MODE else 32)
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-4"))
NUM_EPOCHS = env_int("NUM_EPOCHS", 1 if SMOKE_MODE else 3)
WARMUP_RATIO = 0.03

# Validation
VAL_RATIO = 0.1
VAL_MAX_SAMPLES = env_int("VAL_MAX_SAMPLES", 2 if SMOKE_MODE else 100)
EVAL_EVERY_STEPS = env_int("EVAL_EVERY_STEPS", 1 if SMOKE_MODE else 100)
EVAL_MAX_NEW_TOKENS = env_int("EVAL_MAX_NEW_TOKENS", 16 if SMOKE_MODE else 128)
EVAL_OUTPUT_DIR = os.environ.get(
    "EVAL_OUTPUT_DIR",
    str(Path(OUTPUT_DIR).resolve().parent / "eval"),
)
EVAL_SAVE_DETAILS = env_bool("EVAL_SAVE_DETAILS", True)
RESPONSE_ONLY_LOSS = env_bool("RESPONSE_ONLY_LOSS", True)
TRAIN_FINAL_ANSWER_PREFILL = env_bool("TRAIN_FINAL_ANSWER_PREFILL", True)
INFERENCE_FINAL_ANSWER_PREFILL = env_bool("INFERENCE_FINAL_ANSWER_PREFILL", True)
INFERENCE_STOP_AFTER_BOXED = env_bool("INFERENCE_STOP_AFTER_BOXED", True)
TRAIN_ROW_LIMIT = env_int("TRAIN_ROW_LIMIT", 100 if SMOKE_MODE else 0)
MAX_TRAIN_STEPS = env_int("MAX_TRAIN_STEPS", 2 if SMOKE_MODE else 0)
ENABLE_BASELINE_EVAL = env_bool("ENABLE_BASELINE_EVAL", True)
NUM_WORKERS = env_int("NUM_WORKERS", 0 if SMOKE_MODE else 4)
SEED = 42

# GPU selection
GPU_ID = env_int("GPU_ID", 6)
LOCAL_RANK = env_int("LOCAL_RANK", 0)
RANK = env_int("RANK", 0)
WORLD_SIZE = env_int("WORLD_SIZE", 1)
IS_DISTRIBUTED = WORLD_SIZE > 1
IS_MAIN_PROCESS = RANK == 0

# W&B logging
WANDB_ENABLED = env_bool("WANDB_ENABLED", False)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "llm-infer-sft")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "").strip()
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", "").strip()
WANDB_RUN_ID = os.environ.get("WANDB_RUN_ID", "").strip()
WANDB_RESUME = os.environ.get("WANDB_RESUME", "allow" if WANDB_RUN_ID else "").strip()
WANDB_MODE = os.environ.get("WANDB_MODE", "").strip()
WANDB_DIR = os.environ.get(
    "WANDB_DIR",
    str(Path(OUTPUT_DIR).resolve().parent / "wandb"),
)
WANDB_LOG_ARTIFACTS = env_bool("WANDB_LOG_ARTIFACTS", False)

SYSTEM_PROMPT = (
    "You are a precise reasoning model. "
    "Infer the hidden rule from the examples and answer the final query. "
    "Always put only the final answer inside \\boxed{}."
)
FINAL_ANSWER_PREFILL_TEXT = os.environ.get(
    "FINAL_ANSWER_PREFILL_TEXT",
    "</think>\nThe final answer is \\boxed{",
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


def format_answer_text(answer):
    return f"{FINAL_ANSWER_PREFILL_TEXT}{answer}}}."


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


def format_training_parts(tokenizer, prompt, answer):
    if TRAIN_FINAL_ANSWER_PREFILL:
        prompt_text = format_inference_prompt(tokenizer, prompt)
        return prompt_text, format_answer_text(answer)
    return "", format_example(tokenizer, prompt, answer)


class StopAfterBoxClose(StoppingCriteria):
    def __init__(self, tokenizer, start_len: int):
        self.tokenizer = tokenizer
        self.start_len = start_len

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0][self.start_len :]
        if generated.numel() == 0:
            return False
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return "}" in text


def generate_completion(model, tokenizer, prompt, max_new_tokens=128, use_cache=True):
    prompt_text = format_inference_prompt(tokenizer, prompt)
    answer_prefix = FINAL_ANSWER_PREFILL_TEXT if INFERENCE_FINAL_ANSWER_PREFILL else ""
    prompt_text = prompt_text + answer_prefix
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    first_device = next(model.parameters()).device
    enc = {k: v.to(first_device) for k, v in enc.items()}
    input_len = enc["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=use_cache,
    )
    if INFERENCE_FINAL_ANSWER_PREFILL and INFERENCE_STOP_AFTER_BOXED:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [StopAfterBoxClose(tokenizer, input_len)]
        )

    try:
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
    except RuntimeError as exc:
        msg = str(exc)
        if use_cache and ("weight must have shape" in msg or "causal_conv1d" in msg):
            print("  Generation cache fast path failed; retrying with use_cache=False.")
            gen_kwargs["use_cache"] = False
            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
        else:
            raise

    generated = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    if INFERENCE_FINAL_ANSWER_PREFILL and INFERENCE_STOP_AFTER_BOXED:
        close_idx = generated.find("}")
        if close_idx >= 0:
            generated = generated[: close_idx + 1]
    return answer_prefix + generated


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
        prompt_text, answer_text = format_training_parts(
            self.tokenizer,
            row["prompt"],
            row["answer"],
        )
        text = prompt_text + answer_text
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        if RESPONSE_ONLY_LOSS and prompt_text:
            prompt_enc = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )
            prompt_len = min(len(prompt_enc["input_ids"]), len(enc["input_ids"]))
            enc["labels"] = [-100] * prompt_len + enc["input_ids"][prompt_len:].copy()
            if all(label == -100 for label in enc["labels"]):
                enc["labels"] = enc["input_ids"].copy()
        else:
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


# ─── Checkpointing ────────────────────────────────────────────────────────────
def resolve_resume_checkpoint(path_value: str, checkpoint_dir: str) -> Path | None:
    if not path_value:
        return None
    if path_value.lower() == "latest":
        latest_file = Path(checkpoint_dir) / "latest_checkpoint.txt"
        if not latest_file.exists():
            raise FileNotFoundError(f"Missing latest checkpoint pointer: {latest_file}")
        path_value = latest_file.read_text(encoding="utf-8").strip()
    checkpoint_path = Path(path_value).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not (checkpoint_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing adapter_config.json in checkpoint: {checkpoint_path}")
    return checkpoint_path


def checkpoint_sort_key(path: Path):
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def rotate_checkpoints(root_dir: Path, keep_last: int):
    if keep_last <= 0 or not root_dir.exists():
        return
    checkpoints = sorted(
        [p for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=checkpoint_sort_key,
    )
    for old_dir in checkpoints[:-keep_last]:
        shutil.rmtree(old_dir)
        log_main(f"  Removed old checkpoint: {old_dir}")


def save_training_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_root: str,
    epoch: int,
    next_step: int,
    global_step: int,
    best_acc: float,
    running_loss: float,
    total_steps: int,
):
    root_dir = Path(checkpoint_root)
    root_dir.mkdir(parents=True, exist_ok=True)
    final_dir = root_dir / f"checkpoint-{global_step:06d}"
    tmp_dir = root_dir / f".tmp-checkpoint-{global_step:06d}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if final_dir.exists():
        shutil.rmtree(final_dir)

    log_main(f"  Saving training checkpoint to {final_dir} ...")
    unwrap_model(model).save_pretrained(tmp_dir)
    state = {
        "epoch": epoch,
        "next_step": next_step,
        "global_step": global_step,
        "best_acc": best_acc,
        "running_loss": running_loss,
        "total_steps": total_steps,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": {
            "train_csv": TRAIN_CSV,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "max_seq_len": MAX_SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "max_train_steps": MAX_TRAIN_STEPS,
            "seed": SEED,
        },
    }
    torch.save(state, tmp_dir / "training_state.pt")
    tmp_dir.rename(final_dir)
    (root_dir / "latest_checkpoint.txt").write_text(str(final_dir) + "\n", encoding="utf-8")
    rotate_checkpoints(root_dir, KEEP_LAST_CHECKPOINTS)
    log_main(f"  Checkpoint saved: {final_dir}")
    return final_dir


def load_training_state(checkpoint_path: Path, optimizer, scheduler, device):
    state_path = checkpoint_path / "training_state.pt"
    if not state_path.exists():
        log_main(f"  No training_state.pt found in {checkpoint_path}; adapter weights loaded only.")
        return {
            "epoch": 0,
            "next_step": 0,
            "global_step": 0,
            "best_acc": 0.0,
            "running_loss": 0.0,
        }

    state = torch.load(state_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    if state.get("python_random_state") is not None:
        random.setstate(state["python_random_state"])
    if state.get("torch_rng_state") is not None:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    cuda_rng_state_all = state.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_rng_state_all is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state_all)

    return {
        "epoch": int(state.get("epoch", 0)),
        "next_step": int(state.get("next_step", 0)),
        "global_step": int(state.get("global_step", 0)),
        "best_acc": float(state.get("best_acc", 0.0)),
        "running_loss": float(state.get("running_loss", 0.0)),
    }


# ─── W&B Logging ──────────────────────────────────────────────────────────────
class WandbLogger:
    def __init__(self):
        self.enabled = WANDB_ENABLED and IS_MAIN_PROCESS
        self.wandb = None
        self.run = None
        self.log_failed = False

    def init(self, config: dict):
        if not self.enabled:
            return

        wandb_base_dir = Path(WANDB_DIR)
        wandb_base_dir.mkdir(parents=True, exist_ok=True)
        for env_name, dirname in {
            "WANDB_CACHE_DIR": "cache",
            "WANDB_CONFIG_DIR": "config",
            "WANDB_DATA_DIR": "data",
        }.items():
            env_path = Path(os.environ.get(env_name, str(wandb_base_dir / dirname)))
            env_path.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault(env_name, str(env_path))

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "WANDB_ENABLED=1 but the wandb package is not installed. "
                "Install it with `.venv/bin/python -m pip install wandb`, "
                "or set WANDB_ENABLED=0."
            ) from exc

        run_name = WANDB_RUN_NAME or Path(OUTPUT_DIR).resolve().parent.name
        init_kwargs = {
            "project": WANDB_PROJECT,
            "name": run_name,
            "config": config,
            "dir": WANDB_DIR,
        }
        if WANDB_ENTITY:
            init_kwargs["entity"] = WANDB_ENTITY
        if WANDB_RUN_ID:
            init_kwargs["id"] = WANDB_RUN_ID
        if WANDB_RESUME:
            init_kwargs["resume"] = WANDB_RESUME
        if WANDB_MODE:
            init_kwargs["mode"] = WANDB_MODE

        self.wandb = wandb
        self.run = wandb.init(**init_kwargs)
        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")
        wandb.define_metric("eval/*", step_metric="train/global_step")
        wandb.define_metric("eval_family/*", step_metric="train/global_step")
        wandb.define_metric("checkpoint/*", step_metric="train/global_step")
        wandb.define_metric("final/*", step_metric="train/global_step")
        print(f"  W&B run: {self.run.url or self.run.name}")

    def log(self, metrics: dict, step: int | None = None):
        if self.run is None:
            return
        try:
            self.wandb.log(metrics, step=step)
        except Exception as exc:
            if not self.log_failed:
                print(f"  W&B logging failed; continuing training without blocking: {exc}")
                self.log_failed = True

    def log_data_profile(self, train_rows: int, val_rows: int, family_dist: Counter, val_family_dist: Counter):
        metrics = {
            "data/train_rows": train_rows,
            "data/val_rows": val_rows,
        }
        for family, count in sorted(family_dist.items()):
            metrics[f"data/family/{family}"] = count
        for family, count in sorted(val_family_dist.items()):
            metrics[f"data/val_family/{family}"] = count
        self.log(metrics, step=0)

    def log_eval(
        self,
        step_label: str,
        accuracy: float,
        boxed_rate: float,
        correct: int,
        total: int,
        family_summary: dict,
        step: int | None,
    ):
        if self.run is None:
            return

        metrics = {
            "train/global_step": step or 0,
            "eval/accuracy": accuracy,
            "eval/boxed_rate": boxed_rate,
            "eval/correct": correct,
            "eval/total": total,
        }
        for family, fam_metrics in family_summary.items():
            prefix = f"eval_family/{family}"
            metrics[f"{prefix}/accuracy"] = fam_metrics["accuracy"]
            metrics[f"{prefix}/boxed_rate"] = fam_metrics["boxed_rate"]
            metrics[f"{prefix}/correct"] = fam_metrics["correct"]
            metrics[f"{prefix}/total"] = fam_metrics["total"]
        self.run.summary["last_eval_label"] = step_label
        self.run.summary["last_eval_accuracy"] = accuracy
        self.log(metrics, step=step)

    def log_checkpoint(self, checkpoint_path: Path, global_step: int, best_acc: float):
        if self.run is None:
            return
        self.run.summary["latest_checkpoint"] = str(checkpoint_path)
        self.log(
            {
                "train/global_step": global_step,
                "checkpoint/saved": 1,
                "checkpoint/best_accuracy": best_acc,
            },
            step=global_step,
        )
        if WANDB_LOG_ARTIFACTS:
            self._log_artifact(checkpoint_path, "checkpoint", f"checkpoint-{global_step:06d}", ["latest"])

    def log_final(self, global_step: int, best_acc: float, submit_dir: str, zip_path: str):
        if self.run is None:
            return
        adapter_path = Path(submit_dir) / "adapter_model.safetensors"
        zip_file = Path(zip_path)
        metrics = {
            "train/global_step": global_step,
            "final/best_accuracy": best_acc,
        }
        if adapter_path.exists():
            metrics["final/adapter_size_mb"] = adapter_path.stat().st_size / 1e6
        if zip_file.exists():
            metrics["final/submission_zip_size_mb"] = zip_file.stat().st_size / 1e6
        self.run.summary["best_val_accuracy"] = best_acc
        self.run.summary["final_global_step"] = global_step
        self.run.summary["submit_dir"] = submit_dir
        self.run.summary["submission_zip"] = zip_path
        self.log(metrics, step=global_step)
        if WANDB_LOG_ARTIFACTS:
            self._log_artifact(Path(submit_dir), "adapter", "adapter-final", ["final"])
            self._log_artifact(zip_file, "submission", "submission-final", ["final"])

    def _log_artifact(self, path: Path, artifact_type: str, name: str, aliases: list[str]):
        if self.run is None or not path.exists():
            return
        artifact = self.wandb.Artifact(name=name, type=artifact_type)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        self.run.log_artifact(artifact, aliases=aliases)

    def finish(self):
        if self.run is not None:
            self.wandb.finish()


# ─── Evaluation ───────────────────────────────────────────────────────────────
def safe_eval_label(step_label: str) -> str:
    label = step_label or "eval"
    label = label.replace("=", "-").replace(",", "")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    return label or "eval"


def evaluate(
    model,
    tokenizer,
    val_df,
    max_samples=100,
    step_label="",
    max_new_tokens=128,
    wandb_logger: WandbLogger | None = None,
    wandb_step: int | None = None,
):
    model.eval()
    # Generation and training own their checkpointing policy separately.
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    samples = val_df.to_dicts()
    val_size = len(samples)
    if max_samples > 0 and len(samples) > max_samples:
        rng = random.Random(SEED)
        samples = rng.sample(samples, max_samples)

    correct = 0
    total = 0
    family_correct = defaultdict(int)
    family_total = defaultdict(int)
    family_boxed = defaultdict(int)
    boxed_count = 0
    eval_records = []

    for sample_idx, row in enumerate(tqdm(samples, desc=f"Eval {step_label}", leave=False)):
        decoded = generate_completion(
            model,
            tokenizer,
            row["prompt"],
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        pred = extract_boxed(decoded)
        gold = str(row["answer"]).strip()
        family = classify_prompt(row["prompt"])

        family_total[family] += 1
        total += 1

        has_boxed = pred is not None
        if pred is not None:
            boxed_count += 1
            family_boxed[family] += 1
            if family in NUMERIC_FAMILIES:
                is_correct = numeric_equal(pred, gold)
            else:
                is_correct = pred.strip() == gold
        else:
            is_correct = False

        if is_correct:
            correct += 1
            family_correct[family] += 1

        eval_records.append(
            {
                "sample_index": sample_idx,
                "step_label": step_label,
                "family": family,
                "correct": bool(is_correct),
                "has_boxed_answer": bool(has_boxed),
                "gold": gold,
                "pred": pred,
                "prompt": row["prompt"],
                "decoded": decoded,
            }
        )

    acc = correct / max(total, 1)
    boxed_rate = boxed_count / max(total, 1)
    family_summary = {}
    for fam in sorted(family_total.keys()):
        ft = family_total[fam]
        fc = family_correct[fam]
        fb = family_boxed[fam]
        family_summary[fam] = {
            "correct": fc,
            "total": ft,
            "accuracy": fc / max(ft, 1),
            "boxed_count": fb,
            "boxed_rate": fb / max(ft, 1),
        }

    log_main(f"\n{'='*60}")
    log_main(f"Evaluation @ {step_label}")
    log_main(f"  Overall accuracy: {correct}/{total} = {acc:.4f}")
    log_main(f"  Boxed answer rate: {boxed_count}/{total} = {boxed_rate:.4f}")
    for fam, metrics in family_summary.items():
        log_main(
            f"  {fam}: {metrics['correct']}/{metrics['total']} = "
            f"{metrics['accuracy']:.4f}, boxed={metrics['boxed_count']}/{metrics['total']}"
        )

    if EVAL_SAVE_DETAILS:
        eval_dir = Path(EVAL_OUTPUT_DIR)
        eval_dir.mkdir(parents=True, exist_ok=True)
        stem = f"eval_{safe_eval_label(step_label)}"
        jsonl_path = eval_dir / f"{stem}.jsonl"
        csv_path = eval_dir / f"{stem}.csv"
        summary_path = eval_dir / f"{stem}_summary.json"

        with jsonl_path.open("w", encoding="utf-8") as f:
            for record in eval_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        csv_fields = [
            "sample_index",
            "step_label",
            "family",
            "correct",
            "has_boxed_answer",
            "gold",
            "pred",
            "prompt",
            "decoded",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(eval_records)

        summary = {
            "step_label": step_label,
            "val_size": val_size,
            "max_samples": max_samples,
            "evaluated_samples": total,
            "max_new_tokens": max_new_tokens,
            "correct": correct,
            "accuracy": acc,
            "boxed_count": boxed_count,
            "boxed_rate": boxed_rate,
            "family_summary": family_summary,
            "details_jsonl": str(jsonl_path),
            "details_csv": str(csv_path),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        log_main(f"  Eval details JSONL: {jsonl_path}")
        log_main(f"  Eval details CSV:   {csv_path}")
        log_main(f"  Eval summary JSON:  {summary_path}")
    log_main(f"{'='*60}\n")

    if wandb_logger is not None:
        wandb_logger.log_eval(
            step_label=step_label,
            accuracy=acc,
            boxed_rate=boxed_rate,
            correct=correct,
            total=total,
            family_summary=family_summary,
            step=wandb_step,
        )

    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return acc


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    if IS_DISTRIBUTED:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(LOCAL_RANK)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    # Optimize CUDA memory and compute
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    random.seed(SEED + RANK)
    torch.manual_seed(SEED + RANK)

    log_main("[0/8] Run config")
    log_main(f"  SMOKE_MODE={SMOKE_MODE}")
    log_main(
        f"  DISTRIBUTED={IS_DISTRIBUTED}, WORLD_SIZE={WORLD_SIZE}, "
        f"RANK={RANK}, LOCAL_RANK={LOCAL_RANK}, GPU_ID={GPU_ID}"
    )
    log_main(f"  LORA_RANK={LORA_RANK}, LORA_ALPHA={LORA_ALPHA}, LORA_DROPOUT={LORA_DROPOUT}")
    log_main(f"  MAX_SEQ_LEN={MAX_SEQ_LEN}, BATCH_SIZE={BATCH_SIZE}, GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS}")
    log_main(f"  NUM_EPOCHS={NUM_EPOCHS}, MAX_TRAIN_STEPS={MAX_TRAIN_STEPS}, TRAIN_ROW_LIMIT={TRAIN_ROW_LIMIT}")
    log_main(f"  VAL_MAX_SAMPLES={VAL_MAX_SAMPLES}, EVAL_EVERY_STEPS={EVAL_EVERY_STEPS}, EVAL_MAX_NEW_TOKENS={EVAL_MAX_NEW_TOKENS}")
    log_main(f"  EVAL_OUTPUT_DIR={EVAL_OUTPUT_DIR}")
    log_main(f"  EVAL_SAVE_DETAILS={EVAL_SAVE_DETAILS}")
    log_main(
        "  "
        f"RESPONSE_ONLY_LOSS={RESPONSE_ONLY_LOSS}, "
        f"TRAIN_FINAL_ANSWER_PREFILL={TRAIN_FINAL_ANSWER_PREFILL}, "
        f"INFERENCE_FINAL_ANSWER_PREFILL={INFERENCE_FINAL_ANSWER_PREFILL}, "
        f"INFERENCE_STOP_AFTER_BOXED={INFERENCE_STOP_AFTER_BOXED}"
    )
    log_main(f"  OUTPUT_DIR={OUTPUT_DIR}")
    log_main(f"  SUBMISSION_DIR={SUBMISSION_DIR}")
    log_main(f"  CHECKPOINT_DIR={CHECKPOINT_DIR}")
    log_main(f"  CHECKPOINT_EVERY_STEPS={CHECKPOINT_EVERY_STEPS}, KEEP_LAST_CHECKPOINTS={KEEP_LAST_CHECKPOINTS}")
    log_main(f"  RESUME_FROM_CHECKPOINT={RESUME_FROM_CHECKPOINT or '(none)'}")
    log_main(f"  WANDB_ENABLED={WANDB_ENABLED}, WANDB_PROJECT={WANDB_PROJECT}, WANDB_RUN_NAME={WANDB_RUN_NAME or '(auto)'}")
    log_main(f"  WANDB_MODE={WANDB_MODE or '(default)'}, WANDB_LOG_ARTIFACTS={WANDB_LOG_ARTIFACTS}")
    log_main()

    resume_checkpoint = resolve_resume_checkpoint(RESUME_FROM_CHECKPOINT, CHECKPOINT_DIR)
    wandb_logger = WandbLogger()
    wandb_logger.init(
        {
            "model_path": MODEL_PATH,
            "train_csv": TRAIN_CSV,
            "output_dir": OUTPUT_DIR,
            "submission_dir": SUBMISSION_DIR,
            "checkpoint_dir": CHECKPOINT_DIR,
            "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else "",
            "smoke_mode": SMOKE_MODE,
            "gpu_id": GPU_ID,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "max_seq_len": MAX_SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "val_ratio": VAL_RATIO,
            "val_max_samples": VAL_MAX_SAMPLES,
            "eval_every_steps": EVAL_EVERY_STEPS,
            "eval_max_new_tokens": EVAL_MAX_NEW_TOKENS,
            "eval_output_dir": EVAL_OUTPUT_DIR,
            "eval_save_details": EVAL_SAVE_DETAILS,
            "response_only_loss": RESPONSE_ONLY_LOSS,
            "train_final_answer_prefill": TRAIN_FINAL_ANSWER_PREFILL,
            "inference_final_answer_prefill": INFERENCE_FINAL_ANSWER_PREFILL,
            "inference_stop_after_boxed": INFERENCE_STOP_AFTER_BOXED,
            "train_row_limit": TRAIN_ROW_LIMIT,
            "max_train_steps": MAX_TRAIN_STEPS,
            "enable_baseline_eval": ENABLE_BASELINE_EVAL,
            "num_workers": NUM_WORKERS,
            "checkpoint_every_steps": CHECKPOINT_EVERY_STEPS,
            "keep_last_checkpoints": KEEP_LAST_CHECKPOINTS,
            "seed": SEED,
        }
    )

    log_main(f"[1/8] Loading train.csv ...")
    full_df = pl.read_csv(
        TRAIN_CSV,
        schema_overrides={
            "prompt": pl.String,
            "answer": pl.String,
        },
    )
    log_main(f"  Total rows: {len(full_df)}")

    # Task family distribution
    families = [classify_prompt(r["prompt"]) for r in full_df.to_dicts()]
    family_dist = Counter(families)
    log_main(f"  Task family distribution:")
    for k, v in sorted(family_dist.items()):
        log_main(f"    {k}: {v}")

    log_main(f"\n[2/8] Stratified train/val split ({1-VAL_RATIO:.0%}/{VAL_RATIO:.0%}) ...")
    train_df, val_df = stratified_split(full_df, val_ratio=VAL_RATIO, seed=SEED)
    if TRAIN_ROW_LIMIT > 0 and len(train_df) > TRAIN_ROW_LIMIT:
        train_df = train_df.head(TRAIN_ROW_LIMIT)
        log_main(f"  Smoke train row limit applied: {len(train_df)}")
    log_main(f"  Train: {len(train_df)}, Val: {len(val_df)}")

    # Val family distribution
    val_families = [classify_prompt(r["prompt"]) for r in val_df.to_dicts()]
    val_dist = Counter(val_families)
    log_main(f"  Val family distribution:")
    for k, v in sorted(val_dist.items()):
        log_main(f"    {k}: {v}")
    wandb_logger.log_data_profile(len(train_df), len(val_df), family_dist, val_dist)

    log_main(f"\n[3/8] Loading tokenizer from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log_main(f"  Vocab size: {tokenizer.vocab_size}")

    # Show a sample formatted example
    sample_row = train_df.to_dicts()[0]
    sample_prompt_text, sample_answer_text = format_training_parts(
        tokenizer,
        sample_row["prompt"],
        sample_row["answer"],
    )
    sample_text = sample_prompt_text + sample_answer_text
    log_main(f"\n  Sample formatted text (first 500 chars):")
    log_main(f"  {sample_text[:500]}")
    log_main()

    log_main(f"[4/8] Loading model from {MODEL_PATH} ...")
    model_device = LOCAL_RANK if IS_DISTRIBUTED else 0
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": model_device},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    log_main(f"  Model loaded.")

    if patch_nemotron_h_cache_compat(model):
        log_main("  Patched Nemotron-H generation cache compatibility.")

    if resume_checkpoint is not None:
        log_main(f"\n[5/8] Loading LoRA adapter from checkpoint {resume_checkpoint} ...")
        model = PeftModel.from_pretrained(model, str(resume_checkpoint), is_trainable=True)
    else:
        log_main(f"\n[5/8] Attaching LoRA adapter (rank={LORA_RANK}) ...")
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    if IS_MAIN_PROCESS:
        model.print_trainable_parameters()

    if IS_DISTRIBUTED:
        model = DistributedDataParallel(
            model,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
            find_unused_parameters=True,
        )

    log_main(f"\n[6/8] Preparing dataset ...")
    train_dataset = SFTDataset(train_df, tokenizer, max_length=MAX_SEQ_LEN)
    batches_per_epoch = math.ceil(math.ceil(len(train_dataset) / WORLD_SIZE) / BATCH_SIZE)

    def make_train_loader(epoch_idx: int):
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=WORLD_SIZE,
            rank=RANK,
            shuffle=True,
            seed=SEED,
        ) if IS_DISTRIBUTED else None
        if sampler is not None:
            sampler.set_epoch(epoch_idx)
        generator = None
        if sampler is None:
            generator = torch.Generator()
            generator.manual_seed(SEED + epoch_idx)
        return DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=sampler is None,
            sampler=sampler,
            generator=generator,
            collate_fn=lambda feats: collate_fn(feats, tokenizer),
            num_workers=NUM_WORKERS,
            pin_memory=True,
            **({"prefetch_factor": 2} if NUM_WORKERS > 0 else {}),
        )

    # Baseline eval (kept optional for smoke runs)
    if ENABLE_BASELINE_EVAL:
        log_main(f"\n[6.5/8] Baseline evaluation (before training) ...")
        acc = 0.0
        if IS_MAIN_PROCESS:
            acc = evaluate(
                unwrap_model(model),
                tokenizer,
                val_df,
                max_samples=VAL_MAX_SAMPLES,
                step_label="step=0 (baseline)",
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                wandb_logger=wandb_logger,
                wandb_step=0,
            )
        acc = broadcast_float(acc, torch.device("cuda", LOCAL_RANK if IS_DISTRIBUTED else 0))
        barrier()
        model.train()
    else:
        log_main(f"\n[6.5/8] Baseline evaluation skipped by ENABLE_BASELINE_EVAL=0")

    log_main(f"\n[7/8] Training ...")
    model.train()
    # Gradient checkpointing: disable for now to reduce overhead
    # (model fits in 80GB with batch_size=4, seq_len=512)
    # If OOM, re-enable and reduce batch_size
    log_main("  Gradient checkpointing disabled (to reduce overhead).")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.0,
        foreach=False,
    )

    # LR scheduler: linear warmup + cosine decay
    total_steps = max(1, math.ceil(batches_per_epoch / GRAD_ACCUM_STEPS) * NUM_EPOCHS)
    warmup_steps = min(total_steps - 1, max(1, int(total_steps * WARMUP_RATIO))) if total_steps > 1 else 0
    log_main(f"  Total optimizer steps: {total_steps}, warmup: {warmup_steps}")

    if warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine_t_max = max(1, total_steps - warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_t_max, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=1e-6)

    global_step = 0
    best_acc = 0.0
    start_epoch = 0
    start_step = 0
    optimizer.zero_grad()
    running_loss = 0.0
    stop_training = False

    if resume_checkpoint is not None:
        first_device = next(model.parameters()).device
        resume_state = load_training_state(resume_checkpoint, optimizer, scheduler, first_device)
        start_epoch = resume_state["epoch"]
        start_step = resume_state["next_step"]
        global_step = resume_state["global_step"]
        best_acc = resume_state["best_acc"]
        running_loss = resume_state["running_loss"]
        log_main(
            f"  Resumed from epoch={start_epoch}, next_batch_step={start_step}, "
            f"global_step={global_step}, best_acc={best_acc:.4f}"
        )
        if MAX_TRAIN_STEPS > 0 and global_step >= MAX_TRAIN_STEPS:
            log_main(f"  Resume checkpoint already reached MAX_TRAIN_STEPS={MAX_TRAIN_STEPS}.")
            stop_training = True

    for epoch in range(start_epoch, NUM_EPOCHS):
        if stop_training:
            break
        train_loader = make_train_loader(epoch)
        epoch_start_step = start_step if epoch == start_epoch else 0
        if epoch_start_step >= len(train_loader):
            continue
        train_iter = itertools.islice(train_loader, epoch_start_step, None)
        pbar = tqdm(
            train_iter,
            total=len(train_loader),
            initial=epoch_start_step,
            desc=f"Epoch {epoch}/{NUM_EPOCHS}",
            disable=not IS_MAIN_PROCESS,
        )
        for step, batch in enumerate(pbar, start=epoch_start_step):
            first_device = next(model.parameters()).device
            batch = {k: v.to(first_device) for k, v in batch.items()}

            sync_context = model.no_sync() if (
                IS_DISTRIBUTED and (step + 1) % GRAD_ACCUM_STEPS != 0
            ) else nullcontext()
            with sync_context:
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
                avg_loss = average_float(running_loss, first_device)
                running_loss = 0.0

                current_lr = optimizer.param_groups[0]["lr"]
                if IS_MAIN_PROCESS:
                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        lr=f"{current_lr:.2e}",
                        step=global_step,
                    )
                wandb_logger.log(
                    {
                        "train/global_step": global_step,
                        "train/loss": avg_loss,
                        "train/lr": current_lr,
                        "train/epoch": epoch,
                        "train/batch_step": step + 1,
                        "train/batch_progress": (step + 1) / max(len(train_loader), 1),
                        "train/best_accuracy": best_acc,
                    },
                    step=global_step,
                )

                # Periodic evaluation
                if global_step % EVAL_EVERY_STEPS == 0:
                    acc = 0.0
                    if IS_MAIN_PROCESS:
                        acc = evaluate(
                            unwrap_model(model), tokenizer, val_df,
                            max_samples=VAL_MAX_SAMPLES,
                            step_label=f"step={global_step}",
                            max_new_tokens=EVAL_MAX_NEW_TOKENS,
                            wandb_logger=wandb_logger,
                            wandb_step=global_step,
                        )
                    acc = broadcast_float(acc, first_device)
                    barrier()
                    if acc > best_acc:
                        best_acc = acc
                        if IS_MAIN_PROCESS:
                            best_dir = OUTPUT_DIR + "_best"
                            os.makedirs(best_dir, exist_ok=True)
                            unwrap_model(model).save_pretrained(best_dir)
                            log_main(f"  New best accuracy: {best_acc:.4f}, saved to {best_dir}")
                    model.train()

                if CHECKPOINT_EVERY_STEPS > 0 and global_step % CHECKPOINT_EVERY_STEPS == 0:
                    next_epoch = epoch
                    next_step = step + 1
                    if next_step >= len(train_loader):
                        next_epoch = epoch + 1
                        next_step = 0
                    if IS_MAIN_PROCESS:
                        checkpoint_path = save_training_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            CHECKPOINT_DIR,
                            next_epoch,
                            next_step,
                            global_step,
                            best_acc,
                            running_loss,
                            total_steps,
                        )
                        wandb_logger.log_checkpoint(checkpoint_path, global_step, best_acc)
                    barrier()

                if MAX_TRAIN_STEPS > 0 and global_step >= MAX_TRAIN_STEPS:
                    log_main(f"  Reached MAX_TRAIN_STEPS={MAX_TRAIN_STEPS}, stopping early.")
                    if CHECKPOINT_EVERY_STEPS > 0 and global_step % CHECKPOINT_EVERY_STEPS != 0:
                        next_epoch = epoch
                        next_step = step + 1
                        if next_step >= len(train_loader):
                            next_epoch = epoch + 1
                            next_step = 0
                        if IS_MAIN_PROCESS:
                            checkpoint_path = save_training_checkpoint(
                                model,
                                optimizer,
                                scheduler,
                                CHECKPOINT_DIR,
                                next_epoch,
                                next_step,
                                global_step,
                                best_acc,
                                running_loss,
                                total_steps,
                            )
                            wandb_logger.log_checkpoint(checkpoint_path, global_step, best_acc)
                        barrier()
                    stop_training = True
                    break

        # End-of-epoch evaluation
        if stop_training:
            break
        acc = 0.0
        if IS_MAIN_PROCESS:
            acc = evaluate(
                unwrap_model(model), tokenizer, val_df,
                max_samples=VAL_MAX_SAMPLES,
                step_label=f"epoch={epoch}, step={global_step}",
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                wandb_logger=wandb_logger,
                wandb_step=global_step,
            )
        acc = broadcast_float(acc, torch.device("cuda", LOCAL_RANK if IS_DISTRIBUTED else 0))
        barrier()
        if acc > best_acc:
            best_acc = acc
            if IS_MAIN_PROCESS:
                best_dir = OUTPUT_DIR + "_best"
                os.makedirs(best_dir, exist_ok=True)
                unwrap_model(model).save_pretrained(best_dir)
                log_main(f"  New best accuracy: {best_acc:.4f}, saved to {best_dir}")
        if CHECKPOINT_EVERY_STEPS > 0 and global_step > 0 and global_step % CHECKPOINT_EVERY_STEPS != 0:
            if IS_MAIN_PROCESS:
                checkpoint_path = save_training_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    CHECKPOINT_DIR,
                    epoch + 1,
                    0,
                    global_step,
                    best_acc,
                    running_loss,
                    total_steps,
                )
                wandb_logger.log_checkpoint(checkpoint_path, global_step, best_acc)
            barrier()

    log_main(f"\n  Training complete. Total optimizer steps: {global_step}")
    log_main(f"  Best validation accuracy: {best_acc:.4f}")

    if IS_MAIN_PROCESS:
        # ─── Save final adapter ───
        log_main(f"\n[8/8] Saving final adapter to {OUTPUT_DIR} ...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        unwrap_model(model).save_pretrained(OUTPUT_DIR)

        # Use best checkpoint for submission if it exists
        best_dir = OUTPUT_DIR + "_best"
        submit_dir = best_dir if os.path.exists(best_dir) else OUTPUT_DIR
        log_main(f"  Using {submit_dir} for submission (best checkpoint)")

        # Verify files
        adapter_config = os.path.join(submit_dir, "adapter_config.json")
        adapter_weights = os.path.join(submit_dir, "adapter_model.safetensors")
        assert os.path.exists(adapter_config), f"Missing {adapter_config}"
        assert os.path.exists(adapter_weights), f"Missing {adapter_weights}"
        log_main(f"  adapter_config.json: {os.path.getsize(adapter_config)} bytes")
        log_main(f"  adapter_model.safetensors: {os.path.getsize(adapter_weights) / 1e6:.1f} MB")

        # Package submission.zip
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        zip_path = os.path.join(SUBMISSION_DIR, "submission.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        subprocess.run(
            f"cd {submit_dir} && zip -r {zip_path} adapter_config.json adapter_model.safetensors",
            shell=True, check=True,
        )
        log_main(f"  submission.zip: {os.path.getsize(zip_path) / 1e6:.1f} MB")
        wandb_logger.log_final(global_step, best_acc, submit_dir, zip_path)

        # List zip contents
        result = subprocess.run(f"unzip -l {zip_path}", shell=True, capture_output=True, text=True)
        log_main(f"\n  submission.zip contents:")
        log_main(result.stdout)

        log_main("\n" + "=" * 60)
        log_main(f"TRAINING COMPLETE — Best val accuracy: {best_acc:.4f}")
        log_main("=" * 60)
        wandb_logger.finish()
    barrier()
    if IS_DISTRIBUTED:
        dist.destroy_process_group()
    return


if __name__ == "__main__":
    main()
