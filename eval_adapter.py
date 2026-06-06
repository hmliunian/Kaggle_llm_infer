"""
Evaluate a saved LoRA adapter and persist per-sample outputs for analysis.
"""

import os
from pathlib import Path

GPU_ID = os.environ.get("GPU_ID", "6")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import polars as pl
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import train_sft
from nemotron_cache_compat import patch_nemotron_h_cache_compat


def main():
    model_path = os.environ.get("MODEL_PATH", train_sft.MODEL_PATH)
    adapter_dir = os.environ.get("ADAPTER_DIR", train_sft.OUTPUT_DIR)
    train_csv = os.environ.get("TRAIN_CSV", train_sft.TRAIN_CSV)
    eval_label = os.environ.get("EVAL_LABEL", "adapter_eval")

    if "EVAL_OUTPUT_DIR" in os.environ:
        train_sft.EVAL_OUTPUT_DIR = os.environ["EVAL_OUTPUT_DIR"]
    else:
        train_sft.EVAL_OUTPUT_DIR = str(Path(adapter_dir).resolve().parent / "eval")

    adapter_path = Path(adapter_dir)
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing adapter_config.json in ADAPTER_DIR={adapter_path}")

    print("[0/5] Eval config")
    print(f"  GPU_ID={GPU_ID}")
    print(f"  MODEL_PATH={model_path}")
    print(f"  ADAPTER_DIR={adapter_path}")
    print(f"  TRAIN_CSV={train_csv}")
    print(f"  VAL_MAX_SAMPLES={train_sft.VAL_MAX_SAMPLES}")
    print(f"  EVAL_MAX_NEW_TOKENS={train_sft.EVAL_MAX_NEW_TOKENS}")
    print(f"  INFERENCE_FINAL_ANSWER_PREFILL={train_sft.INFERENCE_FINAL_ANSWER_PREFILL}")
    print(f"  INFERENCE_STOP_AFTER_BOXED={train_sft.INFERENCE_STOP_AFTER_BOXED}")
    print(f"  EVAL_OUTPUT_DIR={train_sft.EVAL_OUTPUT_DIR}")
    print(f"  EVAL_LABEL={eval_label}")
    print()

    print("[1/5] Loading validation split ...")
    full_df = pl.read_csv(
        train_csv,
        schema_overrides={
            "prompt": pl.String,
            "answer": pl.String,
        },
    )
    _, val_df = train_sft.split_train_val(full_df, val_ratio=train_sft.VAL_RATIO, seed=train_sft.SEED)
    print(f"  Total rows: {len(full_df)}, Val rows: {len(val_df)} (EVAL_BASE_ONLY={train_sft.EVAL_BASE_ONLY}, BASE_SOURCE={train_sft.BASE_SOURCE!r})")

    # Optional data-parallel sharding: split the val set across processes/GPUs.
    # gather_every keeps the split deterministic and interleaved so each shard sees
    # a comparable family mix. Each shard writes its own EVAL_LABEL outputs.
    num_shards = int(os.environ.get("NUM_SHARDS", "1"))
    shard_index = int(os.environ.get("SHARD_INDEX", "0"))
    if num_shards > 1:
        val_df = val_df.gather_every(num_shards, offset=shard_index)
        print(f"  Shard {shard_index}/{num_shards}: {len(val_df)} rows")

    print(f"\n[2/5] Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n[3/5] Loading base model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    patch_nemotron_h_cache_compat(model)

    print(f"\n[4/5] Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    print("\n[5/5] Running eval ...")
    acc = train_sft.evaluate(
        model,
        tokenizer,
        val_df,
        max_samples=train_sft.VAL_MAX_SAMPLES,
        step_label=eval_label,
        max_new_tokens=train_sft.EVAL_MAX_NEW_TOKENS,
    )
    print(f"Eval complete. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
