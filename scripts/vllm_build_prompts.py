"""
Stage 1 (TRAINING venv): build the EXACT eval val set + tokenized prompts that
eval_adapter.py / train_sft.evaluate() would feed the HF model, and dump them so
the vLLM stage can consume byte-identical input tokens (no cross-venv tokenizer
drift, no double-BOS surprises).

Mirrors eval_adapter.py: split_train_val -> optional shard (gather_every) ->
random.Random(SEED).sample cap -> enumerate sample_index. Prompt is built with
train_sft.format_inference_prompt + the same answer_prefix as generate_completion,
then tokenized with the same truncation as generate_completion.

Env (same names eval_adapter uses): TRAIN_CSV, NUM_SHARDS, SHARD_INDEX,
VAL_MAX_SAMPLES, EVAL_BASE_ONLY, OFFICIAL_PROMPT_ALIGN, INFERENCE_FINAL_ANSWER_PREFILL,
MAX_SEQ_LEN, MODEL_PATH. Output: --out jsonl with one row per sample.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from transformers import AutoTokenizer

import train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_path = os.environ.get("MODEL_PATH", train_sft.MODEL_PATH)
    train_csv = os.environ.get("TRAIN_CSV", train_sft.TRAIN_CSV)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    full_df = pl.read_csv(
        train_csv,
        schema_overrides={"prompt": pl.String, "answer": pl.String},
    )
    _, val_df = train_sft.split_train_val(
        full_df, val_ratio=train_sft.VAL_RATIO, seed=train_sft.SEED
    )

    num_shards = int(os.environ.get("NUM_SHARDS", "1"))
    shard_index = int(os.environ.get("SHARD_INDEX", "0"))
    if num_shards > 1:
        val_df = val_df.gather_every(num_shards, offset=shard_index)

    # --- replicate train_sft.evaluate() sample selection EXACTLY ---
    samples = val_df.to_dicts()
    max_samples = train_sft.VAL_MAX_SAMPLES
    if max_samples > 0 and len(samples) > max_samples:
        rng = random.Random(train_sft.SEED)
        samples = rng.sample(samples, max_samples)

    # answer_prefix mirrors generate_completion()
    answer_prefix = (
        train_sft.FINAL_ANSWER_PREFILL_TEXT
        if train_sft.INFERENCE_FINAL_ANSWER_PREFILL
        else ""
    )

    n = 0
    with open(args.out, "w") as f:
        for sample_idx, row in enumerate(samples):
            prompt = row["prompt"]
            prompt_text = train_sft.format_inference_prompt(tok, prompt) + answer_prefix
            enc = tok(
                prompt_text,
                truncation=True,
                max_length=train_sft.MAX_SEQ_LEN,
                add_special_tokens=True,
            )
            rec = {
                "sample_index": sample_idx,
                "prompt_token_ids": enc["input_ids"],
                "answer_prefix": answer_prefix,
                "gold": str(row["answer"]).strip(),
                "family": train_sft.classify_prompt(prompt),
                "source": (str(row.get("source", "unknown") or "unknown").strip() or "unknown"),
                "prompt": prompt,
            }
            f.write(json.dumps(rec) + "\n")
            n += 1

    print(
        f"[build_prompts] wrote {n} rows -> {args.out} "
        f"(shard {shard_index}/{num_shards}, VAL_MAX_SAMPLES={max_samples}, "
        f"MAX_SEQ_LEN={train_sft.MAX_SEQ_LEN}, EVAL_BASE_ONLY={train_sft.EVAL_BASE_ONLY}, "
        f"prefill={train_sft.INFERENCE_FINAL_ANSWER_PREFILL!r})"
    )


if __name__ == "__main__":
    main()
