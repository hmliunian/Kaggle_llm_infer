#!/usr/bin/env bash
# Smoke-test raw-answer CoT v5 training from base on one GPU.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data2/yaoxuran/llm_infer}"
GPU_ID="${GPU_ID:-3}"
RUN_NAME="${RUN_NAME:-cot_v5_raw_smoke_gpu${GPU_ID}}"
RUN_DIR="$ROOT_DIR/runs/$RUN_NAME"

mkdir -p "$RUN_DIR/artifacts" "$RUN_DIR/eval" "$RUN_DIR/checkpoints" "$RUN_DIR/wandb"

cd "$ROOT_DIR"
env CUDA_VISIBLE_DEVICES="$GPU_ID" \
  GPU_ID="$GPU_ID" \
  SMOKE_MODE=1 \
  TRAIN_CSV="$ROOT_DIR/data/train_plus_synthetic_v5.csv" \
  RESUME_FROM_CHECKPOINT= \
  MAX_SEQ_LEN="${MAX_SEQ_LEN:-896}" \
  LORA_RANK=16 \
  LORA_ALPHA=16 \
  BATCH_SIZE=1 \
  GRAD_ACCUM_STEPS=4 \
  MAX_TRAIN_STEPS=2 \
  TRAIN_ROW_LIMIT=80 \
  TRAIN_FINAL_ANSWER_PREFILL=1 \
  INFERENCE_FINAL_ANSWER_PREFILL=0 \
  INFERENCE_STOP_AFTER_BOXED=1 \
  ENABLE_BASELINE_EVAL=0 \
  EVAL_BASE_ONLY="${EVAL_BASE_ONLY:-1}" \
  BASE_SOURCE="${BASE_SOURCE:-official_train}" \
  VAL_MAX_SAMPLES=2 \
  EVAL_EVERY_STEPS=1 \
  EVAL_MAX_NEW_TOKENS=128 \
  CHECKPOINT_EVERY_STEPS=1 \
  KEEP_LAST_CHECKPOINTS=2 \
  OUTPUT_DIR="$RUN_DIR/adapter" \
  SUBMISSION_DIR="$RUN_DIR/artifacts" \
  EVAL_OUTPUT_DIR="$RUN_DIR/eval" \
  CHECKPOINT_DIR="$RUN_DIR/checkpoints" \
  WANDB_ENABLED=0 \
  DISABLE_CUDNN_SDP=1 \
  TRAIN_USE_CACHE=0 \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  .venv/bin/python train_sft.py \
  > "$RUN_DIR/$RUN_NAME.log" 2>&1
