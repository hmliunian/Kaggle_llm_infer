#!/usr/bin/env bash
# CoT v4 finetune on a single GPU. Defaults to resuming from the first stable
# checkpoint and capping sequence length so long CoT samples are truncated.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data2/yaoxuran/llm_infer}"
TRAIN_GPU_ID="${TRAIN_GPU_ID:-5}"
RUN_NAME="${RUN_NAME:-cot_v4_gpu${TRAIN_GPU_ID}_resume_trunc896}"
RUN_DIR="$ROOT_DIR/runs/$RUN_NAME"
TRAIN_CSV="${TRAIN_CSV:-$ROOT_DIR/data/train_plus_synthetic_v4.csv}"
RESUME_CKPT="${RESUME_CKPT:-$ROOT_DIR/runs/cot_v4_ddp56_from_base_retry/checkpoints/checkpoint-000050}"

mkdir -p "$RUN_DIR" "$RUN_DIR/artifacts" "$RUN_DIR/eval" "$RUN_DIR/checkpoints" "$RUN_DIR/wandb"

cd "$ROOT_DIR"
env CUDA_VISIBLE_DEVICES="$TRAIN_GPU_ID" \
  GPU_ID="$TRAIN_GPU_ID" \
  TRAIN_CSV="$TRAIN_CSV" \
  RESUME_FROM_CHECKPOINT="$RESUME_CKPT" \
  MAX_SEQ_LEN="${MAX_SEQ_LEN:-896}" \
  LORA_RANK=16 \
  LORA_ALPHA=16 \
  BATCH_SIZE=1 \
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}" \
  NUM_EPOCHS=3 \
  MAX_TRAIN_STEPS=10000 \
  TRAIN_FINAL_ANSWER_PREFILL=1 \
  INFERENCE_FINAL_ANSWER_PREFILL=0 \
  INFERENCE_STOP_AFTER_BOXED=1 \
  ENABLE_BASELINE_EVAL=0 \
  VAL_MAX_SAMPLES=48 \
  EVAL_EVERY_STEPS=200 \
  EVAL_MAX_NEW_TOKENS=512 \
  CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-50}" \
  KEEP_LAST_CHECKPOINTS=20 \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  OUTPUT_DIR="$RUN_DIR/adapter" \
  SUBMISSION_DIR="$RUN_DIR/artifacts" \
  EVAL_OUTPUT_DIR="$RUN_DIR/eval" \
  CHECKPOINT_DIR="$RUN_DIR/checkpoints" \
  WANDB_ENABLED=1 \
  WANDB_PROJECT=llm-infer-sft \
  WANDB_RUN_NAME="$RUN_NAME" \
  WANDB_DIR="$RUN_DIR/wandb" \
  WANDB_LOG_ARTIFACTS=0 \
  DISABLE_CUDNN_SDP=1 \
  TRAIN_USE_CACHE=0 \
  .venv/bin/python train_sft.py \
  > "$RUN_DIR/$RUN_NAME.log" 2>&1 &
echo "$!" > "$RUN_DIR/$RUN_NAME.pid"
echo "launched python pid=$(cat "$RUN_DIR/$RUN_NAME.pid") log=$RUN_DIR/$RUN_NAME.log"
