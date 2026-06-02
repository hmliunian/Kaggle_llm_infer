#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-synthetic_v1_wandb_long}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/$RUN_NAME}"
GPU="${GPU_ID:-5}"
LOG_PATH="$RUN_DIR/$RUN_NAME.log"
PID_PATH="$RUN_DIR/$RUN_NAME.pid"

mkdir -p "$RUN_DIR/artifacts" "$RUN_DIR/eval" "$RUN_DIR/checkpoints" "$RUN_DIR/wandb"

if [[ -f "$PID_PATH" ]]; then
  old_pid="$(cat "$PID_PATH")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Training already appears to be running: PID $old_pid"
    exit 0
  fi
fi

echo "Starting $RUN_NAME on GPU $GPU"
echo "Log: $LOG_PATH"

nohup env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  GPU_ID="$GPU" \
  TRAIN_CSV="$ROOT_DIR/data/train_plus_synthetic_v1.csv" \
  RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-$ROOT_DIR/runs/synthetic_v1_13h/adapter}" \
  LORA_RANK="${LORA_RANK:-16}" \
  LORA_ALPHA="${LORA_ALPHA:-16}" \
  BATCH_SIZE="${BATCH_SIZE:-2}" \
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}" \
  NUM_EPOCHS="${NUM_EPOCHS:-20}" \
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}" \
  ENABLE_BASELINE_EVAL="${ENABLE_BASELINE_EVAL:-0}" \
  VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-100}" \
  EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-100}" \
  EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}" \
  CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-100}" \
  KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}" \
  OUTPUT_DIR="$RUN_DIR/adapter" \
  SUBMISSION_DIR="$RUN_DIR/artifacts" \
  EVAL_OUTPUT_DIR="$RUN_DIR/eval" \
  CHECKPOINT_DIR="$RUN_DIR/checkpoints" \
  WANDB_ENABLED="${WANDB_ENABLED:-1}" \
  WANDB_PROJECT="${WANDB_PROJECT:-llm-infer-sft}" \
  WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_NAME}" \
  WANDB_MODE="${WANDB_MODE:-online}" \
  WANDB_DIR="$RUN_DIR/wandb" \
  WANDB_LOG_ARTIFACTS="${WANDB_LOG_ARTIFACTS:-0}" \
  "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/train_sft.py" \
  >> "$LOG_PATH" 2>&1 &

pid="$!"
echo "$pid" > "$PID_PATH"
echo "PID: $pid"
