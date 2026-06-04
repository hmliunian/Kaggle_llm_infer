#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data2/yaoxuran/llm_infer}"
EVAL_PID="${EVAL_PID:-1154476}"
SINGLE_RUN_NAME="${SINGLE_RUN_NAME:-synthetic_v3_gpu6_v2best_ckpt50}"
DDP_RUN_NAME="${DDP_RUN_NAME:-synthetic_v3_ddp56_from_gpu6_latest}"
POLL_SECONDS="${POLL_SECONDS:-60}"

TRAIN_CSV="${TRAIN_CSV:-$ROOT_DIR/data/train_plus_synthetic_v3.csv}"
SINGLE_RUN_DIR="$ROOT_DIR/runs/$SINGLE_RUN_NAME"
DDP_RUN_DIR="$ROOT_DIR/runs/$DDP_RUN_NAME"
WATCH_LOG="${WATCH_LOG:-$SINGLE_RUN_DIR/watch_start_ddp_after_eval.log}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$WATCH_LOG"
}

checkpoint_ready() {
  local checkpoint_path="$1"
  [[ -n "$checkpoint_path" ]] \
    && [[ -f "$checkpoint_path/adapter_config.json" ]] \
    && [[ -f "$checkpoint_path/training_state.pt" ]]
}

mkdir -p "$SINGLE_RUN_DIR" "$DDP_RUN_DIR" "$DDP_RUN_DIR/artifacts" "$DDP_RUN_DIR/eval" "$DDP_RUN_DIR/checkpoints" "$DDP_RUN_DIR/wandb"
log "Watcher started. eval_pid=$EVAL_PID single_run=$SINGLE_RUN_NAME ddp_run=$DDP_RUN_NAME"

while ps -p "$EVAL_PID" >/dev/null 2>&1; do
  log "Waiting for eval PID $EVAL_PID to finish."
  sleep "$POLL_SECONDS"
done
log "Eval PID $EVAL_PID is no longer running."

latest_file="$SINGLE_RUN_DIR/checkpoints/latest_checkpoint.txt"
while true; do
  if [[ -f "$latest_file" ]]; then
    checkpoint_path="$(tr -d '\n' < "$latest_file")"
    if checkpoint_ready "$checkpoint_path"; then
      break
    fi
    log "Latest checkpoint pointer exists but checkpoint is not complete yet: $checkpoint_path"
  else
    log "Waiting for first single-GPU checkpoint at $latest_file"
  fi
  sleep "$POLL_SECONDS"
done
log "Using single-GPU checkpoint for DDP resume: $checkpoint_path"

single_pid_file="$SINGLE_RUN_DIR/$SINGLE_RUN_NAME.pid"
if [[ -f "$single_pid_file" ]]; then
  single_pid="$(tr -d '\n' < "$single_pid_file")"
  if [[ -n "$single_pid" ]] && ps -p "$single_pid" >/dev/null 2>&1; then
    log "Stopping single-GPU training PID $single_pid before starting DDP."
    kill -TERM "$single_pid"
    for _ in $(seq 1 60); do
      if ! ps -p "$single_pid" >/dev/null 2>&1; then
        break
      fi
      sleep 5
    done
    if ps -p "$single_pid" >/dev/null 2>&1; then
      log "Single-GPU PID $single_pid is still running after SIGTERM; not starting DDP to avoid GPU contention."
      exit 1
    fi
  else
    log "Single-GPU PID is not running: ${single_pid:-empty}"
  fi
else
  log "Single-GPU PID file not found: $single_pid_file"
fi

log "Starting DDP v3 run on GPUs 5,6."
(
  cd "$ROOT_DIR"
  env CUDA_VISIBLE_DEVICES=5,6 \
    TRAIN_CSV="$TRAIN_CSV" \
    RESUME_FROM_CHECKPOINT="$checkpoint_path" \
    LORA_RANK=16 \
    LORA_ALPHA=16 \
    BATCH_SIZE=1 \
    GRAD_ACCUM_STEPS=4 \
    NUM_EPOCHS=3 \
    MAX_TRAIN_STEPS=10000 \
    ENABLE_BASELINE_EVAL=0 \
    VAL_MAX_SAMPLES=100 \
    EVAL_EVERY_STEPS=100 \
    EVAL_MAX_NEW_TOKENS=64 \
    CHECKPOINT_EVERY_STEPS=50 \
    KEEP_LAST_CHECKPOINTS=20 \
    OUTPUT_DIR="$DDP_RUN_DIR/adapter" \
    SUBMISSION_DIR="$DDP_RUN_DIR/artifacts" \
    EVAL_OUTPUT_DIR="$DDP_RUN_DIR/eval" \
    CHECKPOINT_DIR="$DDP_RUN_DIR/checkpoints" \
    WANDB_ENABLED=1 \
    WANDB_PROJECT=llm-infer-sft \
    WANDB_RUN_NAME="$DDP_RUN_NAME" \
    WANDB_DIR="$DDP_RUN_DIR/wandb" \
    WANDB_LOG_ARTIFACTS=0 \
    DISABLE_CUDNN_SDP=1 \
    .venv/bin/torchrun --standalone --nproc_per_node=2 train_sft.py \
    > "$DDP_RUN_DIR/$DDP_RUN_NAME.log" 2>&1 &
  echo "$!" > "$DDP_RUN_DIR/$DDP_RUN_NAME.pid"
)
log "DDP launch requested. pid=$(cat "$DDP_RUN_DIR/$DDP_RUN_NAME.pid") log=$DDP_RUN_DIR/$DDP_RUN_NAME.log"
