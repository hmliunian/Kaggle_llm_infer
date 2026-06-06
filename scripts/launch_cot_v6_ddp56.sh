#!/usr/bin/env bash
# Fresh-from-base raw-answer CoT finetune on the v6 dataset, GPUs 5,6 (DDP).
# Hyperparameters mirror the v5 run (scripts/launch_cot_v5_gpu3.sh) so v6 is an
# apples-to-apples comparison; only the dataset (v6) and the 2-GPU DDP layout
# differ. Effective batch = 2 GPUs * BATCH_SIZE 1 * GRAD_ACCUM 4 = 8 (same as the
# single-GPU v5 run with GRAD_ACCUM 8).
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data2/yaoxuran/llm_infer}"
RUN_NAME="${RUN_NAME:-cot_v6_ddp56_from_base_raw}"
RUN_DIR="$ROOT_DIR/runs/$RUN_NAME"
TRAIN_CSV="${TRAIN_CSV:-$ROOT_DIR/data/train_plus_synthetic_v6.csv}"

mkdir -p "$RUN_DIR" "$RUN_DIR/artifacts" "$RUN_DIR/eval" "$RUN_DIR/checkpoints" "$RUN_DIR/wandb"

cd "$ROOT_DIR"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6}" \
  TRAIN_CSV="$TRAIN_CSV" \
  RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}" \
  MAX_SEQ_LEN="${MAX_SEQ_LEN:-896}" \
  LORA_RANK=16 \
  LORA_ALPHA=16 \
  BATCH_SIZE=1 \
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}" \
  NUM_EPOCHS=3 \
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}" \
  TRAIN_FINAL_ANSWER_PREFILL=1 \
  INFERENCE_FINAL_ANSWER_PREFILL=0 \
  INFERENCE_STOP_AFTER_BOXED=1 \
  ENABLE_BASELINE_EVAL=0 \
  EVAL_BASE_ONLY="${EVAL_BASE_ONLY:-1}" \
  BASE_SOURCE="${BASE_SOURCE:-official_train}" \
  VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-48}" \
  EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-200}" \
  EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}" \
  CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-50}" \
  KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-20}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  OUTPUT_DIR="$RUN_DIR/adapter" \
  SUBMISSION_DIR="$RUN_DIR/artifacts" \
  EVAL_OUTPUT_DIR="$RUN_DIR/eval" \
  CHECKPOINT_DIR="$RUN_DIR/checkpoints" \
  WANDB_ENABLED="${WANDB_ENABLED:-1}" \
  WANDB_PROJECT=llm-infer-sft \
  WANDB_RUN_NAME="$RUN_NAME" \
  WANDB_DIR="$RUN_DIR/wandb" \
  WANDB_LOG_ARTIFACTS=0 \
  DISABLE_CUDNN_SDP=1 \
  TRAIN_USE_CACHE=0 \
  .venv/bin/torchrun --standalone --nproc_per_node=2 train_sft.py \
  > "$RUN_DIR/$RUN_NAME.log" 2>&1 &
echo "$!" > "$RUN_DIR/$RUN_NAME.pid"
echo "launched torchrun pid=$(cat "$RUN_DIR/$RUN_NAME.pid") log=$RUN_DIR/$RUN_NAME.log"
