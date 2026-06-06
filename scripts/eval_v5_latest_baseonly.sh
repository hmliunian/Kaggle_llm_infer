#!/usr/bin/env bash
# Base-only eval of the latest v5 checkpoint, 2-shard data-parallel on GPU5/6.
# Uses the new EVAL_BASE_ONLY split so the val set is official rows only (947),
# the same config the training's own eval uses (thinking mode, 512 new tokens,
# stop-after-boxed, official prompt alignment). VAL_PER_SHARD caps each shard, so
# total evaluated = 2 * VAL_PER_SHARD (set 0 for the full 947-row official val).
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data2/yaoxuran/llm_infer}"
RUN_DIR="$ROOT_DIR/runs/cot_v5_gpu3_from_base_raw"
CKPT="${CKPT:-$(tr -d '[:space:]' < "$RUN_DIR/checkpoints/latest_checkpoint.txt")}"
STEP="$(basename "$CKPT" | grep -oE '[0-9]+')"
VAL_PER_SHARD="${VAL_PER_SHARD:-150}"
GPU0="${GPU0:-5}"
GPU1="${GPU1:-6}"
LABEL="v5_step${STEP}_baseonly_$([ "$VAL_PER_SHARD" = 0 ] && echo full || echo $((VAL_PER_SHARD*2)))"

cd "$ROOT_DIR"
echo "checkpoint=$CKPT step=$STEP label=$LABEL per_shard=$VAL_PER_SHARD gpus=$GPU0,$GPU1"
for SH in 0 1; do
  GPU=$([ "$SH" = 0 ] && echo "$GPU0" || echo "$GPU1")
  env GPU_ID="$GPU" \
    ADAPTER_DIR="$CKPT" \
    TRAIN_CSV="$ROOT_DIR/data/train_plus_synthetic_v5.csv" \
    EVAL_BASE_ONLY=1 \
    OFFICIAL_PROMPT_ALIGN=1 \
    INFERENCE_FINAL_ANSWER_PREFILL=0 \
    INFERENCE_STOP_AFTER_BOXED=1 \
    EVAL_MAX_NEW_TOKENS=512 \
    MAX_SEQ_LEN=1024 \
    VAL_MAX_SAMPLES="$VAL_PER_SHARD" \
    NUM_SHARDS=2 \
    SHARD_INDEX="$SH" \
    EVAL_LABEL="${LABEL}_s${SH}" \
    EVAL_OUTPUT_DIR="$RUN_DIR/eval" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python eval_adapter.py > "$RUN_DIR/eval/${LABEL}_s${SH}.log" 2>&1 &
  echo "  shard$SH pid=$! gpu=$GPU log=$RUN_DIR/eval/${LABEL}_s${SH}.log"
done
wait
echo "both shards done; merge with scripts/merge_shard_eval.py"
