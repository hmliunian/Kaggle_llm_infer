#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data2/yaoxuran/llm_infer}"
RUN_NAME="${RUN_NAME:-cot_v4_ddp56_from_base_resume_nocache}"
RESUME_CKPT="${RESUME_CKPT:-$ROOT_DIR/runs/cot_v4_ddp56_from_base_retry/checkpoints/checkpoint-000050}"
WATCH_SECONDS="${WATCH_SECONDS:-5400}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MEM_THRESHOLD_MB="${MEM_THRESHOLD_MB:-2000}"
WATCH_LOG="$ROOT_DIR/runs/${RUN_NAME}.watch.log"

mkdir -p "$ROOT_DIR/runs"

now_ts() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

gpu_busy() {
  local gpu_id="$1"
  local rows
  rows="$(nvidia-smi --id="$gpu_id" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  if [ -z "$rows" ]; then
    return 1
  fi
  awk -F, -v threshold="$MEM_THRESHOLD_MB" '
    {
      mem=$2
      gsub(/^[ \t]+|[ \t]+$/, "", mem)
      if (mem + 0 >= threshold) {
        found=1
      }
    }
    END { exit(found ? 0 : 1) }
  ' <<<"$rows"
}

deadline=$(( $(date +%s) + WATCH_SECONDS ))
echo "$(now_ts) watcher started run=$RUN_NAME resume=$RESUME_CKPT threshold=${MEM_THRESHOLD_MB}MB timeout=${WATCH_SECONDS}s" >> "$WATCH_LOG"

while [ "$(date +%s)" -lt "$deadline" ]; do
  if ! gpu_busy 5 && ! gpu_busy 6; then
    echo "$(now_ts) GPUs 5 and 6 are available; launching training" >> "$WATCH_LOG"
    RUN_NAME="$RUN_NAME" RESUME_FROM_CHECKPOINT="$RESUME_CKPT" bash "$ROOT_DIR/scripts/launch_cot_v4_ddp.sh" >> "$WATCH_LOG" 2>&1
    echo "$(now_ts) launch command finished" >> "$WATCH_LOG"
    exit 0
  fi
  echo "$(now_ts) waiting for GPUs 5/6 to clear" >> "$WATCH_LOG"
  nvidia-smi --id=5 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits >> "$WATCH_LOG" 2>&1 || true
  nvidia-smi --id=6 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits >> "$WATCH_LOG" 2>&1 || true
  sleep "$POLL_SECONDS"
done

echo "$(now_ts) timeout reached without launching" >> "$WATCH_LOG"
exit 2
