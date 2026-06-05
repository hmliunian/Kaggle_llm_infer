#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data2/yaoxuran/llm_infer}"
RUN_NAME="${RUN_NAME:-cot_v4_gpu5_resume_trunc896}"
WATCH_SECONDS="${WATCH_SECONDS:-1800}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MEM_THRESHOLD_MB="${MEM_THRESHOLD_MB:-2000}"
WATCH_LOG="$ROOT_DIR/runs/${RUN_NAME}.watch.log"

mkdir -p "$ROOT_DIR/runs"

now_ts() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

gpu_busy() {
  local rows
  rows="$(nvidia-smi --id=5 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
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
echo "$(now_ts) watcher started run=$RUN_NAME gpu=5 threshold=${MEM_THRESHOLD_MB}MB timeout=${WATCH_SECONDS}s" >> "$WATCH_LOG"

while [ "$(date +%s)" -lt "$deadline" ]; do
  if ! gpu_busy; then
    echo "$(now_ts) GPU 5 is available; launching training" >> "$WATCH_LOG"
    RUN_NAME="$RUN_NAME" bash "$ROOT_DIR/scripts/launch_cot_v4_gpu5.sh" >> "$WATCH_LOG" 2>&1
    echo "$(now_ts) launch command finished" >> "$WATCH_LOG"
    exit 0
  fi
  echo "$(now_ts) waiting for GPU 5 to clear" >> "$WATCH_LOG"
  nvidia-smi --id=5 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits >> "$WATCH_LOG" 2>&1 || true
  sleep "$POLL_SECONDS"
done

echo "$(now_ts) timeout reached without launching" >> "$WATCH_LOG"
exit 2
