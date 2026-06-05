#!/usr/bin/env bash
# Wait for a GPU with enough free memory to hold the ~59GB base model, then launch
# the full-val official-metric eval of the best CoT checkpoint (step-1000).
# Non-disruptive: it only uses a GPU once it is genuinely free, never evicts others.
set -uo pipefail

cd /data2/yaoxuran/llm_infer

RUN=runs/cot_v4_gpu3_resume_trunc896
ADAPTER=${ADAPTER:-$RUN/checkpoints/checkpoint-001000}
NEED_FREE_MIB=${NEED_FREE_MIB:-68000}      # ~59GB weights + KV cache/overhead
POLL_SECONDS=${POLL_SECONDS:-60}
CANDIDATES=${CANDIDATES:-"6 5 7 4 2 0 1"}  # exclude GPU3 (live training)
LOG=$RUN/eval/cot_step1000_fullval_official.launch.log

mkdir -p "$RUN/eval"
echo "[watch] $(date '+%F %T') waiting for a GPU with >=${NEED_FREE_MIB}MiB free among: $CANDIDATES" | tee -a "$LOG"

while true; do
  for g in $CANDIDATES; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [ -z "$free" ] && continue
    if [ "$free" -ge "$NEED_FREE_MIB" ]; then
      echo "[watch] $(date '+%F %T') GPU $g has ${free}MiB free -> launching eval" | tee -a "$LOG"
      GPU_ID=$g \
      ADAPTER_DIR=$ADAPTER \
      TRAIN_CSV=data/train_plus_synthetic_v4.csv \
      VAL_MAX_SAMPLES=0 \
      EVAL_MAX_NEW_TOKENS=1024 \
      INFERENCE_FINAL_ANSWER_PREFILL=0 \
      INFERENCE_STOP_AFTER_BOXED=1 \
      MAX_SEQ_LEN=1024 \
      EVAL_LABEL=cot_step1000_fullval_official \
      EVAL_OUTPUT_DIR=$RUN/eval \
      nohup .venv/bin/python eval_adapter.py >> "$LOG" 2>&1 &
      echo $! > "$RUN/eval/cot_step1000_fullval_official.pid"
      echo "[watch] $(date '+%F %T') eval PID $(cat "$RUN/eval/cot_step1000_fullval_official.pid") on GPU $g" | tee -a "$LOG"
      exit 0
    fi
  done
  sleep "$POLL_SECONDS"
done
