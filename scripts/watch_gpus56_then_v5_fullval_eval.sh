#!/usr/bin/env bash
# When the in-flight v4 full-val eval on GPU5/6 finishes (its PIDs exit AND both GPUs
# free their memory), run a 2-shard full-val official-metric eval of the BEST v5
# checkpoint: shard0 on GPU5, shard1 on GPU6, then merge into one official summary.
#
# Non-disruptive: only launches once GPU5/6 are genuinely free; never evicts anything.
# The best checkpoint is chosen AT LAUNCH TIME (by 48-val accuracy) so it picks up
# checkpoints the v5 run writes while we wait. Falls back to latest_checkpoint.txt.
set -uo pipefail
cd /data2/yaoxuran/llm_infer

RUN=runs/cot_v5_gpu3_from_base_raw
WATCH_PIDS="${WATCH_PIDS:-1636773 1636775}"   # the in-flight v4 fullval shards on GPU5/6
GPU_A=${GPU_A:-5}                             # -> shard 0
GPU_B=${GPU_B:-6}                             # -> shard 1
NEED_FREE_MIB=${NEED_FREE_MIB:-68000}         # ~59GB weights + KV cache/overhead per GPU
POLL=${POLL:-60}
TRAIN_CSV=${TRAIN_CSV:-data/train_plus_synthetic_v5.csv}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-1024}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}              # eval prompt headroom (>= v5 train 896)
ADAPTER_OVERRIDE=${ADAPTER_OVERRIDE:-}        # set to force a specific checkpoint dir

mkdir -p "$RUN/eval"
LOG=$RUN/eval/v5_fullval_official.watch.log
log(){ echo "[watch] $(date '+%F %T') $*" | tee -a "$LOG"; }

pids_alive(){ for p in $WATCH_PIDS; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }

log "waiting for v4 fullval PIDs [$WATCH_PIDS] to exit, then GPU $GPU_A/$GPU_B >= ${NEED_FREE_MIB}MiB free"
while pids_alive; do sleep "$POLL"; done
log "v4 fullval PIDs gone; waiting for GPU memory to free"
fa=""; fb=""
while :; do
  fa=$(gpu_free "$GPU_A"); fb=$(gpu_free "$GPU_B")
  if [ -n "$fa" ] && [ -n "$fb" ] && [ "$fa" -ge "$NEED_FREE_MIB" ] && [ "$fb" -ge "$NEED_FREE_MIB" ]; then break; fi
  sleep "$POLL"
done
log "GPU $GPU_A free=${fa}MiB, GPU $GPU_B free=${fb}MiB"

# --- pick best v5 checkpoint by 48-val accuracy (dir must still exist); fallback latest ---
if [ -n "$ADAPTER_OVERRIDE" ]; then
  SEL="$ADAPTER_OVERRIDE	override	NA"
else
  SEL=$(python3 - "$RUN" <<'PY'
import glob, json, os, sys
run = sys.argv[1]
best = None
for f in glob.glob(os.path.join(run, "eval", "eval_step-*_summary.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    lab = str(d.get("step_label", ""))
    digits = "".join(ch for ch in lab if ch.isdigit())
    if not digits:
        continue
    step = int(digits)
    ck = os.path.join(run, "checkpoints", f"checkpoint-{step:06d}")
    if not os.path.isdir(ck):
        continue
    acc = d.get("accuracy") or 0.0
    if best is None or acc > best[0]:
        best = (acc, step, ck)
if best:
    print(f"{best[2]}\t{best[1]}\t{best[0]:.4f}")
else:
    p = os.path.join(run, "checkpoints", "latest_checkpoint.txt")
    if os.path.exists(p):
        print(open(p).read().strip() + "\tlatest\tNA")
PY
)
fi
ADAPTER=$(printf '%s' "$SEL" | cut -f1)
STEP=$(printf '%s' "$SEL" | cut -f2)
ACC=$(printf '%s' "$SEL" | cut -f3)
if [ -z "$ADAPTER" ] || [ ! -d "$ADAPTER" ]; then
  log "ERROR: no usable v5 checkpoint found (SEL='$SEL')"; exit 1
fi
LABEL="v5_step${STEP}_fullval_official"
log "selected checkpoint $ADAPTER (48val step=$STEP acc=$ACC) -> label $LABEL"

launch_shard(){
  local gpu=$1 idx=$2
  GPU_ID=$gpu \
  NUM_SHARDS=2 SHARD_INDEX=$idx \
  ADAPTER_DIR=$ADAPTER \
  TRAIN_CSV=$TRAIN_CSV \
  VAL_MAX_SAMPLES=0 \
  EVAL_MAX_NEW_TOKENS=$EVAL_MAX_NEW_TOKENS \
  MAX_SEQ_LEN=$MAX_SEQ_LEN \
  INFERENCE_FINAL_ANSWER_PREFILL=0 \
  INFERENCE_STOP_AFTER_BOXED=1 \
  OFFICIAL_PROMPT_ALIGN=1 \
  EVAL_LABEL=${LABEL}_s${idx} \
  EVAL_OUTPUT_DIR=$RUN/eval \
  nohup .venv/bin/python eval_adapter.py >> "$RUN/eval/${LABEL}_s${idx}.log" 2>&1 &
  echo $!
}

P0=$(launch_shard "$GPU_A" 0); log "shard0 PID $P0 on GPU $GPU_A -> $RUN/eval/${LABEL}_s0.log"
P1=$(launch_shard "$GPU_B" 1); log "shard1 PID $P1 on GPU $GPU_B -> $RUN/eval/${LABEL}_s1.log"
echo "$P0 $P1" > "$RUN/eval/${LABEL}.pids"

wait "$P0"; r0=$?
wait "$P1"; r1=$?
log "shards finished (rc0=$r0 rc1=$r1)"

S0=$RUN/eval/eval_${LABEL}_s0.jsonl
S1=$RUN/eval/eval_${LABEL}_s1.jsonl
if [ -s "$S0" ] && [ -s "$S1" ]; then
  .venv/bin/python scripts/merge_shard_eval.py "$S0" "$S1" \
    --out "$RUN/eval/eval_${LABEL}_merged_summary.json" >> "$LOG" 2>&1
  log "merged -> $RUN/eval/eval_${LABEL}_merged_summary.json"
else
  log "ERROR: shard jsonl missing/empty, not merging (S0='$S0' S1='$S1')"
fi
log "done"
