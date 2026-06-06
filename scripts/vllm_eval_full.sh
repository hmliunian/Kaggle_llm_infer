#!/usr/bin/env bash
# Fast batched vLLM eval of the latest v5 checkpoint, 2-shard data-parallel on GPU5/6,
# scored byte-for-byte with the HF eval's official metric (stop-after-boxed, 512 new
# tokens, official prompt align, base-only official val). Pipeline:
#   stage1 build_prompts (training venv) -> stage2 vLLM generate (vllm venv) -> stage3 score.
# The adapter is snapshotted to a rotation-safe dir so training's checkpoint pruning
# can't delete it mid-eval.
set -euo pipefail

ROOT="${ROOT_DIR:-/data2/yaoxuran/llm_infer}"
RUN="$ROOT/runs/cot_v5_gpu3_from_base_raw"
cd "$ROOT"

CKPT="${CKPT:-$(tr -d '[:space:]' < "$RUN/checkpoints/latest_checkpoint.txt")}"
STEP="$(basename "$CKPT" | grep -oE '[0-9]+')"
VAL_PER_SHARD="${VAL_PER_SHARD:-0}"          # 0 = full official val (~947), split across 2 shards
GPU0="${GPU0:-5}"; GPU1="${GPU1:-6}"
MAXNEW="${EVAL_MAX_NEW_TOKENS:-512}"
TRAIN_CSV="${TRAIN_CSV:-$ROOT/data/train_plus_synthetic_v5.csv}"
LABEL="v5_step${STEP}_vllm_$([ "$VAL_PER_SHARD" = 0 ] && echo full || echo $((VAL_PER_SHARD*2)))_t${MAXNEW}"
OUT="$RUN/eval/vllm_full"
mkdir -p "$OUT"

# --- snapshot adapter (config + weights only) to a stable dir ---
ADP="$OUT/adapter_step${STEP}"
mkdir -p "$ADP"
cp -f "$CKPT/adapter_config.json" "$CKPT/adapter_model.safetensors" "$ADP/"
echo "[full] checkpoint=$CKPT step=$STEP label=$LABEL per_shard=$VAL_PER_SHARD gpus=$GPU0,$GPU1 maxnew=$MAXNEW"
echo "[full] adapter snapshot -> $ADP"

COMMON_ENV=(TRAIN_CSV="$TRAIN_CSV" EVAL_BASE_ONLY=1 OFFICIAL_PROMPT_ALIGN=1
  INFERENCE_FINAL_ANSWER_PREFILL=0 INFERENCE_STOP_AFTER_BOXED=1
  EVAL_MAX_NEW_TOKENS="$MAXNEW" MAX_SEQ_LEN=1024 VAL_MAX_SAMPLES="$VAL_PER_SHARD")

# --- stage 1: build per-shard prompts (training venv, CPU) ---
for SH in 0 1; do
  env PYTHONPATH="$ROOT" "${COMMON_ENV[@]}" NUM_SHARDS=2 SHARD_INDEX="$SH" CUDA_VISIBLE_DEVICES="" \
    "$ROOT/.venv/bin/python" scripts/vllm_build_prompts.py --out "$OUT/${LABEL}_prompts_s${SH}.jsonl"
done

# --- stage 2: vLLM generate per shard, one GPU each (vllm venv) ---
gen_shard(){ local gpu=$1 sh=$2
  env CUDA_VISIBLE_DEVICES="$gpu" CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=9.0a VLLM_LOGGING_LEVEL=WARNING \
    ADAPTER_DIR="$ADP" EVAL_MAX_NEW_TOKENS="$MAXNEW" MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}" GPU_MEM_UTIL=0.90 \
    "$ROOT/.venv-vllm/bin/python" scripts/vllm_eval_run.py \
      --prompts "$OUT/${LABEL}_prompts_s${sh}.jsonl" --out "$OUT/${LABEL}_compl_s${sh}.jsonl" \
      > "$OUT/${LABEL}_vllm_s${sh}.log" 2>&1
}
gen_shard "$GPU0" 0 & P0=$!
gen_shard "$GPU1" 1 & P1=$!
echo "[full] vLLM shard0 pid=$P0 gpu=$GPU0 ; shard1 pid=$P1 gpu=$GPU1"
r0=0; r1=0
wait "$P0" || r0=$?
wait "$P1" || r1=$?
echo "[full] vLLM shards done rc0=$r0 rc1=$r1"
if [ "$r0" -ne 0 ] || [ "$r1" -ne 0 ]; then
  echo "[full] ERROR: a vLLM shard failed (see ${OUT}/${LABEL}_vllm_s*.log)"; exit 1
fi

# --- stage 3: score merged (training venv) ---
env PYTHONPATH="$ROOT" "${COMMON_ENV[@]}" \
  "$ROOT/.venv/bin/python" scripts/vllm_eval_score.py \
    --prompts "$OUT/${LABEL}_prompts_s0.jsonl,$OUT/${LABEL}_prompts_s1.jsonl" \
    --completions "$OUT/${LABEL}_compl_s0.jsonl,$OUT/${LABEL}_compl_s1.jsonl" \
    --summary "$OUT/${LABEL}_summary.json" \
    --records "$OUT/${LABEL}_records.jsonl" --label "$LABEL"
echo "[full] DONE -> $OUT/${LABEL}_summary.json"
