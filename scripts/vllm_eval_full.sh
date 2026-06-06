#!/usr/bin/env bash
# Batched vLLM eval of a v5 checkpoint on the base-only official val, 2-shard
# data-parallel on GPU5/6. Mirrors the official Kaggle metric's generate_predictions
# (docs/nvidia-nemotron-metric.ipynb): same vLLM engine config + sampling params, no
# stop-after-boxed, extract_final_answer on the raw generation (OFFICIAL_SCORING=1).
# Pipeline: stage1 build_prompts (training venv) -> stage2 vLLM generate (vllm venv)
# -> stage3 score (training venv). Adapter is snapshotted to a rotation-safe dir.
#
# Defaults are the official FINAL eval params, except max_lora_rank is inferred
# from adapter_config.json. The current adapter is rank 16. Set MAX_LORA_RANK=32
# for a rank-32 adapter or any manual override.
#   max_lora_rank=16  max_tokens=7680  top_p=1.0  temperature=0.0
#   max_num_seqs=64   gpu_memory_utilization=0.85  max_model_len=8192
# (max_tokens = generation budget = the CoT length knob; max_model_len = total ctx.)
# Override any via env. ENFORCE_EAGER=1 swaps CUDA graphs for eager (speed only; the
# greedy accuracy is identical) if graph capture is problematic on this box.
set -euo pipefail

ROOT="${ROOT_DIR:-/data2/yaoxuran/llm_infer}"
RUN="$ROOT/runs/cot_v5_gpu3_from_base_raw"
cd "$ROOT"

CKPT="${CKPT:-$(tr -d '[:space:]' < "$RUN/checkpoints/latest_checkpoint.txt")}"
STEP="$(basename "$CKPT" | grep -oE '[0-9]+')"
VAL_PER_SHARD="${VAL_PER_SHARD:-0}"          # 0 = full official val (~947), split across 2 shards
GPU0="${GPU0:-5}"; GPU1="${GPU1:-6}"
TRAIN_CSV="${TRAIN_CSV:-$ROOT/data/train_plus_synthetic_v5.csv}"

# --- official final eval params (override via env) ---
MAX_TOKENS="${MAX_TOKENS:-${EVAL_MAX_NEW_TOKENS:-7680}}"  # SamplingParams.max_tokens
TOP_P="${TOP_P:-1.0}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_LORA_RANK="${MAX_LORA_RANK:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${GPU_MEM_UTIL:-0.85}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
OFFICIAL_SCORING="${OFFICIAL_SCORING:-1}"

OUT="$RUN/eval/vllm_full"; mkdir -p "$OUT"
SIZE=$([ "$VAL_PER_SHARD" = 0 ] && echo full || echo $((VAL_PER_SHARD*2)))
LABEL="v5_step${STEP}_vllm_${SIZE}_t${MAX_TOKENS}$([ "$OFFICIAL_SCORING" = 1 ] && echo _official)"

# --- snapshot adapter (config + weights only) to a stable dir ---
ADP="$OUT/adapter_step${STEP}"; mkdir -p "$ADP"
cp -f "$CKPT/adapter_config.json" "$CKPT/adapter_model.safetensors" "$ADP/"
if [ -z "$MAX_LORA_RANK" ]; then
  MAX_LORA_RANK="$("$ROOT/.venv/bin/python" -c 'import json, sys
try:
    cfg = json.load(open(sys.argv[1]))
    ranks = [int(cfg.get("r") or 0)]
    ranks += [int(v) for v in (cfg.get("rank_pattern") or {}).values()]
    ranks = [r for r in ranks if r > 0]
    print(max(ranks) if ranks else 16)
except Exception:
    print(16)
' "$ADP/adapter_config.json" 2>/dev/null || printf '16\n')"
fi
echo "[full] checkpoint=$CKPT step=$STEP label=$LABEL per_shard=$VAL_PER_SHARD gpus=$GPU0,$GPU1"
echo "[full] params: max_tokens=$MAX_TOKENS top_p=$TOP_P temp=$TEMPERATURE max_num_seqs=$MAX_NUM_SEQS"
echo "[full]         max_lora_rank=$MAX_LORA_RANK gpu_memory_utilization=$GPU_MEMORY_UTILIZATION max_model_len=$MAX_MODEL_LEN"
echo "[full]         eager=$ENFORCE_EAGER prefix_cache=$ENABLE_PREFIX_CACHING chunked_prefill=$ENABLE_CHUNKED_PREFILL official_scoring=$OFFICIAL_SCORING"
echo "[full] adapter snapshot -> $ADP"

# Prompt build: official prompt align, enable_thinking, NO answer prefill, no truncation
# (MAX_SEQ_LEN headroom). OFFICIAL_SCORING propagates to stage 3.
COMMON_ENV=(TRAIN_CSV="$TRAIN_CSV" EVAL_BASE_ONLY=1 OFFICIAL_PROMPT_ALIGN=1
  INFERENCE_FINAL_ANSWER_PREFILL=0 INFERENCE_STOP_AFTER_BOXED=1 OFFICIAL_SCORING="$OFFICIAL_SCORING"
  MAX_TOKENS="$MAX_TOKENS" EVAL_MAX_NEW_TOKENS="$MAX_TOKENS" TOP_P="$TOP_P" TEMPERATURE="$TEMPERATURE"
  MAX_NUM_SEQS="$MAX_NUM_SEQS" MAX_LORA_RANK="$MAX_LORA_RANK"
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" MAX_MODEL_LEN="$MAX_MODEL_LEN"
  MAX_SEQ_LEN=2048 VAL_MAX_SAMPLES="$VAL_PER_SHARD")

# --- stage 1: build per-shard prompts (training venv, CPU) ---
for SH in 0 1; do
  env PYTHONPATH="$ROOT" "${COMMON_ENV[@]}" NUM_SHARDS=2 SHARD_INDEX="$SH" CUDA_VISIBLE_DEVICES="" \
    "$ROOT/.venv/bin/python" scripts/vllm_build_prompts.py --out "$OUT/${LABEL}_prompts_s${SH}.jsonl"
done

# --- stage 2: vLLM generate per shard, one GPU each (vllm venv) ---
gen_shard(){ local gpu=$1 sh=$2
  env CUDA_VISIBLE_DEVICES="$gpu" CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=9.0a VLLM_LOGGING_LEVEL=WARNING \
    ADAPTER_DIR="$ADP" MAX_TOKENS="$MAX_TOKENS" EVAL_MAX_NEW_TOKENS="$MAX_TOKENS" TOP_P="$TOP_P" TEMPERATURE="$TEMPERATURE" \
    MAX_NUM_SEQS="$MAX_NUM_SEQS" MAX_LORA_RANK="$MAX_LORA_RANK" GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" ENFORCE_EAGER="$ENFORCE_EAGER" \
    ENABLE_PREFIX_CACHING="$ENABLE_PREFIX_CACHING" ENABLE_CHUNKED_PREFILL="$ENABLE_CHUNKED_PREFILL" \
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
