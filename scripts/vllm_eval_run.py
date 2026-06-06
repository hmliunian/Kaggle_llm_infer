"""
Stage 2 (vLLM venv): batched generation with the LoRA adapter, mirroring the
official Kaggle metric's `generate_predictions` (docs/nvidia-nemotron-metric.ipynb).

Engine + sampling config are parameterized via env to match the official harness:
  MAX_LORA_RANK, MAX_TOKENS (= SamplingParams.max_tokens, the *generation*
  budget; same knob as HF max_new_tokens / the CoT length we sweep), TOP_P,
  TEMPERATURE, MAX_NUM_SEQS, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN
  (total context = prompt + generated tokens). EVAL_MAX_NEW_TOKENS and
  GPU_MEM_UTIL remain accepted as compatibility aliases.

Official `generate_predictions` does NOT use a stop string and does NOT stop after
\\boxed{} — it generates to natural EOS (or max_tokens) and the metric takes the LAST
boxed span. We match that: no stop string here; stage 3 extracts with OFFICIAL_SCORING.

Prompts arrive pre-tokenized from stage 1 (byte-identical model input, no cross-venv
tokenizer drift). This is equivalent to the official string-prompt path since stage 1
builds the same chat template (official prompt align, enable_thinking, no prefill).

Env: ADAPTER_DIR, MODEL_PATH, MAX_TOKENS, TOP_P, TEMPERATURE, MAX_NUM_SEQS,
MAX_LORA_RANK, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, ENFORCE_EAGER,
ENABLE_PREFIX_CACHING, ENABLE_CHUNKED_PREFILL. CUDA_VISIBLE_DEVICES picks the GPU;
CUDA_HOME/PATH must point at a >=12.x nvcc for flashinfer JIT on H100 (sm_90a).
"""
import argparse
import json
import os
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def _envf(name, default):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def _envf_any(names, default):
    for name in names:
        v = os.environ.get(name)
        if v not in (None, ""):
            return float(v)
    return default


def _envi(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _envi_any(names, default):
    for name in names:
        v = os.environ.get(name)
        if v not in (None, ""):
            return int(v)
    return default


def _envb(name, default):
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def infer_lora_rank(adapter_dir, default=16):
    try:
        cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
        ranks = [int(cfg.get("r") or 0)]
        ranks.extend(int(v) for v in (cfg.get("rank_pattern") or {}).values())
        ranks = [r for r in ranks if r > 0]
        return max(ranks) if ranks else default
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_path = os.environ.get("MODEL_PATH", "/data2/yaoxuran/models")
    adapter_dir = os.environ["ADAPTER_DIR"]

    # --- sampling / engine params ---
    # Current adapters are rank 16; MAX_LORA_RANK can still override the config.
    max_tokens = _envi_any(("MAX_TOKENS", "EVAL_MAX_NEW_TOKENS"), 7680)
    top_p = _envf("TOP_P", 1.0)
    temperature = _envf("TEMPERATURE", 0.0)
    max_num_seqs = _envi("MAX_NUM_SEQS", 64)
    max_lora_rank = _envi("MAX_LORA_RANK", infer_lora_rank(adapter_dir, 16))
    gpu_util = _envf_any(("GPU_MEMORY_UTILIZATION", "GPU_MEM_UTIL"), 0.85)
    max_model_len = _envi("MAX_MODEL_LEN", 8192)
    enforce_eager = _envb("ENFORCE_EAGER", False)            # official: not set (CUDA graphs)
    enable_prefix_caching = _envb("ENABLE_PREFIX_CACHING", True)
    enable_chunked_prefill = _envb("ENABLE_CHUNKED_PREFILL", True)

    rows = [json.loads(l) for l in open(args.prompts) if l.strip()]
    token_prompts = [{"prompt_token_ids": r["prompt_token_ids"]} for r in rows]
    print(f"[vllm_run] {len(rows)} prompts | max_tokens={max_tokens} top_p={top_p} "
          f"temp={temperature} max_num_seqs={max_num_seqs} max_lora_rank={max_lora_rank} "
          f"gpu_memory_utilization={gpu_util} max_model_len={max_model_len} eager={enforce_eager} "
          f"prefix_cache={enable_prefix_caching} chunked_prefill={enable_chunked_prefill} "
          f"| adapter={adapter_dir}", flush=True)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_util,
        dtype="auto",
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        enforce_eager=enforce_eager,
    )
    # Match official generate_predictions: temperature, top_p, max_tokens; no stop string.
    sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    lreq = LoRARequest("adapter", 1, adapter_dir)

    outs = llm.generate(token_prompts, sp, lora_request=lreq)

    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({
                "sample_index": r["sample_index"],
                "decoded_raw": o.outputs[0].text,
                "num_gen_tokens": len(o.outputs[0].token_ids),
                "finish_reason": o.outputs[0].finish_reason,
            }) + "\n")
    print(f"[vllm_run] wrote {len(outs)} completions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
