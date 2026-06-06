"""
Stage 2 (vLLM venv): batched greedy generation with the LoRA adapter, consuming
the pre-tokenized prompts from stage 1 (byte-identical model input). Writes raw
completions; scoring/stop-emulation happens in stage 3 (training venv) for exact
parity with the HF eval's official metric.

We deliberately do NOT use a vLLM `stop` string: the HF eval stops at the first
"}." that follows a "\\boxed{", and replicating that precisely in post-processing
avoids the rare early-"}." trap. We generate up to EVAL_MAX_NEW_TOKENS and let
stage 3 cut at the HF-equivalent terminator.

Env: ADAPTER_DIR, MODEL_PATH, EVAL_MAX_NEW_TOKENS, MAX_MODEL_LEN, GPU_MEM_UTIL.
CUDA_VISIBLE_DEVICES selects the GPU. CUDA_HOME/PATH must point at a >=12.x nvcc
(H100/sm_90a) for flashinfer JIT.
"""
import argparse
import json
import os

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_path = os.environ.get("MODEL_PATH", "/data2/yaoxuran/models")
    adapter_dir = os.environ["ADAPTER_DIR"]
    max_new = int(os.environ.get("EVAL_MAX_NEW_TOKENS", "512"))
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "2048"))
    gpu_util = float(os.environ.get("GPU_MEM_UTIL", "0.90"))

    rows = [json.loads(l) for l in open(args.prompts) if l.strip()]
    # Preserve sample_index alignment; vLLM keeps output order = input order.
    token_prompts = [{"prompt_token_ids": r["prompt_token_ids"]} for r in rows]
    print(f"[vllm_run] {len(rows)} prompts | max_new={max_new} | adapter={adapter_dir}", flush=True)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        enforce_eager=True,
        tensor_parallel_size=1,
    )
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new, n=1)
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
