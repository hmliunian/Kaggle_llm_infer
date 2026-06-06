"""
Decisive test: can vLLM load + APPLY this PEFT LoRA adapter to the Nemotron-H
hybrid MoE model? The adapter targets Mamba mixer (in_proj/out_proj) AND per-expert
MoE layers (mixer.experts.N.up_proj/down_proj). We must confirm vLLM actually
applies the adapter (not silently dropping the MoE-expert LoRA), else any vLLM
accuracy number is wrong.

Checks:
  1) Does vLLM build + accept the LoRA adapter at all (load-time error?).
  2) Greedy generation WITH vs WITHOUT the adapter must DIFFER (adapter active).
  3) Report verbatim any warnings about skipped/unsupported LoRA modules.

Run inside the vLLM venv with CUDA_VISIBLE_DEVICES pointed at one free GPU.
"""
import csv
import os
import sys
import traceback

MODEL = os.environ.get("MODEL_PATH", "/data2/yaoxuran/models")
ADAPTER = os.environ.get(
    "ADAPTER_DIR",
    "/data2/yaoxuran/llm_infer/runs/cot_v5_gpu3_from_base_raw/checkpoints/checkpoint-000900",
)
CSV = os.environ.get("TRAIN_CSV", "/data2/yaoxuran/llm_infer/data/train_plus_synthetic_v5.csv")
OFFICIAL_BOXED_INSTRUCTION = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def grab_prompts(n=3):
    """A few representative official_train prompts (no model deps)."""
    out = []
    with open(CSV, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            src = (row.get("source") or "").strip()
            if src in ("", "official_train"):
                out.append(row["prompt"])
            if len(out) >= n:
                break
    return out


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    raw_prompts = grab_prompts(3)
    prompts = []
    for p in raw_prompts:
        messages = [{"role": "user", "content": p + OFFICIAL_BOXED_INSTRUCTION}]
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        except TypeError:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompts.append(text)
    print(f"[verify] built {len(prompts)} prompts; first prompt head:\n{prompts[0][:300]}\n---")

    print("[verify] building LLM (enable_lora=True) ...", flush=True)
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        max_model_len=1024,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        tensor_parallel_size=1,
    )

    sp = SamplingParams(temperature=0.0, max_tokens=160)

    print("[verify] generating WITHOUT adapter ...", flush=True)
    base_out = llm.generate(prompts, sp)
    base_txt = [o.outputs[0].text for o in base_out]

    print("[verify] generating WITH adapter (LoRARequest) ...", flush=True)
    lreq = LoRARequest("v5_900", 1, ADAPTER)
    lora_out = llm.generate(prompts, sp, lora_request=lreq)
    lora_txt = [o.outputs[0].text for o in lora_out]

    n_diff = 0
    for i, (b, l) in enumerate(zip(base_txt, lora_txt)):
        same = (b == l)
        n_diff += (not same)
        print(f"\n===== PROMPT {i}  identical={same} =====")
        print(f"--- BASE  ---\n{b[:400]}")
        print(f"--- LoRA  ---\n{l[:400]}")

    print(f"\n[verify] RESULT: {n_diff}/{len(prompts)} prompts changed by the adapter.")
    if n_diff == 0:
        print("[verify] VERDICT: FAIL — adapter had NO effect (LoRA not applied).")
        sys.exit(2)
    else:
        print("[verify] VERDICT: adapter changed outputs (applied to SOMETHING). "
              "Still must confirm MoE-expert coverage separately.")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:  # capture load-time unsupported-LoRA errors verbatim
        print("\n[verify] EXCEPTION during vLLM load/generate:")
        traceback.print_exc()
        # surface the most relevant line
        print(f"\n[verify] ERROR TYPE: {type(e).__name__}: {e}")
        sys.exit(3)
