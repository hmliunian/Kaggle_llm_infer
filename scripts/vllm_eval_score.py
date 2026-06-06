"""
Stage 3 (TRAINING venv): score vLLM completions with the SAME official metric and
stop/truncate semantics as train_sft.evaluate(), so the accuracy is directly
comparable to the HF evals (steps 200-800) the run already produced.

HF eval stops generation at the first "}." that follows a "\\boxed{"
(StopAfterBoxClose) and then truncate_after_first_boxed(). vLLM generated the full
window, so we first emulate that exact cut, then reuse train_sft's truncate +
official_extract + official_verify.

Accepts comma-separated shard files; merges into one summary.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_sft
from official_metric import extract_final_answer as official_extract_final_answer
from official_metric import verify as official_verify

_BOXED_OPEN = "\\boxed{"
_TERM = "}."


def emulate_hf_stop(text: str) -> str:
    """Reproduce StopAfterBoxClose: cut at the first '}.' that follows a '\\boxed{'.

    Matches train_sft's incremental stop (which uses rfind(boxed) each step): for
    any number of boxed spans, the first firing is the first '}.' occurring after
    the earliest boxed open. If no boxed or no terminator, HF would run to the token
    cap, so we return the text unchanged.
    """
    open_idx = text.find(_BOXED_OPEN)
    if open_idx < 0:
        return text
    term = text.find(_TERM, open_idx + len(_BOXED_OPEN))
    if term < 0:
        return text
    return text[: term + len(_TERM)]


def load_pairs(prompts_files, completions_files):
    rows = {}
    for pf in prompts_files:
        for line in open(pf):
            if not line.strip():
                continue
            r = json.loads(line)
            rows[(pf, r["sample_index"])] = r
    # index completions by file+sample_index, matching shard order
    out = []
    for pf, cf in zip(prompts_files, completions_files):
        prm = {}
        for line in open(pf):
            if line.strip():
                r = json.loads(line)
                prm[r["sample_index"]] = r
        for line in open(cf):
            if not line.strip():
                continue
            c = json.loads(line)
            p = prm[c["sample_index"]]
            out.append((pf, p, c))
    return out


def env_float_any(names, default):
    for name in names:
        v = os.environ.get(name)
        if v not in (None, ""):
            return float(v)
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="comma-separated stage-1 jsonl(s)")
    ap.add_argument("--completions", required=True, help="comma-separated stage-2 jsonl(s)")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--label", default="vllm_eval")
    args = ap.parse_args()

    pf = args.prompts.split(",")
    cf = args.completions.split(",")
    assert len(pf) == len(cf), "prompts/completions shard counts differ"
    pairs = load_pairs(pf, cf)

    # OFFICIAL_SCORING mirrors docs/nvidia-nemotron-metric.ipynb: no stop-after-boxed,
    # extract_final_answer on the RAW generation (takes the LAST boxed span). Default
    # on matches the competition metric; set OFFICIAL_SCORING=0 for HF eval parity.
    official_scoring = train_sft.env_bool("OFFICIAL_SCORING", True)
    eval_params = {
        "max_lora_rank": train_sft.env_int("MAX_LORA_RANK", 16),
        "max_tokens": train_sft.env_int(
            "MAX_TOKENS",
            train_sft.env_int("EVAL_MAX_NEW_TOKENS", 7680),
        ),
        "top_p": env_float_any(("TOP_P",), 1.0),
        "temperature": env_float_any(("TEMPERATURE",), 0.0),
        "max_num_seqs": train_sft.env_int("MAX_NUM_SEQS", 64),
        "gpu_memory_utilization": env_float_any(("GPU_MEMORY_UTILIZATION", "GPU_MEM_UTIL"), 0.85),
        "max_model_len": train_sft.env_int("MAX_MODEL_LEN", 8192),
    }

    correct = total = boxed_count = 0
    fam_c = defaultdict(int); fam_t = defaultdict(int); fam_b = defaultdict(int)
    src_c = defaultdict(int); src_t = defaultdict(int); src_b = defaultdict(int)
    n_hit_cap = 0
    records = []

    for _, p, c in pairs:
        answer_prefix = p.get("answer_prefix", "")
        full = answer_prefix + c["decoded_raw"]
        if official_scoring:
            decoded = full  # official: full generation; extract_final_answer takes LAST boxed
        else:
            stopped = emulate_hf_stop(full)
            decoded = train_sft.truncate_after_first_boxed(stopped) \
                if train_sft.INFERENCE_STOP_AFTER_BOXED else stopped
        boxed_pred = train_sft.extract_boxed(decoded)
        pred = official_extract_final_answer(decoded)
        gold = p["gold"]
        fam = p["family"]
        src = p["source"]
        ok = official_verify(gold, pred)
        has_boxed = boxed_pred is not None

        total += 1; fam_t[fam] += 1; src_t[src] += 1
        if ok: correct += 1; fam_c[fam] += 1; src_c[src] += 1
        if has_boxed: boxed_count += 1; fam_b[fam] += 1; src_b[src] += 1
        if c.get("finish_reason") == "length":
            n_hit_cap += 1

        records.append({
            "sample_index": p["sample_index"],
            "family": fam, "source": src,
            "correct": ok, "has_boxed_answer": has_boxed,
            "gold": gold, "pred": pred, "boxed_pred": boxed_pred,
            "num_gen_tokens": c.get("num_gen_tokens"),
            "finish_reason": c.get("finish_reason"),
            "decoded": decoded,
        })

    acc = correct / total if total else 0.0
    summary = {
        "label": args.label,
        "scoring": "official" if official_scoring else "train_sft_parity",
        "eval_params": eval_params,
        "accuracy": acc,
        "num_total": total,
        "num_correct": correct,
        "boxed_rate": boxed_count / total if total else 0.0,
        "frac_hit_token_cap": n_hit_cap / total if total else 0.0,
        "family": {
            k: {"acc": fam_c[k] / fam_t[k], "n": fam_t[k],
                "boxed": fam_b[k] / fam_t[k]}
            for k in sorted(fam_t)
        },
        "source": {
            k: {"acc": src_c[k] / src_t[k], "n": src_t[k]}
            for k in sorted(src_t)
        },
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    with open(args.records, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("family", "source")}, indent=2))
    print("by family:")
    for k, v in summary["family"].items():
        print(f"  {k:32s} acc={v['acc']:.3f} n={v['n']} boxed={v['boxed']:.3f}")
    print(f"summary -> {args.summary}")
    print(f"records -> {args.records}")


if __name__ == "__main__":
    main()
