"""Join HF (eval_adapter) records and vLLM records by sample_index and report
faithfulness: per-sample answer agreement, correctness agreement, and aggregate
boxed-rate / accuracy on the SAME rows + checkpoint. High agreement => vLLM applies
the (Mamba + MoE-expert) LoRA faithfully and its accuracy is trustworthy.
"""
import argparse
import json


def load(path, keys):
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        d = json.loads(line)
        out[d["sample_index"]] = {k: d.get(k) for k in keys}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--vllm", required=True)
    args = ap.parse_args()

    hf = load(args.hf, ["pred", "correct", "has_boxed_answer", "gold", "family"])
    vl = load(args.vllm, ["pred", "correct", "has_boxed_answer", "gold", "family"])
    common = sorted(set(hf) & set(vl))
    n = len(common)

    pred_match = sum(str(hf[i]["pred"]).strip() == str(vl[i]["pred"]).strip() for i in common)
    correct_match = sum(bool(hf[i]["correct"]) == bool(vl[i]["correct"]) for i in common)
    hf_acc = sum(bool(hf[i]["correct"]) for i in common) / n
    vl_acc = sum(bool(vl[i]["correct"]) for i in common) / n
    hf_box = sum(bool(hf[i]["has_boxed_answer"]) for i in common) / n
    vl_box = sum(bool(vl[i]["has_boxed_answer"]) for i in common) / n
    both_correct = sum(bool(hf[i]["correct"]) and bool(vl[i]["correct"]) for i in common)

    print(f"n_common={n}")
    print(f"PRED exact-match   : {pred_match}/{n} = {pred_match/n:.1%}")
    print(f"CORRECT agreement  : {correct_match}/{n} = {correct_match/n:.1%}")
    print(f"HF   accuracy={hf_acc:.3f}  boxed_rate={hf_box:.3f}")
    print(f"vLLM accuracy={vl_acc:.3f}  boxed_rate={vl_box:.3f}")
    print(f"both-correct={both_correct}")
    print("\n--- disagreements (pred differs) ---")
    shown = 0
    for i in common:
        if str(hf[i]["pred"]).strip() != str(vl[i]["pred"]).strip():
            print(f"  s{i} fam={hf[i]['family']} gold={hf[i]['gold']!r} "
                  f"HF_pred={hf[i]['pred']!r}(ok={hf[i]['correct']}) "
                  f"vLLM_pred={vl[i]['pred']!r}(ok={vl[i]['correct']})")
            shown += 1
        if shown >= 25:
            print("  ...")
            break

    verdict = "FAITHFUL" if correct_match / n >= 0.9 else "DIVERGENT"
    print(f"\nVERDICT: {verdict} (correctness-agreement {correct_match/n:.1%})")


if __name__ == "__main__":
    main()
