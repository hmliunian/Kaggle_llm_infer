#!/usr/bin/env python3
"""Merge sharded eval jsonl files into one combined official-metric summary.

Each shard's eval_*.jsonl already has per-sample `correct` (from official_verify)
and `has_boxed_answer`, so merging is just re-aggregating counts. Usage:

    python scripts/merge_shard_eval.py runs/.../eval/eval_<label>_s0.jsonl \
                                       runs/.../eval/eval_<label>_s1.jsonl \
        [--out runs/.../eval/eval_<label>_merged_summary.json]
"""
import argparse
import json
from collections import defaultdict


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+", help="shard eval_*.jsonl files")
    ap.add_argument("--out", default=None, help="combined summary json path")
    args = ap.parse_args()

    rows = []
    for p in args.jsonl:
        rows.extend(load(p))

    total = len(rows)
    correct = sum(1 for r in rows if r.get("correct"))
    boxed = sum(1 for r in rows if r.get("has_boxed_answer"))

    fam = defaultdict(lambda: [0, 0, 0])  # family -> [correct, total, boxed]
    src = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        f = r.get("family", "?")
        s = r.get("source", "?")
        fam[f][1] += 1
        src[s][1] += 1
        if r.get("correct"):
            fam[f][0] += 1
            src[s][0] += 1
        if r.get("has_boxed_answer"):
            fam[f][2] += 1
            src[s][2] += 1

    summary = {
        "num_total": total,
        "num_correct": correct,
        "accuracy": correct / max(total, 1),
        "boxed_rate": boxed / max(total, 1),
        "family_summary": {
            k: {"correct": v[0], "total": v[1], "accuracy": v[0] / max(v[1], 1),
                "boxed_rate": v[2] / max(v[1], 1)}
            for k, v in sorted(fam.items())
        },
        "source_summary": {
            k: {"correct": v[0], "total": v[1], "accuracy": v[0] / max(v[1], 1)}
            for k, v in sorted(src.items())
        },
        "shards": args.jsonl,
    }

    print(f"MERGED  acc = {correct}/{total} = {summary['accuracy']:.4f}   "
          f"boxed = {boxed}/{total} = {summary['boxed_rate']:.4f}")
    print("by family:")
    for k, v in summary["family_summary"].items():
        print(f"  {k:32s} {v['correct']:4d}/{v['total']:<4d} = {v['accuracy']:.3f}"
              f"   boxed {v['boxed_rate']:.3f}")
    print("by source:")
    for k, v in summary["source_summary"].items():
        print(f"  {k:20s} {v['correct']:4d}/{v['total']:<4d} = {v['accuracy']:.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
