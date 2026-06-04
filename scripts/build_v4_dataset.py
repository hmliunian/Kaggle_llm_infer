#!/usr/bin/env python3
"""Build a CoT-augmented v4 training set from train_plus_synthetic_v3.csv.

For every row we try to SOLVE the task using only the examples shown in the
prompt (no hidden payload), build a ground-truth chain-of-thought, and self-check
that the reasoning reproduces the gold answer (official tolerance). This doubles
as a solvability audit of the v3 data: rows that cannot be solved from their own
examples, or whose examples are ambiguous, are flagged.

Output:
  - data/train_plus_synthetic_v4.csv : v3 columns + `cot` + `has_cot`
  - data/train_plus_synthetic_v4_report.json : per source x family audit

Rows that fail get `cot=""`, `has_cot=False`; they are kept (answer-only) so no
data is lost, but the report surfaces them as potential problems.

Usage:
    python scripts/build_v4_dataset.py [--sample N] [--in CSV] [--out CSV]
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl

from cot_builders import build_cot, CotError


def fail_category(msg: str) -> str:
    if "ambiguous" in msg:
        return "ambiguous_examples"
    if "uniquely identifiable" in msg:
        return "bit_rule_not_unique"
    if "missing from examples" in msg:
        return "query_token_missing"
    if "self-check failed" in msg:
        return "computed_ne_gold"
    if "could not parse" in msg:
        return "parse_error"
    if "not shown in examples" in msg:
        return "query_op_missing"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default="data/train_plus_synthetic_v3.csv")
    ap.add_argument("--out", dest="out_csv", default="data/train_plus_synthetic_v4.csv")
    ap.add_argument("--report", default="data/train_plus_synthetic_v4_report.json")
    ap.add_argument("--sample", type=int, default=0, help="audit only N rows, do not write CSV")
    args = ap.parse_args()

    df = pl.read_csv(args.in_csv, schema_overrides={"prompt": pl.String, "answer": pl.String})
    rows = df.to_dicts()
    if args.sample:
        import random
        random.Random(0).shuffle(rows)
        rows = rows[: args.sample]

    cots, has = [], []
    ok = defaultdict(int)
    tot = defaultdict(int)
    fails = defaultdict(Counter)
    fail_examples = defaultdict(list)

    for r in rows:
        fam = r["family"]
        src = r.get("source", "unknown")
        key = (src, fam)
        tot[key] += 1
        try:
            cot = build_cot(fam, r["prompt"], str(r["answer"]), None)
            cots.append(cot)
            has.append(True)
            ok[key] += 1
        except CotError as e:
            cat = fail_category(str(e))
            fails[key][cat] += 1
            if len(fail_examples[(key, cat)]) < 2:
                fail_examples[(key, cat)].append({"id": r.get(" id"), "answer": r["answer"], "msg": str(e)[:160]})
            cots.append("")
            has.append(False)

    # ── report ──
    report = {"input": args.in_csv, "total_rows": len(rows), "by_source_family": {}}
    print(f"\n{'source':14s} {'family':32s} {'cot_ok':>10}  fail_breakdown")
    for key in sorted(tot):
        src, fam = key
        o, t = ok[key], tot[key]
        fb = dict(fails[key])
        print(f"{src:14s} {fam:32s} {o:>5}/{t:<4} ({o/t:.2f})  {fb if fb else ''}")
        report["by_source_family"][f"{src}|{fam}"] = {
            "total": t, "cot_ok": o, "cot_rate": o / t, "fail_breakdown": fb,
        }
    total_ok = sum(ok.values())
    report["total_cot_ok"] = total_ok
    report["total_cot_rate"] = total_ok / max(len(rows), 1)
    report["fail_examples"] = {f"{k[0][0]}|{k[0][1]}|{k[1]}": v for k, v in fail_examples.items()}
    print(f"\nTOTAL CoT ok: {total_ok}/{len(rows)} ({total_ok/len(rows):.3f})")

    if not args.sample:
        out = df.with_columns(
            pl.Series("cot", cots), pl.Series("has_cot", has)
        )
        out.write_csv(args.out_csv)
        print(f"\nWrote {args.out_csv} ({len(out)} rows, {sum(has)} with CoT)")
    with open(args.report, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
