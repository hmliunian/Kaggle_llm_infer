#!/usr/bin/env python3
"""Gather the 918 rule_unknown official problems (bit + equation) into a JSONL.

Joins extern/nemotron/problems.jsonl (status, granular category) with the
dataset CSV (prompt, gold answer, local family). Output: one JSON object per
line with id, family, category, prompt, answer.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "train_plus_synthetic_v6.csv"))
    ap.add_argument("--problems", default=str(ROOT / "extern" / "nemotron" / "problems.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "rule_unknown_problems.jsonl"))
    args = ap.parse_args()

    status = {}
    category = {}
    with open(args.problems) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            status[e["id"]] = e.get("status")
            category[e["id"]] = e.get("category")

    rows = {}
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            pid = (r.get(" id") or r.get("id") or "").strip()
            rows[pid] = r

    out_path = Path(args.out)
    by_cat = Counter()
    by_fam = Counter()
    written = 0
    with out_path.open("w") as out:
        for pid, st in status.items():
            if st != "rule_unknown":
                continue
            r = rows.get(pid)
            if r is None:
                continue
            rec = {
                "id": pid,
                "family": r.get("family", ""),
                "category": category.get(pid, ""),
                "prompt": r.get("prompt", ""),
                "answer": r.get("answer", ""),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            by_cat[rec["category"]] += 1
            by_fam[rec["family"]] += 1

    print(f"Wrote {written} rule_unknown problems to {out_path}")
    print("by family:", dict(by_fam))
    print("by category:", dict(by_cat.most_common()))


if __name__ == "__main__":
    main()
