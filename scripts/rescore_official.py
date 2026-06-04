#!/usr/bin/env python3
"""Re-score existing eval JSONL files with the official Kaggle metric.

Each historical `eval_*.jsonl` stores the raw `decoded` generation and the
`gold` answer, so we can recompute accuracy under the official grader WITHOUT
re-running the model. Useful for separating scoring artifacts from real model
differences.

Usage:
    python scripts/rescore_official.py <eval_step-*.jsonl> [more.jsonl ...]
"""

import json
import os
import sys
from collections import defaultdict

# Allow running from anywhere in the repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from official_metric import extract_final_answer, verify


def rescore(jsonl_path):
    total = official = stored = 0
    fam_total = defaultdict(int)
    fam_correct = defaultdict(int)
    for line in open(jsonl_path):
        rec = json.loads(line)
        fam = rec.get("family") or rec.get("source_family") or "?"
        gold = str(rec.get("gold"))
        ext = extract_final_answer(rec.get("decoded"))
        ok = verify(gold, ext)
        total += 1
        fam_total[fam] += 1
        if ok:
            official += 1
            fam_correct[fam] += 1
        if rec.get("correct"):
            stored += 1
    return total, official, stored, fam_total, fam_correct


def main(paths):
    for path in paths:
        total, official, stored, ft, fc = rescore(path)
        name = os.path.basename(path)
        print(
            f"{name:32s} official={official:3d}/{total} ({official/max(total,1):.3f})   "
            f"stored={stored:3d}/{total} ({stored/max(total,1):.3f})"
        )
        fams = sorted(ft)
        print("   " + "  ".join(f"{k}:{fc[k]}/{ft[k]}" for k in fams))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
