#!/usr/bin/env python3
"""Split rule_unknown problems into per-id prompt files + clean index arrays.

- data/ru_problems/<id>.txt        : the raw puzzle prompt (read by workflow agents)
- data/ru_index.json               : [{id, category, family}, ...] for all 918
- data/ru_smoke_index.json         : balanced 12-item subset for the smoke test
The index arrays contain no escaping-tricky characters, so they are safe to pass
verbatim as Workflow `args`.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "rule_unknown_problems.jsonl"
PDIR = ROOT / "data" / "ru_problems"


def main() -> None:
    PDIR.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in SRC.open() if l.strip()]
    index = []
    by_cat = defaultdict(list)
    for r in recs:
        (PDIR / f"{r['id']}.txt").write_text(r["prompt"])
        item = {"id": r["id"], "category": r["category"], "family": r["family"]}
        index.append(item)
        by_cat[r["category"]].append(item)

    (ROOT / "data" / "ru_index.json").write_text(json.dumps(index, ensure_ascii=False))

    plan = {
        "bit_manipulation": 3,
        "cryptarithm_deduce": 3,
        "cryptarithm_guess": 2,
        "equation_numeric_deduce": 2,
        "equation_numeric_guess": 2,
    }
    smoke = []
    for cat, n in plan.items():
        smoke.extend(by_cat[cat][:n])
    (ROOT / "data" / "ru_smoke_index.json").write_text(json.dumps(smoke, ensure_ascii=False))

    print(f"wrote {len(index)} prompt files to {PDIR}")
    print(f"smoke index ({len(smoke)}):")
    print(json.dumps(smoke, ensure_ascii=False))


if __name__ == "__main__":
    main()
