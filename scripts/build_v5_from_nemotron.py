#!/usr/bin/env python3
"""Build a v5 CoT dataset by importing verified solver traces from extern/nemotron.

The external repository contains deterministic reasoning traces for the 9,500
official train rows. This script imports only traces whose final boxed answer
matches our local gold answer under the official metric, then strips any
``\\boxed`` lines from the CoT so training still uses train_sft.py's single
canonical final-answer target.

By default we import traces for the official families that v4 could not solve
well from visible examples alone:

  - bit_manipulation
  - equation_symbol_transformation
  - text_decryption
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from official_metric import extract_final_answer, verify


DEFAULT_TARGET_FAMILIES = {
    "bit_manipulation",
    "equation_symbol_transformation",
    "text_decryption",
}


def row_id(row: dict[str, str]) -> str:
    return (row.get(" id") or row.get("id") or "").strip()


def trim_blank_edges(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def clean_lines(text: str) -> list[str]:
    lines = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        if "\\boxed" in line:
            continue
        if "<|im_end|>" in line:
            continue
        lines.append(line)
    return trim_blank_edges(lines)


def find_line(lines: list[str], prefix: str) -> int:
    for idx, line in enumerate(lines):
        if line.startswith(prefix):
            return idx
    return -1


def compact_reasoning(text: str, family: str, extern_category: str) -> str:
    """Keep the solver evidence that is most useful under a short seq budget."""
    lines = clean_lines(text)
    if not lines:
        return ""

    if family == "bit_manipulation":
        idx = find_line(lines, "Selected")
        if idx < 0:
            idx = find_line(lines, "Matched")
        if idx < 0:
            idx = find_line(lines, "Applying to ")
        if idx >= 0:
            lines = lines[idx:]

    elif family == "text_decryption":
        idx = find_line(lines, "Now decrypting")
        if idx >= 0:
            lines = lines[idx:]
        compacted: list[str] = []
        skipping_candidates = False
        for line in lines:
            if line.startswith("Let me find the best matching wonderland words"):
                compacted.append(line)
                skipping_candidates = True
                continue
            if skipping_candidates:
                if line.startswith("Best match:"):
                    compacted.append(line)
                    skipping_candidates = False
                continue
            compacted.append(line)
        lines = compacted

    elif family == "equation_symbol_transformation":
        if extern_category.startswith("equation_numeric"):
            action_lines = [
                line.strip()
                for line in lines
                if "correct, actions:" in line
            ]
            idx = find_line(lines, "Applying to ")
            kept: list[str] = []
            if action_lines:
                kept.append("Identified matching operation from examples:")
                kept.extend(action_lines[-2:])
                kept.append("")
            if idx >= 0:
                kept.extend(lines[idx:])
            if kept:
                lines = kept
        elif extern_category.startswith("cryptarithm"):
            idx = find_line(lines, "Question")
            if idx >= 0:
                lines = lines[idx:]

    lines = trim_blank_edges(lines)
    cot = "\n".join(lines).strip()
    # Do not allow a boxed answer inside <think>; StopAfterBoxClose would stop
    # generation before the canonical final-answer target.
    cot = cot.replace("\\boxed", "boxed")
    return cot


def load_problems(extern_root: Path) -> dict[str, dict]:
    problems = {}
    with (extern_root / "problems.jsonl").open() as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                problems[entry["id"]] = entry
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default="data/train_plus_synthetic_v4.csv")
    ap.add_argument("--out", dest="out_csv", default="data/train_plus_synthetic_v5.csv")
    ap.add_argument("--report", default="data/train_plus_synthetic_v5_report.json")
    ap.add_argument("--extern-root", default="extern/nemotron")
    ap.add_argument(
        "--families",
        default=",".join(sorted(DEFAULT_TARGET_FAMILIES)),
        help="Comma-separated local families to import for official_train rows.",
    )
    args = ap.parse_args()

    extern_root = Path(args.extern_root)
    reasoning_dir = extern_root / "reasoning"
    problems = load_problems(extern_root)
    target_families = {x.strip() for x in args.families.split(",") if x.strip()}

    with open(args.in_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "cot" not in fieldnames:
        fieldnames.append("cot")
    if "has_cot" not in fieldnames:
        fieldnames.append("has_cot")

    stats = Counter()
    by_family = defaultdict(Counter)
    by_extern_category = defaultdict(Counter)
    examples = defaultdict(list)

    for row in rows:
        pid = row_id(row)
        family = row.get("family") or ""
        source = row.get("source") or ""
        had_cot = bool((row.get("cot") or "").strip())

        if source != "official_train" or family not in target_families:
            if had_cot:
                stats["kept_existing_cot"] += 1
                by_family[family]["kept_existing_cot"] += 1
            continue

        problem = problems.get(pid)
        reasoning_path = reasoning_dir / f"{pid}.txt"
        if problem is None or not reasoning_path.exists():
            stats["missing_external_reasoning"] += 1
            by_family[family]["missing_external_reasoning"] += 1
            continue

        text = reasoning_path.read_text()
        pred = extract_final_answer(text)
        extern_category = problem.get("category", "unknown")
        extern_status = problem.get("status", "unknown")
        by_extern_category[extern_category]["total_seen"] += 1

        if not verify(str(row.get("answer", "")), pred):
            stats["external_wrong"] += 1
            by_family[family]["external_wrong"] += 1
            by_extern_category[extern_category]["wrong"] += 1
            if len(examples[(family, "external_wrong")]) < 5:
                examples[(family, "external_wrong")].append(
                    {
                        "id": pid,
                        "extern_category": extern_category,
                        "extern_status": extern_status,
                        "gold": row.get("answer", ""),
                        "pred": pred,
                    }
                )
            continue

        cot = compact_reasoning(text, family, extern_category)
        if not cot:
            stats["empty_after_clean"] += 1
            by_family[family]["empty_after_clean"] += 1
            continue

        row["cot"] = cot
        row["has_cot"] = "True"
        stats["external_imported"] += 1
        by_family[family]["external_imported"] += 1
        by_extern_category[extern_category]["imported"] += 1
        if had_cot:
            stats["external_replaced_existing"] += 1
            by_family[family]["external_replaced_existing"] += 1
        else:
            stats["external_filled_empty"] += 1
            by_family[family]["external_filled_empty"] += 1

    total_has_cot = 0
    for row in rows:
        has = bool((row.get("cot") or "").strip())
        row["has_cot"] = "True" if has else "False"
        total_has_cot += int(has)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "input": args.in_csv,
        "output": args.out_csv,
        "extern_root": str(extern_root),
        "target_families": sorted(target_families),
        "total_rows": len(rows),
        "total_has_cot": total_has_cot,
        "total_cot_rate": total_has_cot / max(len(rows), 1),
        "stats": dict(stats),
        "by_family": {k: dict(v) for k, v in sorted(by_family.items())},
        "by_extern_category": {k: dict(v) for k, v in sorted(by_extern_category.items())},
        "examples": {f"{k[0]}|{k[1]}": v for k, v in examples.items()},
    }
    with open(args.report, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.report}")
    print(f"Total CoT: {total_has_cot}/{len(rows)} ({total_has_cot / max(len(rows), 1):.3f})")
    print("Stats:", dict(stats))
    for family, counter in sorted(by_family.items()):
        print(f"{family}: {dict(counter)}")


if __name__ == "__main__":
    main()
