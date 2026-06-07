#!/usr/bin/env python3
"""Build a v7 CoT dataset by importing LLM investigation notes for hypothesis_formed rows.

The v5 import (``build_v5_from_nemotron.py``) only pulled CoT from the deterministic
solver traces in ``extern/nemotron/reasoning/*.txt``, keeping a row only when the
trace's ``\\boxed{}`` answer verified against gold. For official rows with
``status == "hypothesis_formed"`` the solver formed a partial/wrong hypothesis
(e.g. ``default 1`` fills), so its boxed answer != gold and the row was left
answer-only.

A minority of the files in ``extern/nemotron/investigations/*.txt`` are *free-form
LLM analyses* (inferred rule -> why it fits -> step-by-step application -> predicted
answer) that solve exactly these cases with a genuine chain-of-thought. The rest are
*terse algorithmic-searcher* output (a one-line ``rule:`` formula + examples + query +
predicted answer) with no reasoning -- importing those as "CoT" would yield garbage
like ``query: 10101101``. This script imports ONLY the genuine-prose investigations,
and only when their predicted answer verifies against the local gold under the official
metric. The predicted-answer/confidence tail is stripped and any ``\\boxed`` neutralized
so train_sft.py keeps its single canonical final-answer target.

rule_unknown rows are intentionally not touched: their investigations reach no usable
answer (measured 0 correct), so there is nothing to import.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from official_metric import verify  # noqa: E402


TARGET_FAMILIES = {
    "bit_manipulation",
    "equation_symbol_transformation",
}
TARGET_STATUS = "hypothesis_formed"

# Section markers vary across investigation files (lowercase for bit_manipulation,
# Capitalized for equation_*; the reasoning section itself has many different headings
# -- "inferred rule", "symbol-to-digit mapping", "operator-to-operation mapping", ...).
# What is stable across all files is an "examples:" block up front (duplicates the
# prompt) and a "predicted answer:" / "confidence" tail. So the CoT is everything
# *between* those two, regardless of the internal headings. All matching is
# case-insensitive.
PREDICTED_ANSWER_RE = re.compile(r"(?i)^\s*predicted answer:")
EXAMPLES_RE = re.compile(r"(?i)^\s*examples?:")
TAIL_RE = re.compile(r"(?i)^\s*(predicted answer|confidence note|confidence):")
QUERY_LINE_RE = re.compile(r"(?i)^\s*query:")

# Genuine LLM-prose investigations carry one of these reasoning-section markers; terse
# searcher files (rule formula + examples + query + answer) carry none. Validated to
# agree exactly with a >=120-char body-length cutoff across all hypothesis_formed files
# (100 genuine: 13 bit + 87 equation; 112 terse dropped).
REASONING_MARKER_RE = re.compile(
    r"(?i)(step-by-step|inferred rule|rule i infer|why th|fits the|mapping:|"
    r"consistent |digit (map|assignment|reading)|bitwise majority|interpret|"
    r"checks?\b|under that map|with that map|assignment that fits|support this)"
)


def row_id(row: dict[str, str]) -> str:
    return (row.get(" id") or row.get("id") or "").strip()


def load_statuses(extern_root: Path) -> dict[str, str]:
    """Map official problem id -> investigation status from problems.jsonl."""
    statuses: dict[str, str] = {}
    with (extern_root / "problems.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                statuses[entry["id"]] = entry.get("status", "unknown")
    return statuses


def parse_predicted_answer(text: str) -> str | None:
    """Return the investigation's predicted answer, or None if absent.

    Handles both ``Predicted answer:\\n- 01000011`` (value on the next line, with an
    optional ``-``/``*`` bullet) and ``Predicted answer: 71`` (value inline).
    """
    lines = text.replace("\r\n", "\n").split("\n")
    for idx, line in enumerate(lines):
        if not PREDICTED_ANSWER_RE.match(line):
            continue
        inline = PREDICTED_ANSWER_RE.sub("", line).strip()
        if inline:
            return inline.strip("`").strip()
        for nxt in lines[idx + 1:]:
            if nxt.strip():
                return re.sub(r"^\s*[-*]\s*", "", nxt).strip().strip("`").strip()
        return None
    return None


def clean_investigation(text: str) -> str:
    """Reduce an investigation file to a trainable CoT.

    Keeps the reasoning body -- everything from the end of the ``examples:`` block
    (which duplicates the prompt) up to the ``predicted answer:`` / ``confidence``
    tail -- so the model's inferred rule, mapping, and step-by-step application are
    retained whatever internal headings it used. The predicted-answer line and the
    (hedging) confidence note are dropped, and any ``\\boxed`` is neutralized so no
    competing boxed answer lands inside the ``<think>`` block.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    n = len(lines)

    # Start of reasoning = first real section line after the examples block. The
    # examples block is the "examples:" header followed by bullet ("- ...") or
    # indented/blank lines (including the "- query: ..." line).
    ex = next((i for i, l in enumerate(lines) if EXAMPLES_RE.match(l)), None)
    if ex is None:
        start = 0  # no examples block; keep from the top (rare)
    else:
        start = ex + 1
        while start < n:
            stripped = lines[start].strip()
            is_bullet = stripped.startswith(("-", "*"))
            is_indented = bool(lines[start][:1].isspace())
            if stripped == "" or is_bullet or is_indented:
                start += 1
                continue
            break

    end = next((i for i in range(start, n) if TAIL_RE.match(lines[i])), n)

    # Drop any stray "query: ..." restatement lines -- in terse searcher files that is
    # the entire body; in genuine prose files it never appears here.
    body = [l.rstrip() for l in lines[start:end] if not QUERY_LINE_RE.match(l)]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()

    cot = "\n".join(body).strip()
    cot = cot.replace("\\boxed", "boxed").replace("<|im_end|>", "")
    return cot.strip()


def is_genuine_cot(cot: str) -> bool:
    """True only for real LLM reasoning, not terse searcher rule/answer output.

    Requires a recognized reasoning-section marker and at least two non-empty lines,
    which cleanly separates the prose investigations from the one-line terse files.
    """
    if not cot:
        return False
    nonempty = [l for l in cot.split("\n") if l.strip()]
    return len(nonempty) >= 2 and bool(REASONING_MARKER_RE.search(cot))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default="data/train_plus_synthetic_v6.csv")
    ap.add_argument("--out", dest="out_csv", default="data/train_plus_synthetic_v7.csv")
    ap.add_argument("--report", default="data/train_plus_synthetic_v7_report.json")
    ap.add_argument("--extern-root", default="extern/nemotron")
    args = ap.parse_args()

    extern_root = Path(args.extern_root)
    investigations_dir = extern_root / "investigations"
    statuses = load_statuses(extern_root)

    with open(args.in_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "cot" not in fieldnames:
        fieldnames.append("cot")
    if "has_cot" not in fieldnames:
        fieldnames.append("has_cot")

    stats = Counter()
    by_family_before = defaultdict(Counter)

    for row in rows:
        family = row.get("family") or ""
        had_cot = bool((row.get("cot") or "").strip())
        by_family_before[family]["total"] += 1
        by_family_before[family]["has_cot"] += int(had_cot)

        pid = row_id(row)
        source = row.get("source") or ""
        if (
            source != "official_train"
            or had_cot
            or family not in TARGET_FAMILIES
            or statuses.get(pid) != TARGET_STATUS
        ):
            continue

        stats["hypothesis_formed_seen"] += 1
        inv_path = investigations_dir / f"{pid}.txt"
        if not inv_path.exists():
            stats["missing_investigation"] += 1
            continue

        text = inv_path.read_text()
        pred = parse_predicted_answer(text)
        if pred is None:
            stats["no_predicted_answer"] += 1
            continue

        if not verify(str(row.get("answer", "")), pred):
            stats["skipped_wrong_answer"] += 1
            continue

        cot = clean_investigation(text)
        if not is_genuine_cot(cot):
            # Terse searcher file (rule formula + answer, no reasoning) -- not a CoT.
            stats["skipped_terse_searcher"] += 1
            continue

        row["cot"] = cot
        row["has_cot"] = "True"
        stats["investigation_imported"] += 1
        stats[f"imported_{family}"] += 1

    total_has_cot = 0
    by_family_after = defaultdict(Counter)
    for row in rows:
        has = bool((row.get("cot") or "").strip())
        row["has_cot"] = "True" if has else "False"
        total_has_cot += int(has)
        fam = row.get("family") or ""
        by_family_after[fam]["total"] += 1
        by_family_after[fam]["has_cot"] += int(has)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "input": args.in_csv,
        "output": args.out_csv,
        "extern_root": str(extern_root),
        "target_families": sorted(TARGET_FAMILIES),
        "target_status": TARGET_STATUS,
        "total_rows": len(rows),
        "total_has_cot": total_has_cot,
        "total_cot_rate": total_has_cot / max(len(rows), 1),
        "stats": dict(stats),
        "by_family": {
            fam: {
                "total": by_family_after[fam]["total"],
                "has_cot_before": by_family_before[fam]["has_cot"],
                "has_cot_after": by_family_after[fam]["has_cot"],
            }
            for fam in sorted(by_family_after)
        },
    }
    with open(args.report, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.report}")
    print(f"Total CoT: {total_has_cot}/{len(rows)} ({total_has_cot / max(len(rows), 1):.3f})")
    print("Stats:", dict(stats))
    for fam in sorted(by_family_after):
        b = by_family_before[fam]["has_cot"]
        a = by_family_after[fam]["has_cot"]
        if a != b:
            print(f"  {fam}: has_cot {b} -> {a} (+{a - b})")


if __name__ == "__main__":
    main()
