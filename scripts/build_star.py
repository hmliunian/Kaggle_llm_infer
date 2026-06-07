#!/usr/bin/env python3
"""STaR (Self-Taught Reasoner) data builder — fill verified CoT for the rows the
deterministic solver could not cover, by rejection-sampling model rationales
through a faithful rule gate.

Pipeline position
-----------------
  stage 1  scripts/vllm_build_prompts.py   (gap rows -> tokenized prompts)
  stage 2  scripts/vllm_eval_run.py        (TEMPERATURE>0, sample K per row)
  stage 3  THIS SCRIPT --from-generations  (parse, gate, compact, write CoT rows)
  stage 4  merge kept rows into the training CSV (--merge)

Gates (both required to keep a sample)
  1. answer gate : official_metric.verify(gold, extract_final_answer(text))
  2. rule  gate  : the explicit rule the rationale commits to, re-executed
                   mechanically, reproduces EVERY shown example AND the query.
The rule gate is what stops us training on lucky-guess / leaked answers on the
under-determined cryptarithm rows.

Offline-testable: the gating logic (keep_sample / parse_prompt) is GPU-free.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from official_metric import extract_final_answer, verify  # noqa: E402
from scripts.build_v5_from_nemotron import clean_lines, trim_blank_edges  # noqa: E402
from scripts.star_rules import check_rule  # noqa: E402

csv.field_size_limit(10**7)

SCHEMA = [" id", "prompt", "answer", "family", "rule_name", "source", "cot", "has_cot"]

GAP_FAMILIES = {"bit_manipulation", "equation_symbol_transformation"}


# ---------------------------------------------------------------------------
# Prompt parsing: recover (examples, query) from the competition prompt text
# ---------------------------------------------------------------------------
_BIT_EX_RE = re.compile(r"^\s*([01]{8})\s*->\s*([01]{8})\s*$")
_BIT_Q_RE = re.compile(r"determine the output for:\s*([01]{8})")

_EQ_EX_RE = re.compile(r"^\s*(\S+?)\s*=\s*(\S+)\s*$")
_EQ_Q_RE = re.compile(r"determine the result for:\s*(\S+)")


def parse_prompt(family: str, prompt: str):
    """Return (examples:list[(inp,out)], query:str|None) for a competition prompt."""
    examples: list[tuple[str, str]] = []
    query = None
    if family == "bit_manipulation":
        for ln in prompt.splitlines():
            m = _BIT_EX_RE.match(ln)
            if m:
                examples.append((m.group(1), m.group(2)))
            mq = _BIT_Q_RE.search(ln)
            if mq:
                query = mq.group(1)
    elif family == "equation_symbol_transformation":
        for ln in prompt.splitlines():
            mq = _EQ_Q_RE.search(ln)
            if mq:
                query = mq.group(1)
                continue
            if "=" in ln and "determine" not in ln and "examples" not in ln.lower():
                m = _EQ_EX_RE.match(ln)
                if m:
                    examples.append((m.group(1), m.group(2)))
    return examples, query


# ---------------------------------------------------------------------------
# CoT extraction + compaction from a raw model generation
# ---------------------------------------------------------------------------
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def extract_think(text: str) -> str:
    """Return the reasoning content (inside <think>...</think> if present)."""
    if _THINK_CLOSE in text:
        head = text.split(_THINK_CLOSE, 1)[0]
        if _THINK_OPEN in head:
            head = head.split(_THINK_OPEN, 1)[1]
        return head
    # the stage-1 prompt already ends with an open `<think>\n`, so a generation
    # without a close tag is all reasoning up to the final-answer prefill.
    if "The final answer is" in text:
        return text.split("The final answer is", 1)[0]
    return text


def compact_cot(think_text: str) -> str:
    """Strip boxed lines / im_end and trim — same canonicalization as the
    deterministic importer, so STaR CoT is byte-compatible with imported CoT."""
    lines = clean_lines(think_text)
    lines = trim_blank_edges(lines)
    cot = "\n".join(lines).strip()
    cot = cot.replace("\\boxed", "boxed")
    return cot


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def keep_sample(family: str, prompt: str, gold: str, raw_generation: str,
                require_rule_gate: bool = True):
    """Return (cot, reason) where cot is the kept CoT or None.

    reason is a short tag for stats: 'kept', 'answer_fail', 'no_examples',
    'rule_fail', 'empty_cot'.
    """
    pred = extract_final_answer(raw_generation)
    if not verify(str(gold), pred):
        return None, "answer_fail"

    examples, query = parse_prompt(family, prompt)
    if not examples or query is None:
        return None, "no_examples"

    think = extract_think(raw_generation)

    if require_rule_gate:
        rule_repr = check_rule(family, think, examples, query, gold)
        if rule_repr is None:
            return None, "rule_fail"
        # the rule reproduced all examples + a query output; confirm that
        # output also verifies against gold (rule could output a valid-but-
        # different string the metric still accepts, e.g. leading-zero).
        # check_rule already required exact example reproduction; gold is
        # validated via verify on the model's own boxed answer above.

    cot = compact_cot(think)
    if not cot:
        return None, "empty_cot"
    return cot, "kept"


# ---------------------------------------------------------------------------
# Ingest mode: read generations + prompts metadata, emit kept CoT rows
# ---------------------------------------------------------------------------
def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def ingest(prompts_path: str, gens_path: str, source_tag: str,
           require_rule_gate: bool = True):
    """Combine stage-1 prompts metadata + stage-2 generations into kept rows.

    Supports K samples per prompt: a generation row may carry "samples" (list of
    texts) or a single "decoded_raw"; multiple gen rows with the same
    sample_index are also accepted.
    """
    prompts = {r["sample_index"]: r for r in load_jsonl(prompts_path)}
    gens_by_idx: dict[int, list[str]] = defaultdict(list)
    for g in load_jsonl(gens_path):
        idx = g["sample_index"]
        if "samples" in g and isinstance(g["samples"], list):
            gens_by_idx[idx].extend(g["samples"])
        else:
            gens_by_idx[idx].append(g.get("decoded_raw", ""))

    stats = Counter()
    fam_kept = Counter()
    rule_kept: Counter = Counter()
    rows: list[dict] = []
    kept_idx: set[int] = set()

    for idx, meta in prompts.items():
        family = meta.get("family", "")
        if family not in GAP_FAMILIES:
            stats["skip_non_gap_family"] += 1
            continue
        gold = str(meta.get("gold", "")).strip()
        prompt = meta.get("prompt", "")
        cands = gens_by_idx.get(idx, [])
        if not cands:
            stats["no_generation"] += 1
            continue
        kept_here = False
        for text in cands:
            cot, reason = keep_sample(family, prompt, gold, text, require_rule_gate)
            stats[reason] += 1
            if cot is None:
                continue
            rows.append({
                " id": f"star_{source_tag}_{idx:06d}",
                "prompt": prompt,
                "answer": gold,
                "family": family,
                "rule_name": f"star:{family}",
                "source": source_tag,
                "cot": cot,
                "has_cot": "True",
            })
            fam_kept[family] += 1
            kept_idx.add(idx)
            kept_here = True
            break  # one kept CoT per row is enough
        if not kept_here:
            stats["row_unfilled"] += 1

    report = {
        "prompts": len(prompts),
        "kept_rows": len(rows),
        "kept_by_family": dict(fam_kept),
        "gate_stats": dict(stats),
        "require_rule_gate": require_rule_gate,
    }
    return rows, report


# ---------------------------------------------------------------------------
# Merge kept rows into a training CSV
# ---------------------------------------------------------------------------
def merge(in_csv: str, kept_rows: list[dict], out_csv: str):
    """Fill CoT for matching gap rows in `in_csv` (by prompt), append novelty.

    A kept row updates the existing training row that has the same prompt and
    no CoT yet; if no such row exists it is appended as a fresh synthetic row.
    """
    with open(in_csv, newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for c in SCHEMA:
        if c not in fields:
            fields.append(c)

    by_prompt: dict[str, dict] = {}
    for r in rows:
        if (r.get("family") in GAP_FAMILIES) and not (r.get("cot") or "").strip():
            by_prompt.setdefault(r["prompt"], r)

    filled = 0
    appended = 0
    for kr in kept_rows:
        tgt = by_prompt.get(kr["prompt"])
        if tgt is not None and not (tgt.get("cot") or "").strip():
            tgt["cot"] = kr["cot"]
            tgt["has_cot"] = "True"
            filled += 1
            del by_prompt[kr["prompt"]]
        else:
            rows.append(kr)
            appended += 1

    for r in rows:
        r["has_cot"] = "True" if (r.get("cot") or "").strip() else "False"

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return {"filled_existing": filled, "appended_new": appended, "total_rows": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", help="stage-1 prompts jsonl (sample_index, gold, prompt, family)")
    ap.add_argument("--generations", help="stage-2 generations jsonl (sample_index, decoded_raw|samples)")
    ap.add_argument("--source-tag", default="star_r1")
    ap.add_argument("--kept-out", default="data/star_kept.csv")
    ap.add_argument("--no-rule-gate", action="store_true",
                    help="DANGER: keep on answer match only (admits lucky guesses).")
    ap.add_argument("--merge-into", help="training CSV to merge kept rows into")
    ap.add_argument("--merge-out", default="data/train_plus_synthetic_star.csv")
    ap.add_argument("--report", default="data/star_report.json")
    args = ap.parse_args()

    if not args.prompts or not args.generations:
        ap.error("--prompts and --generations are required for ingest")

    rows, report = ingest(
        args.prompts, args.generations, args.source_tag,
        require_rule_gate=not args.no_rule_gate,
    )

    with open(args.kept_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} kept CoT rows -> {args.kept_out}")

    if args.merge_into:
        mreport = merge(args.merge_into, rows, args.merge_out)
        report["merge"] = mreport
        print(f"Merged -> {args.merge_out}: {mreport}")

    with open(args.report, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
