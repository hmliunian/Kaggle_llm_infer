#!/usr/bin/env python3
"""Extract StructuredOutput results from a workflow transcript dir, verify vs gold."""
from __future__ import annotations
import json, glob, sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from official_metric import verify


def load(jsonl):
    gold, cat = {}, {}
    for l in open(jsonl):
        if not l.strip():
            continue
        e = json.loads(l)
        gold[e["id"]] = e["answer"]
        cat[e["id"]] = e["category"]
    return gold, cat


def extract(wf_dir):
    res = {}
    for f in glob.glob(wf_dir + "/agent-*.jsonl"):
        pid, out = None, None
        for l in open(f):
            try:
                e = json.loads(l)
            except Exception:
                continue
            for c in (e.get("message", {}).get("content") or []):
                if not isinstance(c, dict) or c.get("type") != "tool_use":
                    continue
                if c.get("name") == "StructuredOutput":
                    out = c.get("input")
                elif c.get("name") == "Read":
                    p = c.get("input", {}).get("file_path", "")
                    if "ru_problems/" in p:
                        pid = p.split("ru_problems/")[1].replace(".txt", "")
        if out is not None and pid is not None:
            res[pid] = out
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", required=True)
    ap.add_argument("--jsonl", default=str(ROOT / "data" / "rule_unknown_problems.jsonl"))
    args = ap.parse_args()
    gold, cat = load(args.jsonl)
    res = extract(args.wf)
    print("extracted", len(res), "results\n")
    from collections import Counter
    tot, okc = Counter(), Counter()
    rows = sorted(res.items(), key=lambda kv: cat.get(kv[0], ""))
    for pid, out in rows:
        pa = str(out.get("predicted_answer", ""))
        cs = out.get("can_solve")
        g = gold.get(pid, "")
        corr = verify(g, pa)
        c = cat.get(pid, "?")
        tot[c] += 1
        okc[c] += int(corr)
        mark = "OK " if corr else " x "
        print("%-24s %s can_solve=%-5s %s pred=%-18r gold=%-12r" % (c, pid, str(cs), mark, pa, g))
    print("\nby category (correct/total):")
    for c in sorted(tot):
        print("  %-24s %d/%d" % (c, okc[c], tot[c]))
    print("TOTAL %d/%d" % (sum(okc.values()), sum(tot.values())))


if __name__ == "__main__":
    main()
