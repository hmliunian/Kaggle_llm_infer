#!/usr/bin/env python3
"""Build the v6 dataset by generating reasoner-aligned synthetic CoT.

Strategy: sample a ground-truth rule -> feed examples to the reference reasoner
in extern/nemotron/reasoners -> keep only rows whose boxed answer matches the
constructed gold under official_metric.verify -> compact the trace with the same
``compact_reasoning`` used for the v5 import, so synthetic CoT is byte-compatible
with imported official CoT.

Families generated here (replacing v3 synthetic for these two families):
  - bit_manipulation: full per-output-bit boolean rule space (I/NOT/C/AND/OR/XOR
    and *-NOT), weighted so ~78% of rows use a genuine two-input-bit op (matches
    the official distribution).
  - equation_symbol_transformation:
      * numeric: full op space of equation_numeric.py (concat/rev, add/sub/mul,
        div/mod, +-1, digit-level, determinant) x reversed operands/result.
      * symbolic-concat: forward/reverse concatenation via reasoning_cryptarithm.
      * symbolic-arith (constructed-solvable): symbol->digit substitution plus an
        arithmetic operator, generated so the mapping is uniquely determined, then
        solved + explained by a local backtracking solver.

Official symbolic rows that the solver can verify (pred == gold) also receive CoT.
Everything else keeps its v5 content.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from official_metric import verify  # noqa: E402
from scripts.build_v5_from_nemotron import compact_reasoning  # noqa: E402
from extern.nemotron.reasoners.store_types import Example, Problem  # noqa: E402
from extern.nemotron.reasoners.bit_manipulation import (  # noqa: E402
    reasoning_bit_manipulation,
)
from extern.nemotron.reasoners.equation_numeric import (  # noqa: E402
    reasoning_equation_numeric,
)
from extern.nemotron.reasoners.cryptarithm import reasoning_cryptarithm  # noqa: E402

import re  # noqa: E402

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def extract_boxed(trace: str) -> str | None:
    ms = BOXED_RE.findall(trace)
    return ms[-1] if ms else None


# ---------------------------------------------------------------------------
# Prompt templates (exact official wording)
# ---------------------------------------------------------------------------
BIT_HEADER = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit "
    "binary numbers. The transformation involves operations like bit shifts, "
    "rotations, XOR, AND, OR, NOT, and possibly majority or choice functions."
)
EQ_HEADER = (
    "In Alice's Wonderland, a secret set of transformation rules is applied to "
    "equations. Below are a few examples:"
)


def bit_prompt(examples: list[tuple[str, str]], query: str) -> str:
    lines = [BIT_HEADER, "", "Here are some examples of input -> output:"]
    lines += [f"{i} -> {o}" for i, o in examples]
    lines += ["", f"Now, determine the output for: {query}"]
    return "\n".join(lines)


def eq_prompt(examples: list[tuple[str, str]], query: str) -> str:
    lines = [EQ_HEADER]
    lines += [f"{i} = {o}" for i, o in examples]
    lines += [f"Now, determine the result for: {query}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# bit_manipulation generator
# ---------------------------------------------------------------------------
PAIR_SYM = ["XOR", "OR", "AND"]
PAIR_ASYM = ["XOR-NOT", "OR-NOT", "AND-NOT"]


def _bit_pair(op: str, a: str, b: str) -> str:
    bb = ("1" if b == "0" else "0") if op.endswith("-NOT") else b
    base = op.split("-")[0]
    if base == "AND":
        return "1" if a == "1" and bb == "1" else "0"
    if base == "OR":
        return "1" if a == "1" or bb == "1" else "0"
    return "1" if a != bb else "0"  # XOR


def sample_bit_rule(rng: random.Random):
    """Return a per-bit function f(bits8)->bits8 with stride structure.

    Distribution roughly matches official: ~78% use a genuine two-input op.
    """
    r = rng.random()
    if r < 0.78:
        op = rng.choice(PAIR_SYM + PAIR_ASYM)
        o = rng.randrange(8)
        diff = rng.randrange(1, 8)

        def f(bits: str) -> str:
            out = []
            for i in range(8):
                a = bits[(o + i) % 8]
                b = bits[(o + i + diff) % 8]
                out.append(_bit_pair(op, a, b))
            return "".join(out)

        return f, f"bit_{op}_o{o}_d{diff}"
    elif r < 0.95:
        # unary mask: rotation by k then per-bit identity/NOT mask (XOR-mask family)
        k = rng.randrange(8)
        mask = [rng.randrange(2) for _ in range(8)]

        def f(bits: str) -> str:
            out = []
            for i in range(8):
                v = bits[(i - k) % 8]
                if mask[i]:
                    v = "1" if v == "0" else "0"
                out.append(v)
            return "".join(out)

        return f, f"bit_mask_k{k}"
    else:
        # constant bits mixed with identity (rare)
        const = {i: str(rng.randrange(2)) for i in range(8) if rng.random() < 0.4}

        def f(bits: str) -> str:
            return "".join(const.get(i, bits[i]) for i in range(8))

        return f, "bit_const_id"


def gen_bit(n: int, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    rule_fam = Counter()
    attempts = 0
    while len(rows) < n and attempts < n * 40:
        attempts += 1
        f, name = sample_bit_rule(rng)
        n_ex = rng.choice([7, 8, 9, 10])
        seen = set()
        examples = []
        for _ in range(n_ex * 3):
            x = "".join(rng.choice("01") for _ in range(8))
            if x in seen:
                continue
            seen.add(x)
            examples.append((x, f(x)))
            if len(examples) == n_ex:
                break
        if len(examples) < n_ex:
            continue
        q = "".join(rng.choice("01") for _ in range(8))
        while q in seen:
            q = "".join(rng.choice("01") for _ in range(8))
        gold = f(q)
        prob = Problem(
            id="syn",
            category="bit_manipulation",
            examples=[Example(i, o) for i, o in examples],
            question=q,
            answer=gold,
        )
        try:
            trace = reasoning_bit_manipulation(prob)
        except Exception:
            continue
        if not trace:
            continue
        if extract_boxed(trace) != gold:
            continue  # reasoner picked a different (ambiguous) rule -> drop
        cot = compact_reasoning(trace, "bit_manipulation", "bit_manipulation")
        if not cot:
            continue
        rows.append(
            {
                " id": f"synthetic_v6_bit_{len(rows):06d}",
                "prompt": bit_prompt(examples, q),
                "answer": gold,
                "family": "bit_manipulation",
                "rule_name": name,
                "source": "synthetic_v6",
                "cot": cot,
                "has_cot": "True",
            }
        )
        rule_fam[name.split("_")[1]] += 1
    return rows


# ---------------------------------------------------------------------------
# numeric equation generator
# ---------------------------------------------------------------------------
NUM_OPS_COMMON = [
    "concatenation",
    "reverse concatenation",
    "addition",
    "absolute difference",
    "subtraction (a-b)",
    "reverse subtraction (b-a)",
    "multiplication",
]
NUM_OPS_RARE = [
    "multiply+1",
    "multiply-1",
    "add+1",
    "integer division (a/b)",
    "modulo (a mod b)",
    "digit add mod10",
    "digit sub mod10",
    "digit absolute diff",
    "digit multiply",
    "cross multiply",
    "determinant",
    "abs determinant",
]
EQ_OP_SYMS = list("-*+/|}{@?!:^<>")


def _num_eval(op: str, a: int, b: int, sa: str, sb: str):
    d1, d2 = int(sa[0]), int(sa[1])
    d3, d4 = int(sb[0]), int(sb[1])
    table = {
        "concatenation": sa + sb,
        "reverse concatenation": sb + sa,
        "addition": str(a + b),
        "absolute difference": str(abs(a - b)),
        "subtraction (a-b)": str(a - b),
        "reverse subtraction (b-a)": str(b - a),
        "multiplication": str(a * b),
        "multiply+1": str(a * b + 1),
        "multiply-1": str(a * b - 1),
        "add+1": str(a + b + 1),
        "integer division (a/b)": str(a // b) if b else None,
        "modulo (a mod b)": str(a % b) if b else None,
        "digit add mod10": str((d1 + d3) % 10) + str((d2 + d4) % 10),
        "digit sub mod10": str((d1 - d3) % 10) + str((d2 - d4) % 10),
        "digit absolute diff": str(abs(d1 - d3)) + str(abs(d2 - d4)),
        "digit multiply": str(d1 * d3) + str(d2 * d4),
        "cross multiply": str(d1 * d3 + d2 * d4),
        "determinant": str(d1 * d4 - d2 * d3),
        "abs determinant": str(abs(d1 * d4 - d2 * d3)),
    }
    return table.get(op)


def _rev_str(s: str) -> str:
    return ("-" + s[1:][::-1]) if s.startswith("-") else s[::-1]


def gen_numeric(n: int, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    attempts = 0
    op_pool = NUM_OPS_COMMON * 2 + NUM_OPS_RARE
    while len(rows) < n and attempts < n * 60:
        attempts += 1
        op = rng.choice(op_pool)
        rev_ops = rng.random() < 0.35
        rev_res = rng.random() < 0.35
        sym = rng.choice(EQ_OP_SYMS)

        def transform(a_str: str, b_str: str):
            ta = a_str[::-1] if rev_ops else a_str
            tb = b_str[::-1] if rev_ops else b_str
            raw = _num_eval(op, int(ta), int(tb), ta, tb)
            if raw is None:
                return None
            return _rev_str(raw) if rev_res else raw

        n_ex = rng.choice([3, 4, 5])
        examples = []
        ok = True
        seen = set()
        for _ in range(n_ex * 4):
            a = rng.randrange(10, 100)
            b = rng.randrange(10, 100)
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            out = transform(str(a), str(b))
            if out is None or out == "" or (out.startswith("-") and len(out) == 1):
                ok = False
                break
            examples.append((f"{a}{sym}{b}", out))
            if len(examples) == n_ex:
                break
        if not ok or len(examples) < n_ex:
            continue
        qa, qb = rng.randrange(10, 100), rng.randrange(10, 100)
        while (qa, qb) in seen:
            qa, qb = rng.randrange(10, 100), rng.randrange(10, 100)
        gold = transform(str(qa), str(qb))
        if gold is None or gold == "":
            continue
        query = f"{qa}{sym}{qb}"
        prob = Problem(
            id="syn",
            category="equation_numeric_deduce",
            examples=[Example(i, o) for i, o in examples],
            question=query,
            answer=gold,
        )
        try:
            trace = reasoning_equation_numeric(prob)
        except Exception:
            continue
        if not trace or not verify(gold, extract_boxed(trace) or ""):
            continue
        cot = compact_reasoning(
            trace, "equation_symbol_transformation", "equation_numeric_deduce"
        )
        if not cot:
            continue
        rows.append(
            {
                " id": f"synthetic_v6_numeq_{len(rows):06d}",
                "prompt": eq_prompt(examples, query),
                "answer": gold,
                "family": "equation_symbol_transformation",
                "rule_name": f"numeric:{op}{'|ro' if rev_ops else ''}{'|rr' if rev_res else ''}",
                "source": "synthetic_v6",
                "cot": cot,
                "has_cot": "True",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# symbolic equation: concatenation (via reasoning_cryptarithm)
# ---------------------------------------------------------------------------
SYM_ALPHA = list("!\"#$%&'()/:<>?@[\\]^`{|}")  # 23-symbol operand alphabet (official)


def _strip_brace_tail(cot: str) -> str:
    # cryptarithm.py emits 'output: 【X】-> 【{X}】'; drop the boxed-preview tail.
    return re.sub(r"\s*->\s*【\{[^】]*\}】", "", cot)


def gen_symbolic_concat(n: int, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    attempts = 0
    while len(rows) < n and attempts < n * 30:
        attempts += 1
        ct = rng.choice(["fwd", "rev"])
        op = rng.choice(SYM_ALPHA)
        n_ex = rng.choice([3, 4, 5])
        examples = []
        seen = set()
        for _ in range(n_ex * 3):
            a = (rng.choice(SYM_ALPHA), rng.choice(SYM_ALPHA))
            b = (rng.choice(SYM_ALPHA), rng.choice(SYM_ALPHA))
            inp = a[0] + a[1] + op + b[0] + b[1]
            if inp in seen:
                continue
            seen.add(inp)
            out = (a[0] + a[1] + b[0] + b[1]) if ct == "fwd" else (b[0] + b[1] + a[0] + a[1])
            examples.append((inp, out))
            if len(examples) == n_ex:
                break
        if len(examples) < n_ex:
            continue
        qa = (rng.choice(SYM_ALPHA), rng.choice(SYM_ALPHA))
        qb = (rng.choice(SYM_ALPHA), rng.choice(SYM_ALPHA))
        q = qa[0] + qa[1] + op + qb[0] + qb[1]
        gold = (qa[0] + qa[1] + qb[0] + qb[1]) if ct == "fwd" else (qb[0] + qb[1] + qa[0] + qa[1])
        prob = Problem(
            id="syn",
            category="cryptarithm_deduce",
            examples=[Example(i, o) for i, o in examples],
            question=q,
            answer=gold,
        )
        try:
            trace = reasoning_cryptarithm(prob)
        except Exception:
            continue
        if not trace or not verify(gold, extract_boxed(trace) or ""):
            continue
        cot = compact_reasoning(
            trace, "equation_symbol_transformation", "cryptarithm_deduce"
        )
        cot = _strip_brace_tail(cot)
        if not cot:
            continue
        rows.append(
            {
                " id": f"synthetic_v6_symcat_{len(rows):06d}",
                "prompt": eq_prompt(examples, q),
                "answer": gold,
                "family": "equation_symbol_transformation",
                "rule_name": f"symbolic:concat_{ct}",
                "source": "synthetic_v6",
                "cot": cot,
                "has_cot": "True",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# symbolic equation: arithmetic cryptarithm (constructed-solvable) + solver/CoT
# ---------------------------------------------------------------------------
ARITH_OPS = {
    "add": (lambda a, b: a + b, "{a} + {b} = {r}"),
    "abs_diff": (lambda a, b: abs(a - b), "|{a} - {b}| = {r}"),
    "mul": (lambda a, b: a * b, "{a} * {b} = {r}"),
}


def _num_digits(n: int) -> tuple[int, ...]:
    if n == 0:
        return (0,)
    out = []
    while n > 0:
        out.append(n % 10)
        n //= 10
    return tuple(reversed(out))


def solve_cryptarithm(examples, query):
    """Return the set of distinct query answers over all consistent unique mappings.

    examples: list of (s0,s1,op,s3,s4,out_tuple). query: (s0,s1,op,s3,s4).
    Early-exit once two distinct answers appear (problem is ambiguous).
    """
    answers: set[str] = set()
    info = {}
    mapping: dict[str, int] = {}
    used: set[int] = set()
    opassign: dict[str, str] = {}
    budget = [400000]

    def assign(s, d):
        if s in mapping:
            return False if mapping[s] == d else None
        if d in used:
            return None
        mapping[s] = d
        used.add(d)
        return True

    def undo(s, new):
        if new is True:
            used.discard(mapping[s])
            del mapping[s]

    def vals(s):
        return (mapping[s],) if s in mapping else [d for d in range(10) if d not in used]

    def rec(idx):
        if len(answers) > 1 or budget[0] <= 0:
            return
        budget[0] -= 1
        if idx == len(examples):
            qs0, qs1, qop, qs3, qs4 = query
            if any(s not in mapping for s in (qs0, qs1, qs3, qs4)):
                return
            ql = mapping[qs0] * 10 + mapping[qs1]
            qr = mapping[qs3] * 10 + mapping[qs4]
            d2s: dict[int, str] = {}
            for s, d in mapping.items():
                d2s.setdefault(d, s)
            opc = [opassign[qop]] if qop in opassign else list(ARITH_OPS)
            for opn in opc:
                rv = ARITH_OPS[opn][0](ql, qr)
                rd = _num_digits(rv)
                if all(d in d2s for d in rd):
                    ans = "".join(d2s[d] for d in rd)
                    answers.add(ans)
                    info[ans] = (dict(mapping), {**opassign, qop: opn})
                    if len(answers) > 1:
                        return
            return
        s0, s1, op, s3, s4, out = examples[idx]
        rlen = len(out)
        feas = [o for o in ARITH_OPS]
        n0 = None
        for d0 in vals(s0):
            n0 = assign(s0, d0)
            if n0 is None:
                continue
            for d1 in vals(s1):
                n1 = assign(s1, d1)
                if n1 is None:
                    continue
                lv = d0 * 10 + d1
                for d3 in vals(s3):
                    n3 = assign(s3, d3)
                    if n3 is None:
                        continue
                    for d4 in vals(s4):
                        n4 = assign(s4, d4)
                        if n4 is None:
                            continue
                        rv2 = d3 * 10 + d4
                        ops_try = [opassign[op]] if op in opassign else feas
                        for opn in ops_try:
                            val = ARITH_OPS[opn][0](lv, rv2)
                            rd = _num_digits(val)
                            if len(rd) != rlen:
                                continue
                            ass = []
                            ok = True
                            for rs, rdig in zip(out, rd):
                                ns = assign(rs, rdig)
                                if ns is None:
                                    ok = False
                                    break
                                ass.append((rs, ns))
                            if ok:
                                new = op not in opassign
                                if new:
                                    opassign[op] = opn
                                rec(idx + 1)
                                if new:
                                    del opassign[op]
                            for rs, ns in reversed(ass):
                                undo(rs, ns)
                            if len(answers) > 1:
                                undo(s4, n4); undo(s3, n3); undo(s1, n1); undo(s0, n0)
                                return
                        undo(s4, n4)
                    undo(s3, n3)
                undo(s1, n1)
            undo(s0, n0)

    rec(0)
    return answers, info


def cryptarithm_cot(examples, query, mapping, opassign, gold) -> str:
    lines = ["We treat each symbol as a digit and each operator as an arithmetic rule."]
    lines.append("Deducing the symbol -> digit mapping that is consistent across all examples:")
    for s, d in sorted(mapping.items()):
        lines.append(f"  {s} = {d}")
    lines.append("Operator meaning:")
    for s, opn in sorted(opassign.items()):
        lines.append(f"  {s} = {opn}")
    lines.append("")
    lines.append("Checking the examples:")
    for s0, s1, op, s3, s4, out in examples:
        lv = mapping[s0] * 10 + mapping[s1]
        rv = mapping[s3] * 10 + mapping[s4]
        opn = opassign[op]
        expr = ARITH_OPS[opn][1].format(a=lv, b=rv, r=ARITH_OPS[opn][0](lv, rv))
        lines.append(f"  {s0}{s1}{op}{s3}{s4}: {expr} -> {''.join(out)}")
    qs0, qs1, qop, qs3, qs4 = query
    ql = mapping[qs0] * 10 + mapping[qs1]
    qr = mapping[qs3] * 10 + mapping[qs4]
    opn = opassign[qop]
    expr = ARITH_OPS[opn][1].format(a=ql, b=qr, r=ARITH_OPS[opn][0](ql, qr))
    lines.append("")
    lines.append(f"Applying to {qs0}{qs1}{qop}{qs3}{qs4}: {expr}")
    lines.append(f"Re-encoding {ARITH_OPS[opn][0](ql, qr)} with the digit->symbol map gives {gold}.")
    return "\n".join(lines)


def gen_symbolic_arith(n: int, rng: random.Random) -> list[dict]:
    """Construct uniquely-solvable arithmetic cryptarithms, then solve + explain."""
    rows: list[dict] = []
    attempts = 0
    while len(rows) < n and attempts < n * 80:
        attempts += 1
        # random injective digit->symbol map for all 10 digits
        syms = rng.sample(SYM_ALPHA, 10)
        d2s = {d: syms[d] for d in range(10)}
        op_sym = rng.choice([s for s in SYM_ALPHA if s not in syms[:0]])
        opn = rng.choice(list(ARITH_OPS))

        def encode(n_val: int) -> str:
            return "".join(d2s[d] for d in _num_digits(n_val))

        def make_eq(a: int, b: int):
            inp = d2s[a // 10] + d2s[a % 10] + op_sym + d2s[b // 10] + d2s[b % 10]
            r = ARITH_OPS[opn][0](a, b)
            return inp, encode(r)

        # generate many examples to pin down the mapping for query symbols
        n_ex = rng.choice([4, 5])
        pairs = set()
        examples = []
        for _ in range(n_ex * 5):
            a = rng.randrange(10, 100)
            b = rng.randrange(10, 100)
            if (a, b) in pairs:
                continue
            pairs.add((a, b))
            examples.append(make_eq(a, b))
            if len(examples) == n_ex:
                break
        if len(examples) < n_ex:
            continue
        qa, qb = rng.randrange(10, 100), rng.randrange(10, 100)
        while (qa, qb) in pairs:
            qa, qb = rng.randrange(10, 100), rng.randrange(10, 100)
        q_inp = d2s[qa // 10] + d2s[qa % 10] + op_sym + d2s[qb // 10] + d2s[qb % 10]
        gold = encode(ARITH_OPS[opn][0](qa, qb))

        ex_tuples = [(i[0], i[1], i[2], i[3], i[4], tuple(o)) for i, o in examples]
        query_tuple = (q_inp[0], q_inp[1], q_inp[2], q_inp[3], q_inp[4])
        answers, info = solve_cryptarithm(ex_tuples, query_tuple)
        if len(answers) != 1:
            continue  # not uniquely solvable from examples -> skip
        ans = next(iter(answers))
        if not verify(gold, ans):
            continue
        mp, oa = info[ans]
        cot = cryptarithm_cot(ex_tuples, query_tuple, mp, oa, gold)
        rows.append(
            {
                " id": f"synthetic_v6_symarith_{len(rows):06d}",
                "prompt": eq_prompt(examples, q_inp),
                "answer": gold,
                "family": "equation_symbol_transformation",
                "rule_name": f"symbolic:arith_{opn}",
                "source": "synthetic_v6",
                "cot": cot,
                "has_cot": "True",
            }
        )
    return rows


def _parse_eq_prompt(prompt: str):
    exs = []
    q = None
    for ln in prompt.splitlines():
        ln = ln.strip()
        m = re.match(r"^(\S{5})\s*=\s*(\S+)$", ln)
        if m and "determine" not in ln:
            exs.append((m.group(1), m.group(2)))
        mq = re.search(r"result for:\s*(\S{5})", ln)
        if mq:
            q = mq.group(1)
    return exs, q


def solve_official_symbolic(rows: list[dict]) -> int:
    """Fill verified CoT for official symbolic rows the solver can uniquely solve."""
    filled = 0
    for r in rows:
        if r["family"] != "equation_symbol_transformation":
            continue
        if r.get("source") != "official_train" or (r.get("cot") or "").strip():
            continue
        exs, q = _parse_eq_prompt(r["prompt"])
        if not exs or not q or len(q) != 5:
            continue
        if any(len(i) != 5 for i, _ in exs):
            continue
        ex_tuples = [(i[0], i[1], i[2], i[3], i[4], tuple(o)) for i, o in exs]
        query_tuple = (q[0], q[1], q[2], q[3], q[4])
        try:
            answers, info = solve_cryptarithm(ex_tuples, query_tuple)
        except Exception:
            continue
        if len(answers) != 1:
            continue
        ans = next(iter(answers))
        if not verify(r["answer"], ans):
            continue
        mp, oa = info[ans]
        r["cot"] = cryptarithm_cot(ex_tuples, query_tuple, mp, oa, r["answer"])
        r["has_cot"] = "True"
        filled += 1
    return filled


def assemble_v6(in_csv: str, synth_rows: list[dict], out_csv: str, report_csv: str):
    with open(in_csv, newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    # Drop old v3 synthetic bit/equation; keep official + other synthetic families.
    kept = [
        r
        for r in rows
        if not (
            r.get("source") == "synthetic_v3"
            and r.get("family") in ("bit_manipulation", "equation_symbol_transformation")
        )
    ]
    # Fill verified CoT for official symbolic rows.
    official_filled = solve_official_symbolic(kept)
    out_rows = kept + synth_rows
    # Global cleanup: strip the boxed-preview brace tail that the cryptarithm
    # reasoner leaves on imported official rows ('output: 【X】-> 【{X}】').
    brace_cleaned = 0
    for r in out_rows:
        cot = r.get("cot") or ""
        if "-> 【{" in cot:
            new = _strip_brace_tail(cot)
            if new != cot:
                r["cot"] = new
                brace_cleaned += 1
    print(f"brace tails cleaned: {brace_cleaned}")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    fam_tot, fam_cot = Counter(), Counter()
    for r in out_rows:
        fam_tot[r["family"]] += 1
        if (r.get("cot") or "").strip():
            fam_cot[r["family"]] += 1
    import json

    rep = {
        "total_rows": len(out_rows),
        "official_symbolic_cot_filled": official_filled,
        "synth_added": len(synth_rows),
        "by_family": {
            fam: {"total": fam_tot[fam], "has_cot": fam_cot[fam]} for fam in fam_tot
        },
    }
    with open(report_csv, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bit", type=int, default=3500)
    ap.add_argument("--n-numeric", type=int, default=2500)
    ap.add_argument("--n-symcat", type=int, default=400)
    ap.add_argument("--n-symarith", type=int, default=900)
    ap.add_argument("--seed", type=int, default=20260606)
    ap.add_argument("--in-csv", default="data/train_plus_synthetic_v5.csv")
    ap.add_argument("--out", default="data/train_plus_synthetic_v6.csv")
    ap.add_argument("--report", default="data/train_plus_synthetic_v6_report.json")
    ap.add_argument("--synth-out", default="data/synthetic_v6.csv")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-assemble", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.smoke:
        args.n_bit = args.n_numeric = args.n_symcat = args.n_symarith = 20

    print(f"[bit] target {args.n_bit} ...")
    bit = gen_bit(args.n_bit, rng)
    print(f"  got {len(bit)}")
    print(f"[numeric] target {args.n_numeric} ...")
    num = gen_numeric(args.n_numeric, rng)
    print(f"  got {len(num)}")
    print(f"[symbolic-concat] target {args.n_symcat} ...")
    symcat = gen_symbolic_concat(args.n_symcat, rng)
    print(f"  got {len(symcat)}")
    print(f"[symbolic-arith] target {args.n_symarith} ...")
    symarith = gen_symbolic_arith(args.n_symarith, rng)
    print(f"  got {len(symarith)}")

    synth = bit + num + symcat + symarith
    fields = [" id", "prompt", "answer", "family", "rule_name", "source", "cot", "has_cot"]
    with open(args.synth_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(synth)
    print(f"Wrote {args.synth_out} with {len(synth)} synthetic rows")

    if not args.no_assemble:
        assemble_v6(args.in_csv, synth, args.out, args.report)


if __name__ == "__main__":
    main()
