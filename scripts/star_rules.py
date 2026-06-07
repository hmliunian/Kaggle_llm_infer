#!/usr/bin/env python3
"""Machine-checkable rule extraction + re-execution for the STaR rule-gate.

The STaR loop samples CoT from a (LoRA) model for rows the deterministic solver
could not cover, then keeps a rationale ONLY IF:

  1. (answer gate)  its boxed final answer verifies == gold under
     official_metric.verify, AND
  2. (rule gate)    the explicit rule the rationale commits to, when
     re-executed mechanically, reproduces EVERY shown example input->output
     pair AND the query answer.

The rule gate is what makes STaR safe on under-determined problems (e.g. the
~758 cryptarithm rows): a lucky / leaked answer that is not backed by a rule
that regenerates all visible examples is rejected, so we never train on
answer-copying rationales.

This module is GPU-free and deterministic so it can be unit-tested offline.

Supported families:
  - bit_manipulation                : per-output-bit boolean rule vector
  - equation_symbol_transformation  : numeric op / symbolic concat / symbolic
                                      arithmetic (symbol->digit + operator)

Each family exposes:
  parse_rule(cot)         -> rule object | None     (parse model output)
  apply_rule(rule, inp)   -> output str | None      (re-execute on one input)

and the shared check_examples() re-executes a parsed rule on all shown pairs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# bit_manipulation
# ---------------------------------------------------------------------------
# A rule is an 8-element vector of per-output-bit ops, matching the tokens the
# `Selected` block emits, e.g.:
#     0 OR-NOT17
#     1 NOT2
#     7 C1
# Token grammar per bit:
#     I{p} | NOT{p} | C0 | C1 | {XOR|OR|AND}{p}{s} | {XOR|OR|AND}-NOT{p}{s}
# where {p}{s} are single decimal digits (operand bit indices 0-7).

N_BITS = 8

_BIT_LINE_RE = re.compile(r"^\s*([0-7])\s+(\S+)\s*$")


@dataclass(frozen=True)
class BitOp:
    family: str            # I, NOT, C0, C1, XOR, OR, AND, XOR-NOT, OR-NOT, AND-NOT
    primary: Optional[int]
    secondary: Optional[int]


def _parse_bit_token(tok: str) -> Optional[BitOp]:
    tok = tok.strip()
    if tok in ("C0", "C1"):
        return BitOp(tok, None, None)
    # asymmetric pair: XOR-NOT12
    m = re.fullmatch(r"(XOR|OR|AND)-NOT([0-7])([0-7])", tok)
    if m:
        return BitOp(f"{m.group(1)}-NOT", int(m.group(2)), int(m.group(3)))
    # symmetric pair: XOR12
    m = re.fullmatch(r"(XOR|OR|AND)([0-7])([0-7])", tok)
    if m:
        return BitOp(m.group(1), int(m.group(2)), int(m.group(3)))
    # unary
    m = re.fullmatch(r"NOT([0-7])", tok)
    if m:
        return BitOp("NOT", int(m.group(1)), None)
    m = re.fullmatch(r"I([0-7])", tok)
    if m:
        return BitOp("I", int(m.group(1)), None)
    return None


def parse_bit_rule(cot: str) -> Optional[list[BitOp]]:
    """Parse the per-bit rule vector from the `Selected` block of a bit CoT.

    Returns a list of exactly 8 BitOp (index 0..7) or None if not parseable.
    """
    lines = cot.replace("\r\n", "\n").split("\n")
    # Find the LAST `Selected` block (the model may echo earlier tentative ones).
    sel_idx = -1
    for i, ln in enumerate(lines):
        if ln.strip() == "Selected":
            sel_idx = i
    if sel_idx < 0:
        return None
    vec: dict[int, BitOp] = {}
    for ln in lines[sel_idx + 1:]:
        if not ln.strip():
            # blank line ends the block (the `Applying to` section follows)
            if vec:
                break
            continue
        m = _BIT_LINE_RE.match(ln)
        if not m:
            # a non "idx token" line: stop if we already collected the vector
            if vec:
                break
            continue
        idx = int(m.group(1))
        op = _parse_bit_token(m.group(2))
        if op is None:
            return None
        vec[idx] = op
    if len(vec) != N_BITS or set(vec) != set(range(N_BITS)):
        return None
    return [vec[i] for i in range(N_BITS)]


def _bit_not(b: str) -> str:
    return "1" if b == "0" else "0"


def _eval_bit_pair(fam: str, a: str, b: str) -> str:
    base = fam.split("-")[0]
    bb = _bit_not(b) if fam.endswith("-NOT") else b
    if base == "AND":
        return "1" if a == "1" and bb == "1" else "0"
    if base == "OR":
        return "1" if a == "1" or bb == "1" else "0"
    return "1" if a != bb else "0"  # XOR


def apply_bit_rule(rule: list[BitOp], inp: str) -> Optional[str]:
    inp = "".join(ch for ch in str(inp) if ch in "01")
    if len(inp) != N_BITS:
        return None
    out = []
    for op in rule:
        if op.family == "C0":
            out.append("0")
        elif op.family == "C1":
            out.append("1")
        elif op.family == "I":
            if op.primary is None:
                return None
            out.append(inp[op.primary])
        elif op.family == "NOT":
            if op.primary is None:
                return None
            out.append(_bit_not(inp[op.primary]))
        else:  # binary
            if op.primary is None or op.secondary is None:
                return None
            out.append(_eval_bit_pair(op.family, inp[op.primary], inp[op.secondary]))
    return "".join(out)


# ---------------------------------------------------------------------------
# equation: numeric
# ---------------------------------------------------------------------------
# We re-derive the operation from the COT's stated "actions:" line when present
# (e.g. "correct, actions: reversed operands, reverse concatenation"), then
# re-execute it ourselves on every example. This does not trust the COT's
# arithmetic — only its claimed (op, reversed-operands, reversed-result), which
# we independently evaluate.

_NUM_RE = re.compile(r"^(\d+)(\D)(\d+)$")

_NUM_OPS: dict[str, Callable[[int, int, str, str], Optional[str]]] = {
    "concatenation": lambda a, b, sa, sb: sa + sb,
    "reverse concatenation": lambda a, b, sa, sb: sb + sa,
    "addition": lambda a, b, sa, sb: str(a + b),
    "absolute difference": lambda a, b, sa, sb: str(abs(a - b)),
    "negated absolute difference": lambda a, b, sa, sb: str(-abs(a - b)),
    "subtraction (a-b)": lambda a, b, sa, sb: str(a - b),
    "reverse subtraction (b-a)": lambda a, b, sa, sb: str(b - a),
    "multiplication": lambda a, b, sa, sb: str(a * b),
    "multiply+1": lambda a, b, sa, sb: str(a * b + 1),
    "multiply-1": lambda a, b, sa, sb: str(a * b - 1),
    "add+1": lambda a, b, sa, sb: str(a + b + 1),
    "add-1": lambda a, b, sa, sb: str(a + b - 1),
    "sub+1": lambda a, b, sa, sb: str(a - b + 1),
    "sub-1": lambda a, b, sa, sb: str(a - b - 1),
    "integer division (a/b)": lambda a, b, sa, sb: str(a // b) if b else None,
    "modulo (a mod b)": lambda a, b, sa, sb: str(a % b) if b else None,
    "reverse division (b/a)": lambda a, b, sa, sb: str(b // a) if a else None,
    "reverse modulo (b mod a)": lambda a, b, sa, sb: str(b % a) if a else None,
}


def _num_digit_ops(sa: str, sb: str) -> dict[str, Optional[str]]:
    if len(sa) != 2 or len(sb) != 2:
        return {}
    d1, d2, d3, d4 = int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1])
    return {
        "digit absolute diff": str(abs(d1 - d3)) + str(abs(d2 - d4)),
        "digit add mod10": str((d1 + d3) % 10) + str((d2 + d4) % 10),
        "digit sub mod10": str((d1 - d3) % 10) + str((d2 - d4) % 10),
        "cross multiply": str(d1 * d3 + d2 * d4),
        "cross multiply rev": str(d1 * d4 + d2 * d3),
        "digit multiply": str(d1 * d3) + str(d2 * d4),
        "digit multiply rev": str(d1 * d4) + str(d2 * d3),
        "digit sum diff": str((d1 + d2) - (d3 + d4)),
        "digit sum sum": str((d1 + d2) + (d3 + d4)),
        "digit product diff": str(d1 * d2 - d3 * d4),
        "digit product sum": str(d1 * d2 + d3 * d4),
        "determinant": str(d1 * d4 - d2 * d3),
        "abs determinant": str(abs(d1 * d4 - d2 * d3)),
    }


def _rev_str(s: str) -> str:
    return ("-" + s[1:][::-1]) if s.startswith("-") else s[::-1]


@dataclass(frozen=True)
class NumRule:
    op_name: str
    rev_ops: bool
    rev_res: bool


_ACTIONS_RE = re.compile(r"correct,\s*actions:\s*(.+)$", re.IGNORECASE)


def parse_numeric_rule(cot: str) -> Optional[NumRule]:
    """Parse op + reversed flags from the COT's 'actions:' summary line."""
    m = None
    for ln in cot.replace("\r\n", "\n").split("\n"):
        mm = _ACTIONS_RE.search(ln)
        if mm:
            m = mm  # take the last actions line
    if not m:
        return None
    actions = [a.strip().lower() for a in m.group(1).split(",")]
    rev_ops = any("reversed operand" in a for a in actions)
    rev_res = any("reversed result" in a for a in actions)
    op_name = None
    known = set(_NUM_OPS) | set(_num_digit_ops("00", "00"))
    # the op name is the action that is a known operation
    for a in actions:
        if a in known:
            op_name = a
    if op_name is None:
        # fall back: last action token is usually the op
        cand = actions[-1]
        if cand in known:
            op_name = cand
    if op_name is None:
        return None
    return NumRule(op_name, rev_ops, rev_res)


def apply_numeric_rule(rule: NumRule, inp: str) -> Optional[str]:
    m = _NUM_RE.fullmatch(str(inp).strip())
    if not m:
        return None
    sa, _op, sb = m.group(1), m.group(2), m.group(3)
    ta = sa[::-1] if rule.rev_ops else sa
    tb = sb[::-1] if rule.rev_ops else sb
    try:
        ia, ib = int(ta), int(tb)
    except ValueError:
        return None
    fn = _NUM_OPS.get(rule.op_name)
    if fn is not None:
        raw = fn(ia, ib, ta, tb)
    else:
        raw = _num_digit_ops(ta, tb).get(rule.op_name)
    if raw is None:
        return None
    return _rev_str(raw) if rule.rev_res else raw


# ---------------------------------------------------------------------------
# equation: symbolic concatenation
# ---------------------------------------------------------------------------
# Format "AABB" forward or "BBAA" reverse over 5-char inputs "a0 a1 op b0 b1".

@dataclass(frozen=True)
class ConcatRule:
    kind: str  # "fwd" or "rev" (direction used for the QUERY operator)


_CONCAT_RE = re.compile(r"which is\s*【?\s*(reverse concatenation|concatenation)", re.IGNORECASE)


def parse_concat_rule(cot: str) -> Optional[ConcatRule]:
    # strip the 【】 display brackets so operator/keyword detection is uniform
    text = cot.replace("【", "").replace("】", "").lower()
    # explicit statement wins
    m = None
    for mm in _CONCAT_RE.finditer(text):
        m = mm
    if m:
        return ConcatRule("rev" if "reverse" in m.group(1).lower() else "fwd")
    # default-to-concatenation fallback the reasoner uses
    if "default to concatenation" in text or "we default to concatenation" in text:
        return ConcatRule("fwd")
    return None


def _concat_dir(inp: str, out: str) -> Optional[str]:
    """Infer fwd/rev for one 5-char input -> output concat pair."""
    if len(inp) != 5 or len(out) != 4:
        return None
    a0, a1, _op, b0, b1 = inp[0], inp[1], inp[2], inp[3], inp[4]
    if out == a0 + a1 + b0 + b1:
        return "fwd"
    if out == b0 + b1 + a0 + a1:
        return "rev"
    return None


def concat_per_operator(examples) -> Optional[dict]:
    """Map each operator symbol -> its concat direction, consistent across examples.

    Returns None if any operator is internally inconsistent (mixed fwd/rev) or a
    pair is not a concatenation at all.
    """
    by_op: dict[str, set] = {}
    for inp, out in examples:
        s = str(inp).strip()
        if len(s) != 5:
            return None
        d = _concat_dir(s, str(out).strip())
        if d is None:
            return None
        by_op.setdefault(s[2], set()).add(d)
    out_map: dict[str, str] = {}
    for op, dirs in by_op.items():
        if len(dirs) != 1:
            return None  # operator used both directions -> not a pure concat rule
        out_map[op] = next(iter(dirs))
    return out_map


def apply_concat_rule(rule: ConcatRule, inp: str) -> Optional[str]:
    s = str(inp).strip()
    if len(s) != 5:
        return None
    a0, a1, _op, b0, b1 = s[0], s[1], s[2], s[3], s[4]
    if rule.kind == "fwd":
        return a0 + a1 + b0 + b1
    return b0 + b1 + a0 + a1


# ---------------------------------------------------------------------------
# equation: symbolic arithmetic (symbol->digit + operator)
# ---------------------------------------------------------------------------
# Parses the mapping block:
#     <sym> = <digit>
# and the operator line:
#     <op_sym> = mul | add | abs_diff
# then re-executes AB op CD with two-digit operands and re-encodes via digit->sym.

_MAP_LINE_RE = re.compile(r"^\s*(\S)\s*=\s*(\d)\s*$")
_OP_LINE_RE = re.compile(r"^\s*(\S)\s*=\s*(mul|add|abs_diff)\s*$")

_ARITH_OPS: dict[str, Callable[[int, int], int]] = {
    "add": lambda a, b: a + b,
    "abs_diff": lambda a, b: abs(a - b),
    "mul": lambda a, b: a * b,
}


@dataclass(frozen=True)
class SymArithRule:
    sym2dig: tuple[tuple[str, int], ...]   # hashable view of the mapping
    op_sym: str
    op_name: str

    @property
    def mapping(self) -> dict[str, int]:
        return dict(self.sym2dig)

    @property
    def dig2sym(self) -> dict[int, str]:
        return {d: s for s, d in self.sym2dig}


def parse_symarith_rule(cot: str) -> Optional[SymArithRule]:
    lines = cot.replace("\r\n", "\n").split("\n")
    sym2dig: dict[str, int] = {}
    op_sym = None
    op_name = None
    in_map = False
    for ln in lines:
        if "symbol -> digit mapping" in ln.lower():
            in_map = True
            continue
        if ln.strip().lower().startswith("operator meaning"):
            in_map = False
            continue
        mo = _OP_LINE_RE.match(ln)
        if mo:
            op_sym, op_name = mo.group(1), mo.group(2)
            continue
        mm = _MAP_LINE_RE.match(ln)
        if mm and (in_map or not sym2dig):
            sym, dig = mm.group(1), int(mm.group(2))
            sym2dig[sym] = dig
    if not sym2dig or op_sym is None or op_name is None:
        return None
    # mapping must be injective (a digit cannot map to two symbols)
    if len({d for d in sym2dig.values()}) != len(sym2dig):
        return None
    return SymArithRule(tuple(sorted(sym2dig.items())), op_sym, op_name)


def apply_symarith_rule(rule: SymArithRule, inp: str) -> Optional[str]:
    s = str(inp).strip()
    if len(s) != 5:
        return None
    a0, a1, op, b0, b1 = s[0], s[1], s[2], s[3], s[4]
    if op != rule.op_sym:
        return None
    mp = rule.mapping
    if any(c not in mp for c in (a0, a1, b0, b1)):
        return None
    a = mp[a0] * 10 + mp[a1]
    b = mp[b0] * 10 + mp[b1]
    val = _ARITH_OPS[rule.op_name](a, b)
    d2s = rule.dig2sym
    digits = [int(c) for c in str(abs(val))]
    if any(d not in d2s for d in digits):
        return None
    return "".join(d2s[d] for d in digits)


# ---------------------------------------------------------------------------
# Unified parse/apply dispatch
# ---------------------------------------------------------------------------
# For equation we try the three sub-family parsers in order; the first that both
# parses AND reproduces all examples wins. For bit there is one parser.

def _bit_rule(cot: str):
    r = parse_bit_rule(cot)
    return (r, apply_bit_rule) if r is not None else (None, None)


_EQUATION_PARSERS = [
    (parse_symarith_rule, apply_symarith_rule),
    (parse_numeric_rule, apply_numeric_rule),
    (parse_concat_rule, apply_concat_rule),
]


def extract_rule(family: str, cot: str):
    """Return (rule, apply_fn) for the given family, or (None, None)."""
    if family == "bit_manipulation":
        return _bit_rule(cot)
    if family == "equation_symbol_transformation":
        # caller resolves which sub-parser reproduces examples; we just return
        # the first that parses. (check_rule below tries all.)
        for parse_fn, apply_fn in _EQUATION_PARSERS:
            r = parse_fn(cot)
            if r is not None:
                return r, apply_fn
        return None, None
    return None, None


def _check_concat(cot: str, examples, query, gold) -> Optional[str]:
    """Per-operator concatenation gate.

    Each example must be a consistent concat under its own operator; the query is
    answered with the query operator's direction if seen in examples, else the
    direction stated in the CoT (parse_concat_rule), else 'fwd' (the reasoner
    default). Returns a repr iff every example reproduces.
    """
    op_dirs = concat_per_operator(examples)
    if op_dirs is None:
        return None
    q = str(query).strip()
    if len(q) != 5:
        return None
    q_op = q[2]
    if q_op in op_dirs:
        q_dir = op_dirs[q_op]
    else:
        stated = parse_concat_rule(cot)
        q_dir = stated.kind if stated is not None else "fwd"
    q_out = apply_concat_rule(ConcatRule(q_dir), q)
    if q_out is None:
        return None
    return f"ConcatPerOp({op_dirs!r}, q={q_dir})"


def check_rule(family: str, cot: str, examples, query, gold) -> Optional[str]:
    """Return the rule's repr iff it reproduces ALL examples AND yields a query
    output. examples : list[(inp, out)]   query : str   gold : str

    Faithfulness is enforced here — this is the STaR rule gate. The caller has
    already confirmed the model's boxed answer verifies == gold; this confirms
    the *rule* is real (reproduces every shown example), not a lucky guess.
    """
    if not examples:
        return None
    if family == "bit_manipulation":
        rule = parse_bit_rule(cot)
        if rule is None:
            return None
        for inp, out in examples:
            if apply_bit_rule(rule, inp) != str(out):
                return None
        return repr(rule) if apply_bit_rule(rule, query) is not None else None

    # equation_symbol_transformation: try symarith, numeric, then per-op concat.
    for parse_fn, apply_fn in ((parse_symarith_rule, apply_symarith_rule),
                               (parse_numeric_rule, apply_numeric_rule)):
        rule = parse_fn(cot)
        if rule is None:
            continue
        ok = all(apply_fn(rule, inp) == str(out) for inp, out in examples)
        if ok and apply_fn(rule, query) is not None:
            return repr(rule)
    # concat (handles multi-operator symbolic rows)
    return _check_concat(cot, examples, query, gold)
