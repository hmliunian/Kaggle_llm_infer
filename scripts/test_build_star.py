#!/usr/bin/env python3
"""Offline (GPU-free) tests for the STaR rule gate.

Run: .venv/bin/python -m pytest scripts/test_build_star.py -q
 or: .venv/bin/python scripts/test_build_star.py   (falls back to a runner)

The load-bearing tests are the NEGATIVE ones: a rationale whose boxed answer
equals gold but whose committed rule does NOT reproduce the shown examples
(lucky guess / leaked answer) MUST be rejected.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_star import keep_sample, parse_prompt  # noqa: E402
from scripts.star_rules import (  # noqa: E402
    apply_bit_rule,
    apply_numeric_rule,
    apply_symarith_rule,
    parse_bit_rule,
    parse_numeric_rule,
    parse_symarith_rule,
)


# ---------------------------------------------------------------------------
# helpers to build realistic prompts + generations
# ---------------------------------------------------------------------------
def bit_prompt(examples, query):
    lines = [
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.",
        "",
        "Here are some examples of input -> output:",
    ]
    lines += [f"{i} -> {o}" for i, o in examples]
    lines += ["", f"Now, determine the output for: {query}"]
    return "\n".join(lines)


def eq_prompt(examples, query):
    lines = ["In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:"]
    lines += [f"{i} = {o}" for i, o in examples]
    lines += [f"Now, determine the result for: {query}"]
    return "\n".join(lines)


def think_wrap(cot, boxed):
    return f"<think>\n{cot}\n</think>\nThe final answer is \\boxed{{{boxed}}}."


# ---------------------------------------------------------------------------
# bit_manipulation
# ---------------------------------------------------------------------------
def _bit_rule_fn(o, diff):
    # out[i] = XOR(in[(o+i)%8], in[(o+i+diff)%8])
    def f(bits):
        return "".join(
            ("1" if bits[(o + i) % 8] != bits[(o + i + diff) % 8] else "0")
            for i in range(8)
        )
    return f

def selected_block_xor(o, diff):
    lines = ["Selected"]
    for i in range(8):
        a = (o + i) % 8
        b = (o + i + diff) % 8
        lines.append(f"{i} XOR{a}{b}")
    return "\n".join(lines)


def test_bit_faithful_kept():
    f = _bit_rule_fn(2, 3)
    rng = random.Random(0)
    inputs = []
    seen = set()
    while len(inputs) < 8:
        x = "".join(rng.choice("01") for _ in range(8))
        if x in seen:
            continue
        seen.add(x); inputs.append(x)
    examples = [(x, f(x)) for x in inputs]
    q = "00110100"
    gold = f(q)
    cot = selected_block_xor(2, 3) + f"\n\nApplying to {q}\n(...)"
    gen = think_wrap(cot, gold)
    kept, reason = keep_sample("bit_manipulation", bit_prompt(examples, q), gold, gen)
    assert reason == "kept", reason
    assert kept and "Selected" in kept


def test_bit_lucky_guess_rejected():
    """Boxed answer is correct, but the Selected rule is WRONG (doesn't repro examples)."""
    f = _bit_rule_fn(2, 3)
    rng = random.Random(1)
    inputs = []
    seen = set()
    while len(inputs) < 8:
        x = "".join(rng.choice("01") for _ in range(8))
        if x in seen:
            continue
        seen.add(x); inputs.append(x)
    examples = [(x, f(x)) for x in inputs]
    q = "00110100"
    gold = f(q)
    # commit to a DIFFERENT rule (XOR with diff=1) that won't reproduce examples
    cot = selected_block_xor(0, 1)
    gen = think_wrap(cot, gold)  # gold is right, rule is wrong
    kept, reason = keep_sample("bit_manipulation", bit_prompt(examples, q), gold, gen)
    assert kept is None and reason == "rule_fail", reason


def test_bit_wrong_answer_rejected():
    f = _bit_rule_fn(2, 3)
    examples = [(x, f(x)) for x in ("01010001", "00001001", "00010101", "11111111",
                                    "10011101", "00111011", "10111101", "00100110")]
    q = "00110100"
    cot = selected_block_xor(2, 3)
    gen = think_wrap(cot, "00000000")  # wrong boxed answer
    kept, reason = keep_sample("bit_manipulation", bit_prompt(examples, q), f(q), gen)
    assert kept is None and reason == "answer_fail", reason


# ---------------------------------------------------------------------------
# numeric equation
# ---------------------------------------------------------------------------
def test_numeric_faithful_kept():
    # reverse concatenation on reversed operands: f(a,b) = rev(b) + rev(a)
    def f(a, b):
        return b[::-1] + a[::-1]
    examples = [("29", "16"), ("52", "97"), ("43", "66")]
    ex = [(f"{a}|{b}", f(a, b)) for a, b in examples]
    q = "44|89"
    gold = f("44", "89")
    cot = (
        "Identified matching operation from examples:\n"
        "reverse concatenation f(25, 79) = ..., correct, actions: reversed operands, reverse concatenation\n"
        "\nApplying to 44|89:\n  reversed operands [44->44, 89->98]\n  Result: 【9844】"
    )
    gen = think_wrap(cot, gold)
    kept, reason = keep_sample("equation_symbol_transformation", eq_prompt(ex, q), gold, gen)
    assert reason == "kept", reason


def test_numeric_lucky_guess_rejected():
    def f(a, b):
        return b[::-1] + a[::-1]
    examples = [("29", "16"), ("52", "97"), ("43", "66")]
    ex = [(f"{a}|{b}", f(a, b)) for a, b in examples]
    q = "44|89"
    gold = f("44", "89")
    # claim plain "addition" — wrong op, won't reproduce examples
    cot = "blah, correct, actions: addition\nResult: 【9844】"
    gen = think_wrap(cot, gold)
    kept, reason = keep_sample("equation_symbol_transformation", eq_prompt(ex, q), gold, gen)
    assert kept is None and reason == "rule_fail", reason


# ---------------------------------------------------------------------------
# symbolic arithmetic (the under-determined danger zone)
# ---------------------------------------------------------------------------
def _symarith_setup(seed=3):
    rng = random.Random(seed)
    alpha = list("!\"#$%&'()/:<>?@[]^`{|}")
    syms = rng.sample(alpha, 10)
    d2s = {d: syms[d] for d in range(10)}
    op_sym = rng.choice([s for s in alpha if s not in syms])
    def enc(n):
        return "".join(d2s[int(c)] for c in str(abs(n)))
    def make(a, b):
        inp = d2s[a // 10] + d2s[a % 10] + op_sym + d2s[b // 10] + d2s[b % 10]
        return inp, enc(a * b)
    return d2s, op_sym, enc, make


def test_symarith_faithful_kept():
    d2s, op_sym, enc, make = _symarith_setup()
    pairs = [(13, 16), (74, 69), (38, 49), (57, 91)]
    ex = [make(a, b) for a, b in pairs]
    qa, qb = 61, 76
    q_inp = d2s[qa // 10] + d2s[qa % 10] + op_sym + d2s[qb // 10] + d2s[qb % 10]
    gold = enc(qa * qb)
    map_lines = "\n".join(f"  {d2s[d]} = {d}" for d in range(10))
    cot = (
        "We treat each symbol as a digit and each operator as an arithmetic rule.\n"
        "Deducing the symbol -> digit mapping that is consistent across all examples:\n"
        f"{map_lines}\n"
        "Operator meaning:\n"
        f"  {op_sym} = mul\n"
    )
    gen = think_wrap(cot, gold)
    kept, reason = keep_sample("equation_symbol_transformation", eq_prompt(ex, q_inp), gold, gen)
    assert reason == "kept", reason


def test_symarith_leaked_answer_rejected():
    """Right boxed answer, but the stated mapping does NOT reproduce examples."""
    d2s, op_sym, enc, make = _symarith_setup()
    pairs = [(13, 16), (74, 69), (38, 49), (57, 91)]
    ex = [make(a, b) for a, b in pairs]
    qa, qb = 61, 76
    q_inp = d2s[qa // 10] + d2s[qa % 10] + op_sym + d2s[qb // 10] + d2s[qb % 10]
    gold = enc(qa * qb)
    # corrupt the mapping: swap two digits so example reproduction fails
    bad = dict(d2s)
    bad[0], bad[1] = d2s[1], d2s[0]
    map_lines = "\n".join(f"  {bad[d]} = {d}" for d in range(10))
    cot = (
        "Deducing the symbol -> digit mapping that is consistent across all examples:\n"
        f"{map_lines}\n"
        "Operator meaning:\n"
        f"  {op_sym} = mul\n"
    )
    gen = think_wrap(cot, gold)  # leaked correct answer, wrong derivation
    kept, reason = keep_sample("equation_symbol_transformation", eq_prompt(ex, q_inp), gold, gen)
    assert kept is None and reason == "rule_fail", reason


def test_symarith_no_rule_gate_admits_leak():
    """Sanity: with the rule gate OFF, the leaked-answer case slips through —
    demonstrating exactly why the gate matters."""
    d2s, op_sym, enc, make = _symarith_setup()
    pairs = [(13, 16), (74, 69), (38, 49), (57, 91)]
    ex = [make(a, b) for a, b in pairs]
    qa, qb = 61, 76
    q_inp = d2s[qa // 10] + d2s[qa % 10] + op_sym + d2s[qb // 10] + d2s[qb % 10]
    gold = enc(qa * qb)
    gen = think_wrap("I just guessed.", gold)
    kept, reason = keep_sample(
        "equation_symbol_transformation", eq_prompt(ex, q_inp), gold, gen,
        require_rule_gate=False,
    )
    assert reason == "kept", reason  # admitted — the danger we avoid by default


# ---------------------------------------------------------------------------
# prompt parsing
# ---------------------------------------------------------------------------
def test_parse_prompt_bit():
    ex = [("01010001", "11011101"), ("00001001", "01101101")]
    p = bit_prompt(ex, "00110100")
    pe, q = parse_prompt("bit_manipulation", p)
    assert pe == ex and q == "00110100"


def test_parse_prompt_equation():
    ex = [("%|*\"|", "%|\"|"), ("(%+[@", "(%[@")]
    p = eq_prompt(ex, "\\(*[#")
    pe, q = parse_prompt("equation_symbol_transformation", p)
    assert q == "\\(*[#"
    assert ex[0] in pe


# ---------------------------------------------------------------------------
# unit checks on the executors
# ---------------------------------------------------------------------------
def test_apply_bit_rule_roundtrip():
    rule = parse_bit_rule("Selected\n0 I0\n1 NOT1\n2 C0\n3 C1\n4 I4\n5 NOT5\n6 XOR67\n7 OR-NOT01\n\nApplying to x")
    assert rule is not None
    assert apply_bit_rule(rule, "10101010") is not None


def test_apply_numeric_rule():
    r = parse_numeric_rule("correct, actions: reversed operands, reverse concatenation")
    assert r is not None
    # reversed operands: 44->44, 89->98 ; reverse concat: 98 || 44 -> 9844
    assert apply_numeric_rule(r, "44|89") == "9844"


def test_apply_symarith_rule():
    cot = ("symbol -> digit mapping\n  ! = 1\n  @ = 2\n  # = 3\n  $ = 4\n"
           "Operator meaning\n  + = add\n")
    r = parse_symarith_rule(cot)
    assert r is not None
    assert r.mapping == {"!": 1, "@": 2, "#": 3, "$": 4}
    assert r.op_sym == "+" and r.op_name == "add"


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
