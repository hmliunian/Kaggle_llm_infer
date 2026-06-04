#!/usr/bin/env python3
"""Preview ground-truth CoT traces for a family. Generates N synthetic samples,
builds the CoT, self-checks it reproduces the gold answer, and prints them.

Usage:
    python scripts/preview_cot.py <family> [n] [seed]
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_synthetic_data import GENERATORS
from cot_builders import build_cot, CotError


def main(family, n, seed):
    rng = random.Random(seed)
    gen = GENERATORS[family]
    ok = 0
    for i in range(n):
        ex = gen(f"preview_{family[:3]}_{i:04d}", rng)
        try:
            cot = build_cot(ex.family, ex.prompt, ex.answer, ex.rule_payload)
            ok += 1
        except CotError as e:
            cot = f"<<COT ERROR: {e}>>"
        print("=" * 90)
        print(f"[{i}] rule_name={ex.rule_name}  payload={ex.rule_payload}")
        print("-- PROMPT --")
        print(ex.prompt)
        print("-- GOLD --", repr(ex.answer))
        print("-- COT --")
        print(cot)
        print(f"-- FULL TARGET --\n<think>\n{cot}\n</think>\nThe final answer is \\boxed{{{ex.answer}}}.")
        print()
    print("#" * 90)
    print(f"{family}: CoT self-check passed {ok}/{n}")


if __name__ == "__main__":
    family = sys.argv[1] if len(sys.argv) > 1 else "bit_manipulation"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    main(family, n, seed)
