#!/usr/bin/env python3
"""
Generate verified synthetic SFT data for the Nemotron reasoning task.

The script creates answer-only examples matching the public train.csv prompt
style. Rules are generated and solved programmatically; no chain-of-thought text
is produced.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_CSV = BASE_DIR / "data" / "train.csv"
DEFAULT_OUT_DIR = BASE_DIR / "data"
DEFAULT_SYNTHETIC_VERSION = "synthetic_v3"
WEAK_V3_FAMILY_COUNTS = {
    "bit_manipulation": 3000,
    "equation_symbol_transformation": 3000,
    "gravity_formula": 3000,
    "numeral_system": 500,
    "text_decryption": 1500,
    "unit_conversion": 500,
}

FAMILIES = [
    "bit_manipulation",
    "equation_symbol_transformation",
    "gravity_formula",
    "numeral_system",
    "text_decryption",
    "unit_conversion",
]


@dataclass(frozen=True)
class SyntheticExample:
    row_id: str
    prompt: str
    answer: str
    family: str
    rule_name: str
    rule_payload: dict


def classify_prompt(prompt: str) -> str:
    s = prompt.lower()
    if "secret bit manipulation" in s:
        return "bit_manipulation"
    if "d = 0.5*g*t^2" in s or "falling distance" in s:
        return "gravity_formula"
    if "convert the following measurement" in s or ("measurement" in s and "becomes" in s):
        return "unit_conversion"
    if "decrypt" in s:
        return "text_decryption"
    if "wonderland numeral" in s or "write the number" in s:
        return "numeral_system"
    if "determine the result for" in s:
        return "equation_symbol_transformation"
    return "unknown"


def fmt_float(value: float, places: int = 2) -> str:
    return f"{value:.{places}f}"


def fmt_measure(value: float, rng: random.Random) -> str:
    if rng.random() < 0.25:
        return f"{value:.1f}"
    return f"{value:.2f}"


def to_bin8(value: int) -> str:
    return format(value & 0xFF, "08b")


def rotl8(value: int, shift: int) -> int:
    shift %= 8
    return ((value << shift) | (value >> (8 - shift))) & 0xFF


def rotr8(value: int, shift: int) -> int:
    shift %= 8
    return ((value >> shift) | (value << (8 - shift))) & 0xFF


def bit_reverse8(value: int) -> int:
    return int(format(value & 0xFF, "08b")[::-1], 2)


def make_bit_rule(rng: random.Random) -> tuple[str, Callable[[int], int], dict]:
    kind = rng.choice(["xor", "rotl_xor", "reverse_xor"])
    if kind == "xor":
        mask = rng.randrange(1, 256)
        return f"xor_{mask:02x}", lambda x, m=mask: x ^ m, {"kind": kind, "mask": mask}
    if kind == "rotl_xor":
        shift = rng.randrange(1, 8)
        mask = rng.randrange(0, 256)
        return (
            f"rotl{shift}_xor_{mask:02x}",
            lambda x, s=shift, m=mask: rotl8(x, s) ^ m,
            {"kind": kind, "shift": shift, "mask": mask},
        )
    if kind == "reverse_xor":
        mask = rng.randrange(0, 256)
        return (
            f"reverse_xor_{mask:02x}",
            lambda x, m=mask: bit_reverse8(x) ^ m,
            {"kind": kind, "mask": mask},
        )

    raise ValueError(f"Unknown bit rule kind: {kind}")


def iter_bit_candidate_signatures(examples: list[int]) -> dict[tuple[int, ...], list[str]]:
    signatures: dict[tuple[int, ...], list[str]] = {}

    def add(name: str, rule: Callable[[int], int]) -> None:
        signature = tuple(rule(x) for x in examples)
        signatures.setdefault(signature, []).append(name)

    for mask in range(256):
        add(f"xor_{mask:02x}", lambda x, m=mask: x ^ m)
        add(f"reverse_xor_{mask:02x}", lambda x, m=mask: bit_reverse8(x) ^ m)
    for shift in range(1, 8):
        for mask in range(256):
            add(f"rotl{shift}_xor_{mask:02x}", lambda x, s=shift, m=mask: rotl8(x, s) ^ m)
    return signatures


def gen_bit(row_id: str, rng: random.Random) -> SyntheticExample:
    rule_name, rule, payload = make_bit_rule(rng)
    n_examples = rng.randint(10, 14)
    for _ in range(100):
        values = rng.sample(range(256), n_examples + 1)
        examples = values[:n_examples]
        signature = tuple(rule(x) for x in examples)
        matches = iter_bit_candidate_signatures(examples).get(signature, [])
        if len(matches) == 1 and matches[0] == rule_name:
            break
    else:
        raise RuntimeError(f"Could not make uniquely identifiable bit rule: {rule_name}")
    query = values[-1]
    lines = [f"{to_bin8(x)} -> {to_bin8(rule(x))}" for x in examples]
    prompt = (
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
        "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
        "and possibly majority or choice functions.\n\n"
        "Here are some examples of input -> output:\n"
        + "\n".join(lines)
        + f"\n\nNow, determine the output for: {to_bin8(query)}"
    )
    return SyntheticExample(row_id, prompt, to_bin8(rule(query)), "bit_manipulation", rule_name, payload)


EQUATION_SYMBOLS = list("!@#$%^&*()[]{}<>?/\\|~`'\"-+_:;")
EQUATION_OP_SYMBOLS = list("+-*|@')\"")


def numeric_equation_rule(rule_name: str, left: str, right: str) -> str:
    a, b = int(left[0]), int(left[1])
    c, d = int(right[0]), int(right[1])
    if rule_name == "absdiff":
        return f"{abs(a - c)}{abs(b - d)}"
    if rule_name == "sum_mod":
        return f"{(a + c) % 10}{(b + d) % 10}"
    if rule_name == "prod_mod":
        return f"{(a * c) % 10}{(b * d) % 10}"
    if rule_name == "cross_sum":
        return f"{(a + d) % 10}{(b + c) % 10}"
    if rule_name == "swap_concat":
        return f"{right}{left}"
    if rule_name == "outer_inner_abs":
        return f"{abs(a - d)}{abs(b - c)}"
    raise ValueError(f"Unknown numeric equation rule: {rule_name}")


def gen_numeric_equation(row_id: str, rng: random.Random) -> SyntheticExample:
    op_symbols = rng.sample(EQUATION_OP_SYMBOLS, rng.randint(3, 5))
    rule_names = rng.sample(
        ["absdiff", "sum_mod", "prod_mod", "cross_sum", "swap_concat", "outer_inner_abs"],
        len(op_symbols),
    )
    op_rules = dict(zip(op_symbols, rule_names))

    def make_expr(required_op: str | None = None) -> tuple[str, str]:
        op = required_op if required_op is not None else rng.choice(op_symbols)
        left = f"{rng.randrange(10, 100):02d}"
        right = f"{rng.randrange(10, 100):02d}"
        return f"{left}{op}{right}", numeric_equation_rule(op_rules[op], left, right)

    query_op = rng.choice(op_symbols)
    query, answer = make_expr(query_op)
    n_examples = rng.randint(max(5, len(op_symbols) + 1), 8)
    examples = []
    seen_inputs = set()
    for op in op_symbols:
        expr, out = make_expr(op)
        examples.append((expr, out))
        seen_inputs.add(expr)
    while len(examples) < n_examples:
        expr, out = make_expr()
        if expr in seen_inputs:
            continue
        seen_inputs.add(expr)
        examples.append((expr, out))
    rng.shuffle(examples)

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:\n"
        + "\n".join(f"{src} = {dst}" for src, dst in examples)
        + f"\nNow, determine the result for: {query}"
    )
    payload = {"kind": "numeric_operator_rules", "op_rules": op_rules}
    return SyntheticExample(
        row_id,
        prompt,
        answer,
        "equation_symbol_transformation",
        "numeric_operator_rules",
        payload,
    )


def gen_equation(row_id: str, rng: random.Random) -> SyntheticExample:
    if rng.random() < 0.55:
        return gen_numeric_equation(row_id, rng)

    alphabet = EQUATION_SYMBOLS[:]
    shuffled = alphabet[:]
    rng.shuffle(shuffled)
    mapping = dict(zip(alphabet, shuffled))

    def transform(text: str) -> str:
        return "".join(mapping[ch] for ch in text)

    query_len = rng.randint(3, 6)
    query = "".join(rng.choice(alphabet) for _ in range(query_len))
    query_chars = list(dict.fromkeys(query))
    n_examples = rng.randint(max(4, len(query_chars)), 6)
    examples = []
    used_inputs = set()

    def make_input(required_chars: list[str]) -> str:
        length = rng.randint(max(3, len(required_chars)), 6)
        chars = required_chars[:]
        while len(chars) < length:
            chars.append(rng.choice(alphabet))
        rng.shuffle(chars)
        return "".join(chars)

    for ch in query_chars:
        for _ in range(100):
            inp = make_input([ch])
            if inp not in used_inputs:
                break
        used_inputs.add(inp)
        examples.append((inp, transform(inp)))
    while len(examples) < n_examples:
        inp = make_input([])
        if inp in used_inputs:
            continue
        used_inputs.add(inp)
        examples.append((inp, transform(inp)))
    rng.shuffle(examples)

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:\n"
        + "\n".join(f"{src} = {dst}" for src, dst in examples)
        + f"\nNow, determine the result for: {query}"
    )
    payload = {"kind": "symbol_substitution", "mapping": mapping}
    return SyntheticExample(
        row_id,
        prompt,
        transform(query),
        "equation_symbol_transformation",
        "symbol_substitution",
        payload,
    )


def gen_gravity(row_id: str, rng: random.Random) -> SyntheticExample:
    g = round(rng.uniform(3.0, 24.0), 2)
    n_examples = rng.randint(5, 8)
    times = []
    if rng.random() < 0.35:
        times.append(1.0)
    if rng.random() < 0.25 and len(times) < n_examples:
        times.append(2.0)
    while len(times) < n_examples + 1:
        t = round(rng.uniform(0.7, 5.8), 2)
        if t not in times:
            times.append(t)

    def distance(t: float) -> str:
        return fmt_float(0.5 * g * t * t)

    lines = []
    for t in times[:n_examples]:
        t_text = fmt_measure(t, rng)
        lines.append(f"For t = {t_text}s, distance = {distance(float(t_text))} m")
    query = times[-1]
    query_text = fmt_measure(query, rng)
    prompt = (
        "In Alice's Wonderland, the gravitational constant has been secretly changed. "
        "Here are some example observations:\n"
        + "\n".join(lines)
        + f"\nNow, determine the falling distance for t = {query_text}s given d = 0.5*g*t^2."
    )
    return SyntheticExample(row_id, prompt, distance(float(query_text)), "gravity_formula", "gravity_constant", {"g": g})


def roman(num: int) -> str:
    parts = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    out = []
    for value, token in parts:
        while num >= value:
            out.append(token)
            num -= value
    return "".join(out)


def gen_numeral(row_id: str, rng: random.Random) -> SyntheticExample:
    n_examples = rng.randint(4, 6)
    values = rng.sample(range(1, 400), n_examples + 1)
    examples = values[:n_examples]
    query = values[-1]
    prompt = (
        "In Alice's Wonderland, numbers are secretly converted into a different numeral system. "
        "Some examples are given below:\n"
        + "\n".join(f"{n} -> {roman(n)}" for n in examples)
        + f"\nNow, write the number {query} in the Wonderland numeral system."
    )
    return SyntheticExample(row_id, prompt, roman(query), "numeral_system", "roman_numeral", {"max_value": 399})


SUBJECTS = [
    "cat",
    "dog",
    "queen",
    "wizard",
    "dragon",
    "student",
    "mouse",
    "princess",
    "hatter",
    "bird",
]
VERBS = [
    "finds",
    "sees",
    "opens",
    "reads",
    "writes",
    "creates",
    "discovers",
    "imagines",
    "draws",
    "chases",
    "follows",
    "builds",
]
OBJECTS = [
    "book",
    "mirror",
    "castle",
    "garden",
    "secret",
    "door",
    "map",
    "potion",
    "valley",
    "forest",
    "mountain",
    "river",
]
ADJECTIVES = ["golden", "mysterious", "magical", "wise", "hidden", "bright", "silent"]
PREPOSITIONS = ["near", "inside", "under", "beside", "behind", "over"]


def random_sentence(rng: random.Random) -> str:
    template = rng.randrange(5)
    if template == 0:
        return f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)}"
    if template == 1:
        return f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(PREPOSITIONS)} {rng.choice(OBJECTS)}"
    if template == 2:
        return f"the {rng.choice(ADJECTIVES)} {rng.choice(SUBJECTS)} {rng.choice(VERBS)}"
    if template == 3:
        return f"the {rng.choice(SUBJECTS)} {rng.choice(VERBS)} the {rng.choice(ADJECTIVES)} {rng.choice(OBJECTS)}"
    return f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} the {rng.choice(ADJECTIVES)} {rng.choice(OBJECTS)}"


def sentence_with_word(word: str, rng: random.Random) -> str:
    if word == "the":
        return f"the {rng.choice(ADJECTIVES)} {rng.choice(SUBJECTS)} {rng.choice(VERBS)}"
    if word in SUBJECTS:
        return f"the {rng.choice(ADJECTIVES)} {word} {rng.choice(VERBS)}"
    if word in VERBS:
        return f"{rng.choice(SUBJECTS)} {word} {rng.choice(OBJECTS)}"
    if word in OBJECTS:
        return f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {word}"
    if word in ADJECTIVES:
        return f"the {word} {rng.choice(SUBJECTS)} {rng.choice(VERBS)}"
    if word in PREPOSITIONS:
        return f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {word} {rng.choice(OBJECTS)}"
    return random_sentence(rng)


def make_cipher(rng: random.Random) -> dict[str, str]:
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    shuffled = alphabet[:]
    while True:
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(alphabet, shuffled)):
            return dict(zip(alphabet, shuffled))


def encrypt(text: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(ch, ch) for ch in text)


def gen_text(row_id: str, rng: random.Random) -> SyntheticExample:
    mapping = make_cipher(rng)
    query = random_sentence(rng)
    query_words = list(dict.fromkeys(query.split()))

    examples = []
    seen = set()
    for word in query_words:
        for _ in range(100):
            sent = sentence_with_word(word, rng)
            if sent != query and sent not in seen:
                examples.append(sent)
                seen.add(sent)
                break
    while len(examples) < 5:
        sent = random_sentence(rng)
        if sent != query and sent not in seen:
            examples.append(sent)
            seen.add(sent)

    prompt = (
        "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:\n"
        + "\n".join(f"{encrypt(sent, mapping)} -> {sent}" for sent in examples)
        + f"\nNow, decrypt the following text: {encrypt(query, mapping)}"
    )
    return SyntheticExample(row_id, prompt, query, "text_decryption", "monoalphabetic_substitution", {"mapping": mapping})


def gen_unit(row_id: str, rng: random.Random) -> SyntheticExample:
    scale = round(rng.uniform(0.35, 1.85), 3)
    n_examples = rng.randint(3, 6)
    values = []
    while len(values) < n_examples + 1:
        x = round(rng.uniform(2.0, 80.0), 2)
        if x not in values:
            values.append(x)

    def convert(x: float) -> str:
        return fmt_float(scale * x)

    lines = []
    for x in values[:n_examples]:
        x_text = fmt_measure(x, rng)
        lines.append(f"{x_text} m becomes {convert(float(x_text))}")
    query = values[-1]
    query_text = fmt_measure(query, rng)
    prompt = (
        "In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:\n"
        + "\n".join(lines)
        + f"\nNow, convert the following measurement: {query_text} m"
    )
    return SyntheticExample(row_id, prompt, convert(float(query_text)), "unit_conversion", "linear_scale", {"scale": scale})


GENERATORS: dict[str, Callable[[str, random.Random], SyntheticExample]] = {
    "bit_manipulation": gen_bit,
    "equation_symbol_transformation": gen_equation,
    "gravity_formula": gen_gravity,
    "numeral_system": gen_numeral,
    "text_decryption": gen_text,
    "unit_conversion": gen_unit,
}


def family_counts_from_args(per_family: int, family_counts_arg: str, version: str) -> dict[str, int]:
    if family_counts_arg:
        counts = {family: 0 for family in FAMILIES}
        for item in family_counts_arg.split(","):
            if not item.strip():
                continue
            if "=" not in item:
                raise ValueError(f"Invalid --family-counts item: {item!r}")
            family, count_text = item.split("=", 1)
            family = family.strip()
            if family not in counts:
                raise ValueError(f"Unknown family in --family-counts: {family!r}")
            counts[family] = int(count_text)
        return counts
    if version == "synthetic_v3" and per_family == 1000:
        return WEAK_V3_FAMILY_COUNTS.copy()
    return {family: per_family for family in FAMILIES}


def generate_examples(family_counts: dict[str, int], seed: int, version: str) -> list[SyntheticExample]:
    rng = random.Random(seed)
    rows: list[SyntheticExample] = []
    seen_prompts: set[str] = set()
    for family in FAMILIES:
        generator = GENERATORS[family]
        made = 0
        attempts = 0
        target_count = family_counts.get(family, 0)
        while made < target_count:
            attempts += 1
            if attempts > max(target_count * 100, 100):
                raise RuntimeError(f"Too many duplicate prompts while generating {family}")
            row_id = f"{version}_{family[:3]}_{made:06d}"
            example = generator(row_id, rng)
            if example.prompt in seen_prompts:
                continue
            if classify_prompt(example.prompt) != family:
                raise ValueError(f"Classifier mismatch for {row_id}: {classify_prompt(example.prompt)} != {family}")
            if not example.answer:
                raise ValueError(f"Empty answer for {row_id}")
            seen_prompts.add(example.prompt)
            rows.append(example)
            made += 1
    return rows


def parse_first(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not parse {label}")
    return match.group(1)


def apply_bit_rule(value: int, payload: dict) -> int:
    kind = payload["kind"]
    if kind == "xor":
        return value ^ payload["mask"]
    if kind == "rotl_xor":
        return rotl8(value, payload["shift"]) ^ payload["mask"]
    if kind == "reverse_xor":
        return bit_reverse8(value) ^ payload["mask"]
    if kind == "mask_mix":
        return ((value & payload["and_mask"]) | payload["or_mask"]) ^ payload["xor_mask"]
    raise ValueError(f"Unknown bit rule kind: {kind}")


def validate_synthetic_rows(rows: list[SyntheticExample]) -> None:
    errors = []

    for row in rows:
        try:
            if classify_prompt(row.prompt) != row.family:
                raise ValueError(f"classifier mismatch: {classify_prompt(row.prompt)} != {row.family}")
            if not row.answer:
                raise ValueError("empty answer")

            if row.family == "equation_symbol_transformation":
                query = row.prompt.split("Now, determine the result for: ", 1)[1]
                body = row.prompt.split("Below are a few examples:\n", 1)[1].split("\nNow, determine", 1)[0]
                if row.rule_payload["kind"] == "numeric_operator_rules":
                    op_rules = row.rule_payload["op_rules"]
                    visible_ops = set()
                    for line in body.splitlines():
                        src = line.split(" = ", 1)[0]
                        for op in op_rules:
                            if op in src:
                                visible_ops.add(op)
                    query_op = next((op for op in op_rules if op in query), None)
                    if query_op is None:
                        raise ValueError("could not identify query operator")
                    if query_op not in visible_ops:
                        raise ValueError(f"query operator missing from examples: {query_op!r}")
                    left, right = query.split(query_op, 1)
                    expected = numeric_equation_rule(op_rules[query_op], left, right)
                else:
                    visible_src_chars = set()
                    for line in body.splitlines():
                        visible_src_chars.update(line.split(" = ", 1)[0])
                    missing = set(query) - visible_src_chars
                    if missing:
                        raise ValueError(f"query chars missing from examples: {''.join(sorted(missing))!r}")
                    mapping = row.rule_payload["mapping"]
                    expected = "".join(mapping[ch] for ch in query)
                if row.answer != expected:
                    raise ValueError(f"answer mismatch: {row.answer!r} != {expected!r}")

            elif row.family == "text_decryption":
                encrypted_query = row.prompt.split("Now, decrypt the following text: ", 1)[1]
                body = row.prompt.split("Here are some examples:\n", 1)[1].split("\nNow, decrypt", 1)[0]
                visible_encrypted_chars = set()
                for line in body.splitlines():
                    visible_encrypted_chars.update(ch for ch in line.split(" -> ", 1)[0] if ch.isalpha())
                missing = set(ch for ch in encrypted_query if ch.isalpha()) - visible_encrypted_chars
                if missing:
                    raise ValueError(f"encrypted query chars missing from examples: {''.join(sorted(missing))!r}")
                if encrypt(row.answer, row.rule_payload["mapping"]) != encrypted_query:
                    raise ValueError("cipher answer mismatch")

            elif row.family == "gravity_formula":
                g = row.rule_payload["g"]
                for t_text, d_text in re.findall(
                    r"For t = ([0-9]+(?:\.[0-9]+)?)s, distance = ([0-9]+(?:\.[0-9]+)?) m",
                    row.prompt,
                ):
                    expected = fmt_float(0.5 * g * float(t_text) * float(t_text))
                    if d_text != expected:
                        raise ValueError(f"gravity example mismatch: {d_text} != {expected}")
                query_text = parse_first(
                    r"falling distance for t = ([0-9]+(?:\.[0-9]+)?)s",
                    row.prompt,
                    "gravity query",
                )
                expected = fmt_float(0.5 * g * float(query_text) * float(query_text))
                if row.answer != expected:
                    raise ValueError(f"gravity answer mismatch: {row.answer} != {expected}")

            elif row.family == "unit_conversion":
                scale = row.rule_payload["scale"]
                for x_text, y_text in re.findall(
                    r"([0-9]+(?:\.[0-9]+)?) m becomes ([0-9]+(?:\.[0-9]+)?)",
                    row.prompt,
                ):
                    expected = fmt_float(scale * float(x_text))
                    if y_text != expected:
                        raise ValueError(f"unit example mismatch: {y_text} != {expected}")
                query_text = parse_first(
                    r"measurement: ([0-9]+(?:\.[0-9]+)?) m",
                    row.prompt,
                    "unit query",
                )
                expected = fmt_float(scale * float(query_text))
                if row.answer != expected:
                    raise ValueError(f"unit answer mismatch: {row.answer} != {expected}")

            elif row.family == "numeral_system":
                query_text = parse_first(r"write the number ([0-9]+)", row.prompt, "numeral query")
                expected = roman(int(query_text))
                if row.answer != expected:
                    raise ValueError(f"roman answer mismatch: {row.answer} != {expected}")

            elif row.family == "bit_manipulation":
                example_values = []
                example_outputs = []
                body = row.prompt.split("Here are some examples of input -> output:\n", 1)[1].split(
                    "\n\nNow, determine", 1
                )[0]
                for line in body.splitlines():
                    src, dst = line.split(" -> ", 1)
                    example_values.append(int(src, 2))
                    example_outputs.append(int(dst, 2))
                signature = tuple(example_outputs)
                matches = iter_bit_candidate_signatures(example_values).get(signature, [])
                if len(matches) != 1 or matches[0] != row.rule_name:
                    raise ValueError(f"bit rule not uniquely identifiable: {matches[:5]}")
                query_text = parse_first(r"determine the output for: ([01]{8})", row.prompt, "bit query")
                expected = to_bin8(apply_bit_rule(int(query_text, 2), row.rule_payload))
                if row.answer != expected:
                    raise ValueError(f"bit answer mismatch: {row.answer} != {expected}")
        except Exception as exc:
            errors.append(f"{row.row_id} [{row.family}]: {exc}")

    if errors:
        preview = "\n".join(errors[:20])
        suffix = f"\n... {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"Synthetic validation failed with {len(errors)} errors:\n{preview}{suffix}")


def interleave_by_family(rows: list[SyntheticExample]) -> list[SyntheticExample]:
    by_family = {family: [] for family in FAMILIES}
    for row in rows:
        by_family[row.family].append(row)

    interleaved = []
    max_len = max(len(group) for group in by_family.values())
    for i in range(max_len):
        for family in FAMILIES:
            group = by_family[family]
            if i < len(group):
                interleaved.append(group[i])
    return interleaved


def load_original_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row_id = row.get(" id") or row.get("id") or ""
            prompt = row["prompt"]
            rows.append(
                {
                    " id": row_id,
                    "prompt": prompt,
                    "answer": row["answer"],
                    "family": classify_prompt(prompt),
                    "rule_name": "official_train",
                    "source": "official_train",
                }
            )
        return rows


def write_synthetic_csv(path: Path, rows: list[SyntheticExample], version: str) -> None:
    fieldnames = [" id", "prompt", "answer", "family", "rule_name", "source"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    " id": row.row_id,
                    "prompt": row.prompt,
                    "answer": row.answer,
                    "family": row.family,
                    "rule_name": row.rule_name,
                    "source": version,
                }
            )


def write_metadata(path: Path, rows: list[SyntheticExample]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "id": row.row_id,
                        "family": row.family,
                        "rule_name": row.rule_name,
                        "rule_payload": row.rule_payload,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )


def write_combined_csv(
    path: Path,
    original_rows: list[dict[str, str]],
    synthetic_rows: list[SyntheticExample],
    version: str,
) -> None:
    fieldnames = [" id", "prompt", "answer", "family", "rule_name", "source"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(original_rows)
        for row in synthetic_rows:
            writer.writerow(
                {
                    " id": row.row_id,
                    "prompt": row.prompt,
                    "answer": row.answer,
                    "family": row.family,
                    "rule_name": row.rule_name,
                    "source": version,
                }
            )


def write_report(
    path: Path,
    original_rows: list[dict[str, str]],
    synthetic_rows: list[SyntheticExample],
    seed: int,
    family_counts: dict[str, int],
    version: str,
) -> None:
    report = {
        "version": version,
        "seed": seed,
        "family_counts": family_counts,
        "original_rows": len(original_rows),
        "synthetic_rows": len(synthetic_rows),
        "combined_rows": len(original_rows) + len(synthetic_rows),
        "synthetic_family_counts": dict(Counter(row.family for row in synthetic_rows)),
        "original_family_counts": dict(Counter(row["family"] for row in original_rows)),
        "rules": {
            "bit_manipulation": "uniquely identifiable 8-bit XOR/rotation/bit-reversal rules",
            "equation_symbol_transformation": "mixed character substitution and numeric operator transformation rules",
            "gravity_formula": "d = 0.5 * g * t^2 with hidden g, more examples, and 2-decimal answers",
            "numeral_system": "Roman numerals",
            "text_decryption": "monoalphabetic substitution cipher over lowercase English sentences",
            "unit_conversion": "hidden linear scale factor with 2-decimal answers",
        },
        "answer_style": "short answer only; train_sft.py wraps as The final answer is \\boxed{answer}.",
    }
    path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--per-family", type=int, default=1000)
    parser.add_argument(
        "--family-counts",
        default="",
        help="Comma-separated overrides like bit_manipulation=3000,gravity_formula=3000.",
    )
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--version", default=DEFAULT_SYNTHETIC_VERSION)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    family_counts = family_counts_from_args(args.per_family, args.family_counts, args.version)
    synthetic_rows = interleave_by_family(
        generate_examples(family_counts=family_counts, seed=args.seed, version=args.version)
    )
    validate_synthetic_rows(synthetic_rows)
    original_rows = load_original_rows(args.train_csv)

    synthetic_csv = args.out_dir / f"{args.version}.csv"
    metadata_jsonl = args.out_dir / f"{args.version}_metadata.jsonl"
    combined_csv = args.out_dir / f"train_plus_{args.version}.csv"
    report_json = args.out_dir / f"{args.version}_report.json"

    write_synthetic_csv(synthetic_csv, synthetic_rows, args.version)
    write_metadata(metadata_jsonl, synthetic_rows)
    write_combined_csv(combined_csv, original_rows, synthetic_rows, args.version)
    write_report(report_json, original_rows, synthetic_rows, args.seed, family_counts, args.version)

    print(f"synthetic_rows={len(synthetic_rows)}")
    print(f"combined_rows={len(original_rows) + len(synthetic_rows)}")
    print(f"synthetic_csv={synthetic_csv}")
    print(f"metadata_jsonl={metadata_jsonl}")
    print(f"combined_csv={combined_csv}")
    print(f"report_json={report_json}")
    print("synthetic_family_counts:")
    for family, count in sorted(Counter(row.family for row in synthetic_rows).items()):
        print(f"  {family}: {count}")


if __name__ == "__main__":
    main()
