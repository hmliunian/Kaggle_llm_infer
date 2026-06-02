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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_CSV = BASE_DIR / "data" / "train.csv"
DEFAULT_OUT_DIR = BASE_DIR / "data"

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
    kind = rng.choice(["xor", "not_xor", "rotl_xor", "rotr_xor", "reverse_xor", "mask_mix"])
    if kind == "xor":
        mask = rng.randrange(1, 256)
        return f"xor_{mask:02x}", lambda x, m=mask: x ^ m, {"kind": kind, "mask": mask}
    if kind == "not_xor":
        mask = rng.randrange(0, 256)
        return f"not_xor_{mask:02x}", lambda x, m=mask: ((~x) & 0xFF) ^ m, {"kind": kind, "mask": mask}
    if kind == "rotl_xor":
        shift = rng.randrange(1, 8)
        mask = rng.randrange(0, 256)
        return (
            f"rotl{shift}_xor_{mask:02x}",
            lambda x, s=shift, m=mask: rotl8(x, s) ^ m,
            {"kind": kind, "shift": shift, "mask": mask},
        )
    if kind == "rotr_xor":
        shift = rng.randrange(1, 8)
        mask = rng.randrange(0, 256)
        return (
            f"rotr{shift}_xor_{mask:02x}",
            lambda x, s=shift, m=mask: rotr8(x, s) ^ m,
            {"kind": kind, "shift": shift, "mask": mask},
        )
    if kind == "reverse_xor":
        mask = rng.randrange(0, 256)
        return (
            f"reverse_xor_{mask:02x}",
            lambda x, m=mask: bit_reverse8(x) ^ m,
            {"kind": kind, "mask": mask},
        )

    and_mask = rng.randrange(0, 256)
    or_mask = rng.randrange(0, 256)
    xor_mask = rng.randrange(0, 256)
    return (
        f"mask_mix_{and_mask:02x}_{or_mask:02x}_{xor_mask:02x}",
        lambda x, a=and_mask, o=or_mask, m=xor_mask: ((x & a) | o) ^ m,
        {"kind": kind, "and_mask": and_mask, "or_mask": or_mask, "xor_mask": xor_mask},
    )


def gen_bit(row_id: str, rng: random.Random) -> SyntheticExample:
    rule_name, rule, payload = make_bit_rule(rng)
    n_examples = rng.randint(6, 9)
    values = rng.sample(range(256), n_examples + 1)
    examples = values[:n_examples]
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


# Exclude braces because train_sft.py wraps answers as \boxed{answer}.
# Braces inside the answer make boxed-answer extraction ambiguous.
EQUATION_SYMBOLS = list("!@#$%^&*()[]<>?/\\|~`'\"-+_:;")


def gen_equation(row_id: str, rng: random.Random) -> SyntheticExample:
    alphabet = EQUATION_SYMBOLS[:]
    shuffled = alphabet[:]
    rng.shuffle(shuffled)
    mapping = dict(zip(alphabet, shuffled))

    def transform(text: str) -> str:
        return "".join(mapping[ch] for ch in text)

    query_len = rng.randint(3, 6)
    query = "".join(rng.choice(alphabet) for _ in range(query_len))
    n_examples = rng.randint(4, 6)
    examples = []
    for i in range(n_examples):
        length = rng.randint(3, 6)
        chars = [rng.choice(query)] if i < min(len(set(query)), n_examples) else []
        while len(chars) < length:
            chars.append(rng.choice(alphabet))
        rng.shuffle(chars)
        inp = "".join(chars)
        examples.append((inp, transform(inp)))

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
    n_examples = rng.randint(4, 6)
    times = []
    while len(times) < n_examples + 1:
        t = round(rng.uniform(0.7, 5.8), 2)
        if t not in times:
            times.append(t)

    def distance(t: float) -> str:
        return fmt_float(0.5 * g * t * t)

    lines = [f"For t = {fmt_measure(t, rng)}s, distance = {distance(t)} m" for t in times[:n_examples]]
    query = times[-1]
    prompt = (
        "In Alice's Wonderland, the gravitational constant has been secretly changed. "
        "Here are some example observations:\n"
        + "\n".join(lines)
        + f"\nNow, determine the falling distance for t = {fmt_measure(query, rng)}s given d = 0.5*g*t^2."
    )
    return SyntheticExample(row_id, prompt, distance(query), "gravity_formula", "gravity_constant", {"g": g})


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
    query_words = [w for w in query.split() if w != "the"]

    examples = []
    seen = set()
    for word in query_words[:5]:
        sent = sentence_with_word(word, rng)
        if sent != query and sent not in seen:
            examples.append(sent)
            seen.add(sent)
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

    lines = [f"{fmt_measure(x, rng)} m becomes {convert(x)}" for x in values[:n_examples]]
    query = values[-1]
    prompt = (
        "In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:\n"
        + "\n".join(lines)
        + f"\nNow, convert the following measurement: {fmt_measure(query, rng)} m"
    )
    return SyntheticExample(row_id, prompt, convert(query), "unit_conversion", "linear_scale", {"scale": scale})


GENERATORS: dict[str, Callable[[str, random.Random], SyntheticExample]] = {
    "bit_manipulation": gen_bit,
    "equation_symbol_transformation": gen_equation,
    "gravity_formula": gen_gravity,
    "numeral_system": gen_numeral,
    "text_decryption": gen_text,
    "unit_conversion": gen_unit,
}


def generate_examples(per_family: int, seed: int) -> list[SyntheticExample]:
    rng = random.Random(seed)
    rows: list[SyntheticExample] = []
    seen_prompts: set[str] = set()
    for family in FAMILIES:
        generator = GENERATORS[family]
        made = 0
        attempts = 0
        while made < per_family:
            attempts += 1
            if attempts > per_family * 50:
                raise RuntimeError(f"Too many duplicate prompts while generating {family}")
            row_id = f"synv1_{family[:3]}_{made:06d}"
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


def write_synthetic_csv(path: Path, rows: list[SyntheticExample]) -> None:
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
                    "source": "synthetic_v1",
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


def write_combined_csv(path: Path, original_rows: list[dict[str, str]], synthetic_rows: list[SyntheticExample]) -> None:
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
                    "source": "synthetic_v1",
                }
            )


def write_report(
    path: Path,
    original_rows: list[dict[str, str]],
    synthetic_rows: list[SyntheticExample],
    seed: int,
    per_family: int,
) -> None:
    report = {
        "version": "synthetic_v1",
        "seed": seed,
        "per_family": per_family,
        "original_rows": len(original_rows),
        "synthetic_rows": len(synthetic_rows),
        "combined_rows": len(original_rows) + len(synthetic_rows),
        "synthetic_family_counts": dict(Counter(row.family for row in synthetic_rows)),
        "original_family_counts": dict(Counter(row["family"] for row in original_rows)),
        "rules": {
            "bit_manipulation": "8-bit deterministic functions using XOR, NOT, rotations, bit reversal, and masks",
            "equation_symbol_transformation": "character-level symbol substitution",
            "gravity_formula": "d = 0.5 * g * t^2 with hidden g and 2-decimal answers",
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
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    synthetic_rows = interleave_by_family(generate_examples(per_family=args.per_family, seed=args.seed))
    original_rows = load_original_rows(args.train_csv)

    synthetic_csv = args.out_dir / "synthetic_v1.csv"
    metadata_jsonl = args.out_dir / "synthetic_v1_metadata.jsonl"
    combined_csv = args.out_dir / "train_plus_synthetic_v1.csv"
    report_json = args.out_dir / "synthetic_v1_report.json"

    write_synthetic_csv(synthetic_csv, synthetic_rows)
    write_metadata(metadata_jsonl, synthetic_rows)
    write_combined_csv(combined_csv, original_rows, synthetic_rows)
    write_report(report_json, original_rows, synthetic_rows, args.seed, args.per_family)

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
