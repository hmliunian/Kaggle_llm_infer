"""Build ground-truth chain-of-thought (CoT) reasoning traces for synthetic tasks.

The synthetic generator stores the exact rule (`rule_payload`) for every example,
so we can emit a *teacher-quality* reasoning trace deterministically — no model
sampling, no rejection filtering. Each builder:

  1. Parses the visible examples + query out of the prompt.
  2. Reconstructs the reasoning a solver *should* follow: infer the rule from the
     examples (grounded in concrete example values), then apply it to the query.
  3. Returns the CoT text only (the caller wraps it as
     `<think>{cot}</think>\nThe final answer is \\boxed{{answer}}.`).

Design rule: the CoT must *derive* the rule from the shown examples rather than
just assert the hidden payload, so the model learns to generalize from examples
instead of memorizing. We use the payload only to know the derivation is correct
and to self-check that applying the CoT reproduces the gold answer.

Currently implemented: bit_manipulation, equation_symbol_transformation.
"""

import re

from generate_synthetic_data import (
    bit_reverse8,
    fmt_float,
    iter_bit_candidate_signatures,
    numeric_equation_rule,
    roman,
    rotl8,
)
from official_metric import verify as official_verify


class CotError(Exception):
    pass


def _b(value: int) -> str:
    return format(value & 0xFF, "08b")


def _payload_from_bit_rule_name(name: str) -> dict:
    """Parse a candidate rule name (from iter_bit_candidate_signatures) into a payload."""
    m = re.fullmatch(r"xor_([0-9a-f]{2})", name)
    if m:
        return {"kind": "xor", "mask": int(m.group(1), 16)}
    m = re.fullmatch(r"reverse_xor_([0-9a-f]{2})", name)
    if m:
        return {"kind": "reverse_xor", "mask": int(m.group(1), 16)}
    m = re.fullmatch(r"rotl([1-7])_xor_([0-9a-f]{2})", name)
    if m:
        return {"kind": "rotl_xor", "shift": int(m.group(1)), "mask": int(m.group(2), 16)}
    raise CotError(f"cannot parse bit rule name: {name}")


# ─── bit_manipulation ─────────────────────────────────────────────────────────
def _parse_bit(prompt: str):
    pairs = re.findall(r"([01]{8})\s*->\s*([01]{8})", prompt)
    qm = re.search(r"determine the output for:\s*([01]{8})", prompt)
    if not pairs or not qm:
        raise CotError("could not parse bit prompt")
    examples = [(int(a, 2), int(b, 2)) for a, b in pairs]
    return examples, int(qm.group(1), 2)


def _infer_bit_payload(examples) -> dict:
    """Brute-force the rule from the shown examples; require a unique match."""
    inputs = [x for x, _ in examples]
    outputs = tuple(y for _, y in examples)
    matches = iter_bit_candidate_signatures(inputs).get(outputs, [])
    if len(matches) != 1:
        raise CotError(f"bit rule not uniquely identifiable from examples ({len(matches)} candidates)")
    return _payload_from_bit_rule_name(matches[0])


def build_bit_cot(prompt: str, answer: str, payload: dict = None) -> str:
    examples, query = _parse_bit(prompt)
    if payload is None:
        payload = _infer_bit_payload(examples)
    kind = payload["kind"]
    (e0_in, e0_out) = examples[0]
    lines = []

    if kind == "xor":
        mask = payload["mask"]
        lines.append(
            f"Check whether each output is the input XOR a fixed 8-bit mask. "
            f"From the first example, {_b(e0_in)} XOR {_b(e0_out)} = {_b(e0_in ^ e0_out)}, "
            f"so the candidate mask is {_b(mask)}."
        )
        if len(examples) > 1:
            e1_in, e1_out = examples[1]
            lines.append(
                f"Verify on the next example: {_b(e1_in)} XOR {_b(mask)} = {_b(e1_in ^ mask)}, "
                f"which matches the given output {_b(e1_out)}. The rule is: XOR with {_b(mask)}."
            )
        result = query ^ mask
        lines.append(
            f"Apply to the query: {_b(query)} XOR {_b(mask)} = {_b(result)}."
        )

    elif kind == "rotl_xor":
        shift, mask = payload["shift"], payload["mask"]
        lines.append(
            f"A constant XOR does not fit (input XOR output is not the same across examples), "
            f"so try rotating the input left before XOR-ing a fixed mask."
        )
        rot0 = rotl8(e0_in, shift)
        lines.append(
            f"Rotate the first input left by {shift}: {_b(e0_in)} -> {_b(rot0)}. "
            f"Then {_b(rot0)} XOR {_b(e0_out)} = {_b(rot0 ^ e0_out)}, giving mask {_b(mask)}."
        )
        if len(examples) > 1:
            e1_in, e1_out = examples[1]
            r1 = rotl8(e1_in, shift)
            lines.append(
                f"Verify: rotate {_b(e1_in)} left by {shift} -> {_b(r1)}, "
                f"then XOR {_b(mask)} = {_b(r1 ^ mask)}, matching {_b(e1_out)}. "
                f"The rule is: rotate left by {shift}, then XOR {_b(mask)}."
            )
        rotq = rotl8(query, shift)
        result = rotq ^ mask
        lines.append(
            f"Apply to the query: rotate {_b(query)} left by {shift} -> {_b(rotq)}, "
            f"then XOR {_b(mask)} = {_b(result)}."
        )

    elif kind == "reverse_xor":
        mask = payload["mask"]
        lines.append(
            "A constant XOR does not fit, so try reversing the bit order before XOR-ing a fixed mask."
        )
        rev0 = bit_reverse8(e0_in)
        lines.append(
            f"Reverse the first input: {_b(e0_in)} -> {_b(rev0)}. "
            f"Then {_b(rev0)} XOR {_b(e0_out)} = {_b(rev0 ^ e0_out)}, giving mask {_b(mask)}."
        )
        if len(examples) > 1:
            e1_in, e1_out = examples[1]
            r1 = bit_reverse8(e1_in)
            lines.append(
                f"Verify: reverse {_b(e1_in)} -> {_b(r1)}, then XOR {_b(mask)} = {_b(r1 ^ mask)}, "
                f"matching {_b(e1_out)}. The rule is: reverse the bits, then XOR {_b(mask)}."
            )
        revq = bit_reverse8(query)
        result = revq ^ mask
        lines.append(
            f"Apply to the query: reverse {_b(query)} -> {_b(revq)}, "
            f"then XOR {_b(mask)} = {_b(result)}."
        )
    else:
        raise CotError(f"unsupported bit kind: {kind}")

    derived = _b(result)
    if derived != answer:
        raise CotError(f"bit CoT self-check failed: {derived} != {answer}")
    return " ".join(lines)


# ─── equation_symbol_transformation ─────────────────────────────────────────────
def _parse_equation(prompt: str):
    body = prompt.split("Below are a few examples:\n", 1)[1].split("\nNow, determine", 1)[0]
    query = prompt.split("Now, determine the result for: ", 1)[1].strip()
    examples = []
    for line in body.splitlines():
        if " = " in line:
            src, dst = line.split(" = ", 1)
            examples.append((src.strip(), dst.strip()))
    if not examples or not query:
        raise CotError("could not parse equation prompt")
    return examples, query


_NUMERIC_RULE_DESC = {
    "absdiff": "take the absolute digit-wise difference: |a-c| then |b-d| (for ab OP cd)",
    "sum_mod": "add digit-wise mod 10: (a+c)%10 then (b+d)%10",
    "prod_mod": "multiply digit-wise mod 10: (a*c)%10 then (b*d)%10",
    "cross_sum": "cross-add mod 10: (a+d)%10 then (b+c)%10",
    "swap_concat": "swap the two operands and concatenate: cd then ab",
    "outer_inner_abs": "outer/inner absolute difference: |a-d| then |b-c|",
}


def _numeric_show(rule_name: str, left: str, right: str) -> str:
    """Return the explicit two-digit arithmetic for one operator application."""
    a, b = int(left[0]), int(left[1])
    c, d = int(right[0]), int(right[1])
    if rule_name == "absdiff":
        return f"|{a}-{c}|={abs(a-c)} and |{b}-{d}|={abs(b-d)}"
    if rule_name == "sum_mod":
        return f"({a}+{c})%10={(a+c)%10} and ({b}+{d})%10={(b+d)%10}"
    if rule_name == "prod_mod":
        return f"({a}*{c})%10={(a*c)%10} and ({b}*{d})%10={(b*d)%10}"
    if rule_name == "cross_sum":
        return f"({a}+{d})%10={(a+d)%10} and ({b}+{c})%10={(b+c)%10}"
    if rule_name == "swap_concat":
        return f"swap {left} and {right} to get {right}{left}"
    if rule_name == "outer_inner_abs":
        return f"|{a}-{d}|={abs(a-d)} and |{b}-{c}|={abs(b-c)}"
    raise CotError(f"no numeric show for {rule_name}")


_NUMERIC_RULE_NAMES = ["absdiff", "sum_mod", "prod_mod", "cross_sum", "swap_concat", "outer_inner_abs"]


def _infer_equation_payload(examples, query) -> dict:
    """Infer the rule from the shown examples (no payload needed)."""
    # Numeric operator form: "dd OP dd = ddd". Detect via the query shape.
    qm = re.fullmatch(r"(\d{2})(\D)(\d{2})", query)
    if qm:
        query_op = qm.group(2)
        left, right = qm.group(1), qm.group(3)
        # Gather examples using the query operator and find the consistent rule.
        same_op = []
        for src, dst in examples:
            em = re.fullmatch(r"(\d{2})(\D)(\d{2})", src)
            if em and em.group(2) == query_op:
                same_op.append((em.group(1), em.group(3), dst))
        if not same_op:
            raise CotError(f"query operator {query_op!r} not shown in examples")
        consistent = [
            name for name in _NUMERIC_RULE_NAMES
            if all(numeric_equation_rule(name, l, r) == d for l, r, d in same_op)
        ]
        if not consistent:
            raise CotError("no numeric rule consistent with examples for query operator")
        # The task is only solvable if every rule that fits the examples agrees on the query.
        query_results = {numeric_equation_rule(name, left, right) for name in consistent}
        if len(query_results) > 1:
            raise CotError(
                f"ambiguous numeric operator {query_op!r}: examples fit {consistent} "
                f"which disagree on the query"
            )
        return {"kind": "numeric_operator_rules", "op_rules": {query_op: consistent[0]}}

    # Symbol substitution: image of a char = dst char at the same position.
    mapping = {}
    for src, dst in examples:
        if len(src) != len(dst):
            continue
        for s, d in zip(src, dst):
            mapping.setdefault(s, d)
    missing = [ch for ch in query if ch not in mapping]
    if missing:
        raise CotError(f"query symbols missing from examples: {''.join(missing)}")
    return {"kind": "symbol_substitution", "mapping": mapping}


def build_equation_cot(prompt: str, answer: str, payload: dict = None) -> str:
    examples, query = _parse_equation(prompt)
    if payload is None:
        payload = _infer_equation_payload(examples, query)
    kind = payload["kind"]
    lines = []

    if kind == "symbol_substitution":
        mapping = payload["mapping"]
        # For each query char, ground its image in an example that shows it.
        query_chars = list(dict.fromkeys(query))
        lines.append(
            "Each symbol is replaced one-for-one by a fixed symbol (a substitution cipher). "
            "Read off the replacement for every symbol in the query from the examples:"
        )
        for ch in query_chars:
            shown = None
            for src, dst in examples:
                if ch in src:
                    idx = src.index(ch)
                    if idx < len(dst):
                        shown = (src, dst, idx, dst[idx])
                        break
            if shown is None or shown[3] != mapping[ch]:
                raise CotError(f"could not ground mapping for {ch!r}")
            src, dst, idx, img = shown
            lines.append(
                f"'{ch}' appears in '{src}' = '{dst}' at position {idx + 1}, so '{ch}' -> '{img}'."
            )
        result = "".join(mapping[ch] for ch in query)
        lines.append(
            f"Apply this to the query '{query}' symbol by symbol to get '{result}'."
        )

    elif kind == "numeric_operator_rules":
        op_rules = payload["op_rules"]
        query_op = next((op for op in op_rules if op in query), None)
        if query_op is None:
            raise CotError("could not find query operator")
        rule_name = op_rules[query_op]
        left, right = query.split(query_op, 1)
        a, b = int(left[0]), int(left[1])
        c, d = int(right[0]), int(right[1])
        # Ground the operator's rule in an example that uses the same operator.
        ex = next(((s, dd) for s, dd in examples if query_op in s and s != query), None)
        lines.append(
            f"Each operator symbol encodes its own two-digit rule. The query uses '{query_op}'."
        )
        if ex is not None:
            el, er = ex[0].split(query_op, 1)
            lines.append(
                f"From the example '{ex[0]} = {ex[1]}', the rule for '{query_op}' is to "
                f"{_NUMERIC_RULE_DESC[rule_name]}: {_numeric_show(rule_name, el, er)}, "
                f"which gives {ex[1]} as shown."
            )
        else:
            lines.append(f"The rule for '{query_op}' is to {_NUMERIC_RULE_DESC[rule_name]}.")
        result = numeric_equation_rule(rule_name, left, right)
        lines.append(
            f"Apply it to '{query}' with a={a}, b={b}, c={c}, d={d}: {_numeric_show(rule_name, left, right)}, "
            f"so the result is '{result}'."
        )
    else:
        raise CotError(f"unsupported equation kind: {kind}")

    if result != answer:
        raise CotError(f"equation CoT self-check failed: {result!r} != {answer!r}")
    return " ".join(lines)


# ─── text_decryption ────────────────────────────────────────────────────────
def _parse_text(prompt: str):
    body = prompt.split("Here are some examples:\n", 1)[1].split("\nNow, decrypt", 1)[0]
    query = prompt.split("Now, decrypt the following text: ", 1)[1].strip()
    pairs = []
    for line in body.splitlines():
        if " -> " in line:
            enc, plain = line.split(" -> ", 1)
            pairs.append((enc.strip(), plain.strip()))
    if not pairs or not query:
        raise CotError("could not parse text prompt")
    return pairs, query


def build_text_cot(prompt: str, answer: str, payload: dict = None) -> str:
    pairs, query = _parse_text(prompt)
    # Infer the decryption map (encrypted letter -> plain letter) by aligning
    # each example position-by-position.
    dec = {}
    for enc, plain in pairs:
        if len(enc) != len(plain):
            continue
        for e, p in zip(enc, plain):
            if e.isalpha() and p.isalpha():
                dec.setdefault(e.lower(), p.lower())
    query_letters = list(dict.fromkeys(ch for ch in query if ch.isalpha()))
    missing = [ch for ch in query_letters if ch not in dec]
    if missing:
        raise CotError(f"cipher letters missing from examples: {''.join(missing)}")
    lines = ["This is a letter-substitution cipher. Recover each ciphertext letter from the examples:"]
    for ch in query_letters:
        shown = next((f"'{enc}' -> '{plain}'" for enc, plain in pairs if ch in enc.lower()), None)
        lines.append(f"'{ch}' decrypts to '{dec[ch]}' (seen in {shown}).")
    result = "".join(dec.get(ch, dec.get(ch.lower(), ch)) if ch.isalpha() else ch for ch in query)
    lines.append(f"Decrypting '{query}' letter by letter gives '{result}'.")
    if result != answer:
        raise CotError(f"text CoT self-check failed: {result!r} != {answer!r}")
    return " ".join(lines)


# ─── numeral_system ─────────────────────────────────────────────────────────
_ROMAN_PARTS = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
    (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
    (5, "V"), (4, "IV"), (1, "I"),
]


def _parse_numeral(prompt: str):
    body = prompt.split("Some examples are given below:\n", 1)[1].split("\nNow, write", 1)[0]
    qm = re.search(r"write the number (\d+)", prompt)
    pairs = []
    for line in body.splitlines():
        m = re.match(r"\s*(\d+)\s*->\s*(\S+)", line)
        if m:
            pairs.append((int(m.group(1)), m.group(2)))
    if not pairs or not qm:
        raise CotError("could not parse numeral prompt")
    return pairs, int(qm.group(1))


def build_numeral_cot(prompt: str, answer: str, payload: dict = None) -> str:
    pairs, query = _parse_numeral(prompt)
    # Confirm the examples are Roman numerals.
    for n, tok in pairs:
        if roman(n) != tok:
            raise CotError("numeral examples are not standard Roman")
    breakdown = []
    remaining = query
    for value, token in _ROMAN_PARTS:
        while remaining >= value:
            breakdown.append(f"{value}->{token}")
            remaining -= value
    result = roman(query)
    cot = (
        "The examples are Roman numerals (I=1, V=5, X=10, L=50, C=100, D=500, M=1000), "
        f"with subtractive pairs like IV=4 and IX=9. Convert {query} greedily: "
        + ", ".join(breakdown)
        + f", concatenating to {result}."
    )
    if result != answer:
        raise CotError(f"numeral CoT self-check failed: {result!r} != {answer!r}")
    return cot


# ─── gravity_formula ────────────────────────────────────────────────────────
def build_gravity_cot(prompt: str, answer: str, payload: dict = None) -> str:
    pairs = re.findall(
        r"For t = ([0-9]+(?:\.[0-9]+)?)s, distance = ([0-9]+(?:\.[0-9]+)?) m", prompt
    )
    qm = re.search(r"falling distance for t = ([0-9]+(?:\.[0-9]+)?)s", prompt)
    if not pairs or not qm:
        raise CotError("could not parse gravity prompt")
    # Recover g from each example (g = 2d/t^2) and use the rounded consensus.
    cands = [round(2 * float(d) / (float(t) ** 2), 2) for t, d in pairs]
    g = max(set(cands), key=cands.count)
    anchor_t, anchor_d = pairs[0]
    query_t = float(qm.group(1))
    computed = fmt_float(0.5 * g * query_t * query_t)
    cot = (
        f"The rule is d = 0.5*g*t^2 with a hidden g. From the example t={anchor_t}s, d={anchor_d} m: "
        f"g = 2*{anchor_d}/{anchor_t}^2 = {fmt_float(g)}. "
        f"Apply to t={qm.group(1)}s: d = 0.5*{fmt_float(g)}*{qm.group(1)}^2 = {computed}."
    )
    if not official_verify(answer, computed):
        raise CotError(f"gravity CoT self-check failed: {computed} vs gold {answer}")
    return cot


# ─── unit_conversion ──────────────────────────────────────────────────────────
def build_unit_cot(prompt: str, answer: str, payload: dict = None) -> str:
    pairs = re.findall(r"([0-9]+(?:\.[0-9]+)?) m becomes ([0-9]+(?:\.[0-9]+)?)", prompt)
    qm = re.search(r"convert the following measurement: ([0-9]+(?:\.[0-9]+)?) m", prompt)
    if not pairs or not qm:
        raise CotError("could not parse unit prompt")
    cands = [round(float(y) / float(x), 3) for x, y in pairs if float(x) != 0]
    scale = max(set(cands), key=cands.count)
    ax, ay = pairs[0]
    query_x = float(qm.group(1))
    computed = fmt_float(scale * query_x)
    cot = (
        f"Each measurement is multiplied by a fixed factor. From '{ax} m becomes {ay}': "
        f"factor = {ay}/{ax} = {scale}. "
        f"Apply to {qm.group(1)} m: {qm.group(1)}*{scale} = {computed}."
    )
    if not official_verify(answer, computed):
        raise CotError(f"unit CoT self-check failed: {computed} vs gold {answer}")
    return cot


BUILDERS = {
    "bit_manipulation": build_bit_cot,
    "equation_symbol_transformation": build_equation_cot,
    "text_decryption": build_text_cot,
    "numeral_system": build_numeral_cot,
    "gravity_formula": build_gravity_cot,
    "unit_conversion": build_unit_cot,
}


def build_cot(family: str, prompt: str, answer: str, payload: dict) -> str:
    builder = BUILDERS.get(family)
    if builder is None:
        raise CotError(f"no CoT builder for family: {family}")
    return builder(prompt, answer, payload)
