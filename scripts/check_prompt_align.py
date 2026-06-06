#!/usr/bin/env python3
"""No-GPU smoke: verify the local prompt matches the official Kaggle harness.

Loads only the tokenizer (CPU) and asserts that train_sft.format_inference_prompt
renders byte-for-byte the same string the official harness feeds at submission:
no system prompt, user = prompt + official boxed instruction, enable_thinking=True.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OFFICIAL_PROMPT_ALIGN", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # tokenizer only

from transformers import AutoTokenizer  # noqa: E402
import train_sft  # noqa: E402


def official_reference(tok, prompt):
    """Build the prompt exactly the way the Kaggle harness does."""
    user = prompt + train_sft.OFFICIAL_BOXED_INSTRUCTION
    messages = [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def main():
    print("OFFICIAL_PROMPT_ALIGN =", train_sft.OFFICIAL_PROMPT_ALIGN)
    tok = AutoTokenizer.from_pretrained(train_sft.MODEL_PATH, trust_remote_code=True)

    sample = "Examples:\n'ab' -> 'cd'\n'ef' -> 'gh'\nQuery: 'ij' -> ?"
    ours = train_sft.format_inference_prompt(tok, sample)
    ref = official_reference(tok, sample)

    print("\n----- rendered prompt (ours) -----")
    print(ours)
    print("----------------------------------\n")

    assert train_sft.SYSTEM_PROMPT not in ours, "FAIL: legacy system prompt leaked in"
    assert train_sft.OFFICIAL_BOXED_INSTRUCTION.strip() in ours, \
        "FAIL: official boxed instruction missing"
    assert ours == ref, "FAIL: local prompt is NOT byte-identical to the official harness"
    print("OK: local prompt is byte-identical to the official Kaggle harness prompt.")


if __name__ == "__main__":
    main()
