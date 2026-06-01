"""Quick test: verify generate() speed after transformers downgrade."""
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemotron_cache_compat import patch_nemotron_h_cache_compat
print(f"transformers version: {__import__('transformers').__version__}")

MODEL_PATH = os.environ.get("MODEL_PATH", "/data2/yaoxuran/models")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.bfloat16,
)
patch_nemotron_h_cache_compat(model)
print("Model loaded.")

# Test prompt
SYSTEM_PROMPT = "You are a precise reasoning model. Infer the hidden rule from the examples and answer the final query. Always put only the final answer inside \\boxed{}."
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.\n\nHere are some examples of input -> output:\n01010001 -> 11011101\n00001001 -> 01101101\n\nWhat is the output for: 11110000"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
enc = {k: v.to(model.device) for k, v in enc.items()}
input_len = enc["input_ids"].shape[1]
print(f"Input length: {input_len} tokens")

model.eval()

# Test 1: No cache
print("\n--- Test 1: generate use_cache=False ---")
t0 = time.time()
with torch.no_grad():
    out1 = model.generate(**enc, max_new_tokens=64, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id, use_cache=False)
t1 = time.time()
decoded1 = tokenizer.decode(out1[0][input_len:], skip_special_tokens=True)
time_no_cache = t1 - t0
print(f"  Time: {time_no_cache:.1f}s | {(out1.shape[1]-input_len)/(t1-t0):.1f} tok/s")
print(f"  Output: {decoded1[:200]}")

# Test 2: With cache (default)
print("\n--- Test 2: generate use_cache=True ---")
t0 = time.time()
with torch.no_grad():
    out2 = model.generate(**enc, max_new_tokens=64, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id, use_cache=True)
t1 = time.time()
decoded2 = tokenizer.decode(out2[0][input_len:], skip_special_tokens=True)
time_with_cache = t1 - t0
print(f"  Time: {time_with_cache:.1f}s | {(out2.shape[1]-input_len)/(t1-t0):.1f} tok/s")
print(f"  Output: {decoded2[:200]}")

print(f"\n=== Speedup: {time_no_cache/time_with_cache:.1f}x with cache ===")
