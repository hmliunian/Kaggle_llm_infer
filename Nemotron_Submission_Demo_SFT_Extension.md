# NVIDIA Nemotron Submission Demo Extension Plan

This document is written for an AI coding agent such as Codex, Claude, Cursor, or a local software agent.

Goal: extend the official Kaggle notebook `NVIDIA Nemotron Submission Demo` into a working LoRA SFT training pipeline for the NVIDIA Nemotron Model Reasoning Challenge.

The official demo already shows how to:

```text
1. Load train.csv.
2. Download the official evaluation model.
3. Load Nemotron-3-Nano-30B-A3B in BF16.
4. Attach a LoRA adapter.
5. Save the adapter.
6. Package submission.zip.
```

The missing part is training. The official notebook leaves this placeholder:

```python
# YOUR CODE HERE
# --------------
# model.train()
# --------------
```

This document explains exactly what should be added there.

---

## 1. Key Competition Understanding

This is not a normal Kaggle CSV prediction competition.

The expected submission is:

```text
submission.zip
```

The zip should contain the trained LoRA adapter files, mainly:

```text
adapter_config.json
adapter_model.safetensors
```

The competition evaluator will load the official base model plus the submitted adapter, run hidden prompts, and score the generated answers.

The model should output the final answer in LaTeX boxed format:

```text
\boxed{ANSWER}
```

The assistant response should end with exactly one clear final boxed answer.

Recommended final response template:

```text
The final answer is \boxed{ANSWER}.
```

---

## 2. Official Demo Code Structure

The official demo uses the following core code:

```python
import polars as pl

train = pl.read_csv('/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv')
train.head()
```

Then it loads special NVIDIA dependencies:

```python
import site

cutlass_pkg_path = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"
site.addsitedir(cutlass_pkg_path)

import kagglehub
import mamba_ssm
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
```

The official model path is:

```python
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
```

The model is loaded as:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16
)
```

The official LoRA configuration is:

```python
LORA_RANK = 32

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

Then the adapter is attached:

```python
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Finally, the adapter is saved and packaged:

```python
model.save_pretrained(OUTPUT_DIR)

import subprocess
subprocess.run("zip -m submission.zip *", shell=True, check=True)
```

---

## 3. Important Implementation Constraints

Follow these constraints strictly.

### 3.1 Keep the official model path

Do not replace the model with another Nemotron checkpoint.

Use:

```python
"metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
```

### 3.2 Keep `trust_remote_code=True`

The model depends on custom implementation code.

Use:

```python
trust_remote_code=True
```

### 3.3 Keep the official LoRA target module pattern first

The official demo uses:

```python
target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$"
```

Do not blindly replace it with common Llama/Qwen module names such as:

```text
q_proj, k_proj, v_proj, o_proj, gate_proj
```

Only change target modules after inspecting the actual model structure.

### 3.4 Do not full-finetune

Only train LoRA adapter parameters.

Do not unfreeze the full base model.

### 3.5 Do not submit full base model weights

Only package the adapter.

### 3.6 Rank limit

The competition allows LoRA rank up to 32.

Recommended experiments:

```text
baseline: rank = 16
stronger run: rank = 32
```

The official demo uses rank 32 and reports about 880M trainable parameters, so rank 32 is not tiny. Use H200 for serious training.

---

## 4. Recommended First Working Version

Build a short-answer SFT baseline first.

Do not start with PPO or GRPO.

Do not train on unverified long chain-of-thought traces.

Use the raw `prompt` and raw `answer` from `train.csv`:

```text
User:
{prompt}

Assistant:
The final answer is \boxed{answer}.
```

This is the safest first version.

---

## 5. Why Avoid Long CoT Initially

A Kaggle discussion comment warned that some public/generated reasoning traces for this competition may be unreliable, especially for:

```text
bit manipulation
equation / symbol transformation
```

Potential risks:

```text
- generated Result lines may not match the official answer
- simple char_map reasoning may fail when input/output lengths differ
- heuristic traces may teach the model wrong reasoning
```

Therefore:

```text
First train short-answer SFT.
Only add CoT later if the trace is generated by a verified solver.
```

---

## 6. Data Preparation

Implement a function to convert `train.csv` into SFT examples.

### 6.1 Minimal chat template format

Use:

```python
SYSTEM_PROMPT = (
    "You are a precise reasoning model. "
    "Infer the hidden rule from the examples and answer the final query. "
    "Always put only the final answer inside \\boxed{}."
)
```

Each row becomes:

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": row["prompt"]},
    {"role": "assistant", "content": f"The final answer is \\boxed{{{row['answer']}}}."},
]
```

### 6.2 Tokenizer

The official demo comments out the tokenizer line. Re-enable it:

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
```

Set pad token if needed:

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### 6.3 Tokenization

Preferred approach:

```text
- tokenize full conversation
- mask labels for system + user tokens
- train loss only on assistant response tokens
```

If this is too much for the first implementation, train on full text first, then improve to completion-only loss.

---

## 7. Recommended Prompt Formatting

If tokenizer supports `apply_chat_template`, use it.

Pseudo-code:

```python
def build_text(prompt, answer=None):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise reasoning model. "
                "Infer the hidden rule from the examples and answer the final query. "
                "Always put only the final answer inside \\boxed{}."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    if answer is not None:
        messages.append(
            {
                "role": "assistant",
                "content": f"The final answer is \\boxed{{{answer}}}.",
            }
        )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=(answer is None),
    )
```

If the tokenizer does not support chat templates, use a simple plain format:

```text
System: You are a precise reasoning model. Infer the hidden rule from the examples and answer the final query. Always put only the final answer inside \boxed{}.

User:
{prompt}

Assistant:
The final answer is \boxed{answer}.
```

---

## 8. Task Family Classification

Add a deterministic classifier for analysis and validation.

```python
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
```

Use it to create stratified train/validation splits.

---

## 9. Local Validation Split

Create local validation before training.

Recommended split:

```text
90% train
10% valid
stratified by task_family
```

Report metrics by task family:

```text
overall accuracy
bit_manipulation accuracy
gravity_formula accuracy
unit_conversion accuracy
text_decryption accuracy
numeral_system accuracy
equation_symbol_transformation accuracy
```

Do not rely only on global accuracy.

---

## 10. Local Answer Extractor

Implement answer extraction to mimic the competition metric.

Priority:

```text
1. extract the last \boxed{...}
2. if no boxed answer, use fallback string/numeric extraction
3. for numeric tasks, optionally extract the last number
```

Basic implementation:

```python
import re

def extract_boxed(text: str):
    matches = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return None
```

For nested braces, write a more robust parser later.

---

## 11. Local Scoring

Use exact match for:

```text
bit_manipulation
text_decryption
numeral_system
equation_symbol_transformation
```

Use numeric tolerance for:

```text
gravity_formula
unit_conversion
```

Suggested numeric scoring:

```python
def numeric_equal(pred, gold, rel_tol=1e-3, abs_tol=1e-4):
    try:
        p = float(pred)
        g = float(gold)
    except Exception:
        return False
    return abs(p - g) <= max(abs_tol, rel_tol * max(abs(g), 1.0))
```

---

## 12. Training Implementation Options

### Option A: Transformers Trainer

Use `transformers.Trainer` if it works with the custom model.

Pros:

```text
- fast to implement
- logging/checkpointing handled
```

Cons:

```text
- may need careful collator for completion-only loss
```

### Option B: Manual PyTorch training loop

Use a manual loop if Trainer has compatibility issues.

Core steps:

```python
model.train()
for batch in dataloader:
    batch = {k: v.to(model.device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    if should_step:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

Because `device_map="auto"` can shard modules, be careful when moving tensors. Usually inputs should go to the first device or the model's expected input device.

---

## 13. Suggested SFT Hyperparameters

For H200 single card:

```text
dtype: bfloat16
LoRA rank: 16 first, then 32
LoRA alpha: 16 or 32
LoRA dropout: 0.05
max_seq_len: 1024 or 2048
per_device_train_batch_size: 1
gradient_accumulation_steps: 16 to 64
learning_rate: 1e-4 to 2e-4
epochs: 1 to 3 for first baseline
optimizer: AdamW or paged AdamW if available
warmup_ratio: 0.03
weight_decay: 0.0
gradient_checkpointing: enable if supported
```

For the first smoke test:

```text
train on 100 samples
run 5-20 optimizer steps
save adapter
package zip
verify files exist
```

Then scale up.

---

## 14. Minimal Code Skeleton for `# YOUR CODE HERE`

This is not guaranteed to run as-is, but it describes the intended logic.

```python
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import math
import random

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

SYSTEM_PROMPT = (
    "You are a precise reasoning model. "
    "Infer the hidden rule from the examples and answer the final query. "
    "Always put only the final answer inside \\boxed{}."
)

def format_example(prompt, answer):
    # Prefer chat template if available.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"The final answer is \\boxed{{{answer}}}."},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return (
            f"System: {SYSTEM_PROMPT}\n\n"
            f"User:\n{prompt}\n\n"
            f"Assistant:\nThe final answer is \\boxed{{{answer}}}."
        )

class SFTDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=2048):
        self.rows = df.to_dicts()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = format_example(row["prompt"], row["answer"])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

def collate_fn(features):
    batch = tokenizer.pad(
        features,
        padding=True,
        return_tensors="pt",
    )
    # Make sure padding tokens are ignored in loss.
    batch["labels"][batch["attention_mask"] == 0] = -100
    return batch

# Optional: quick smoke subset first.
# train_small = train.head(100)
train_dataset = SFTDataset(train, tokenizer, max_length=2048)
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn,
)

model.train()
model.gradient_checkpointing_enable()

optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,
    weight_decay=0.0,
)

grad_accum_steps = 32
num_epochs = 1
global_step = 0

optimizer.zero_grad()

for epoch in range(num_epochs):
    pbar = tqdm(train_loader, desc=f"epoch {epoch}")
    for step, batch in enumerate(pbar):
        # Use the first parameter device as input device.
        first_device = next(model.parameters()).device
        batch = {k: v.to(first_device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            pbar.set_postfix(loss=float(loss.detach().cpu()) * grad_accum_steps)

# Save adapter after training.
model.save_pretrained(OUTPUT_DIR)
```

Important: improve this skeleton before serious training by adding:

```text
- validation split
- completion-only loss
- checkpoint saving
- logging
- type-wise evaluation
```

---

## 15. Better Completion-Only Loss

The minimal skeleton trains on the full prompt and answer. This is acceptable for the first smoke test but not ideal.

Better behavior:

```text
labels for system/user prompt tokens = -100
labels for assistant answer tokens = real token ids
```

Implementation idea:

```python
prompt_text = format_prompt_only(prompt)
answer_text = f"The final answer is \\boxed{{{answer}}}."

full_text = prompt_text + answer_text

full_enc = tokenizer(full_text, ...)
prompt_enc = tokenizer(prompt_text, ...)

labels = full_enc["input_ids"].copy()
labels[:len(prompt_enc["input_ids"])] = [-100] * len(prompt_enc["input_ids"])
```

Be careful with chat template spacing and special tokens.

---

## 16. Evaluation Generation

Add a function to test the trained adapter on validation prompts.

Recommended generation settings:

```text
max_new_tokens: 64 to 256
do_sample: False for validation
temperature: not needed if do_sample=False
```

Example:

```python
model.eval()

def format_inference_prompt(prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"System: {SYSTEM_PROMPT}\n\nUser:\n{prompt}\n\nAssistant:\n"

def generate_answer(prompt):
    text = format_inference_prompt(prompt)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    first_device = next(model.parameters()).device
    enc = {k: v.to(first_device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded
```

---

## 17. Packaging

The official demo uses:

```python
subprocess.run("zip -m submission.zip *", shell=True, check=True)
```

This works but moves files into the zip.

Safer option:

```python
subprocess.run(
    "zip -r submission.zip adapter_config.json adapter_model.safetensors",
    shell=True,
    check=True,
)
```

However, if the official demo expects README and notebook to be present, keep the official command.

Recommended final:

```python
import os
import subprocess

assert os.path.exists(f"{OUTPUT_DIR}/adapter_config.json")
assert os.path.exists(f"{OUTPUT_DIR}/adapter_model.safetensors")

subprocess.run(
    "cd /kaggle/working && zip -r submission.zip adapter_config.json adapter_model.safetensors",
    shell=True,
    check=True,
)
```

---

## 18. Development Milestones

### M1: Official Demo Reproduction

Goal:

```text
Run the official demo unchanged and generate submission.zip.
```

Success criteria:

```text
adapter_config.json exists
adapter_model.safetensors exists
submission.zip exists
```

### M2: Add Tokenizer + SFT Formatting

Goal:

```text
Create formatted training examples from train.csv.
```

Success criteria:

```text
A sample formatted example contains prompt and:
The final answer is \boxed{...}.
```

### M3: Smoke Training

Goal:

```text
Train LoRA on 100 examples for 5-20 optimizer steps.
```

Success criteria:

```text
loss is finite
adapter saves successfully
submission.zip packages successfully
```

### M4: Local Validation

Goal:

```text
Create train/valid split and evaluator.
```

Success criteria:

```text
can report overall and type-wise accuracy
can extract boxed answers
```

### M5: Full SFT Baseline

Goal:

```text
Train rank=16 or rank=32 LoRA on train split.
```

Success criteria:

```text
local validation improves over untrained adapter
model consistently outputs boxed answers
```

### M6: Synthetic Data

Goal:

```text
Generate trusted synthetic data by task family.
```

Success criteria:

```text
synthetic answers are programmatically verified
validation improves on weak task families
```

### M7: Optional GRPO/RLVR

Goal:

```text
Use local evaluator as reward function.
```

Success criteria:

```text
improves validation accuracy or boxed-answer stability
```

---

## 19. Final Agent Instructions

When modifying the official notebook, do the following:

```text
1. Keep all official import and model-loading code.
2. Keep the official MODEL_PATH.
3. Keep trust_remote_code=True.
4. Keep the official LoRA target_modules pattern unless inspection proves otherwise.
5. Insert SFT training code only in the # YOUR CODE HERE section.
6. Train only LoRA parameters.
7. Make the assistant output end with exactly one \boxed{} answer.
8. Save adapter with model.save_pretrained(OUTPUT_DIR).
9. Package adapter_config.json and adapter_model.safetensors into submission.zip.
10. Do a smoke run before full training.
```

Do not:

```text
- Do not train full model weights.
- Do not submit full model weights.
- Do not start with PPO.
- Do not use unverified long CoT traces.
- Do not assume common Llama target module names.
- Do not ignore boxed-answer extraction.
- Do not package unnecessary checkpoint/cache files.
```

---

## 20. Immediate Next Step

Implement M2 and M3 first.

Concrete first task:

```text
Take Ryan Holbrook's official notebook.
Uncomment/load tokenizer.
Build SFT examples from train.csv.
Train on 100 rows for 5-20 optimizer steps.
Save adapter.
Package submission.zip.
Confirm the zip contains adapter_config.json and adapter_model.safetensors.
```

Stop there and report:

```text
- whether model loaded
- LoRA trainable parameter count
- whether loss was finite
- whether adapter saved
- contents of submission.zip
```
