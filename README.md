# Nemotron LoRA SFT Pipeline

This project builds a LoRA SFT training pipeline for the NVIDIA Nemotron reasoning challenge. The submission artifact is a `submission.zip` containing only the LoRA adapter files:

- `adapter_config.json`
- `adapter_model.safetensors`

The base model is expected at `/data2/yaoxuran/models`.

## Main Files

- `train_sft.py` - LoRA SFT training, validation, adapter saving, zip packaging, and resumable checkpoints.
- `generate_synthetic_data.py` - Programmatic synthetic data generator for all six task families.
- `data/train.csv` - Official training data.
- `data/synthetic_v1.csv` - Generated synthetic data.
- `data/train_plus_synthetic_v1.csv` - Official + synthetic mixed training data.
- `PROJECT_PROGRESS.md` - Current run status and experiment notes.
- `runs/` - Training outputs, logs, adapters, checkpoints, and packaged submissions.

## Generate Synthetic Data

```bash
python generate_synthetic_data.py
```

Default output:

```text
data/synthetic_v1.csv
data/synthetic_v1_metadata.jsonl
data/synthetic_v1_report.json
data/train_plus_synthetic_v1.csv
```

`synthetic_v1` contains 6,000 verified answer-only rows, 1,000 per task family.

## Train

Example synthetic-mix run:

```bash
nohup env GPU_ID=5 \
  TRAIN_CSV=/data2/yaoxuran/llm_infer/data/train_plus_synthetic_v1.csv \
  LORA_RANK=16 LORA_ALPHA=16 \
  BATCH_SIZE=2 GRAD_ACCUM_STEPS=8 \
  NUM_EPOCHS=2 MAX_TRAIN_STEPS=1100 \
  ENABLE_BASELINE_EVAL=0 \
  VAL_MAX_SAMPLES=100 EVAL_EVERY_STEPS=100 EVAL_MAX_NEW_TOKENS=128 \
  OUTPUT_DIR=/data2/yaoxuran/llm_infer/runs/synthetic_v1_13h/adapter \
  SUBMISSION_DIR=/data2/yaoxuran/llm_infer/runs/synthetic_v1_13h/artifacts \
  EVAL_OUTPUT_DIR=/data2/yaoxuran/llm_infer/runs/synthetic_v1_13h/eval \
  CHECKPOINT_DIR=/data2/yaoxuran/llm_infer/runs/synthetic_v1_13h/checkpoints \
  .venv/bin/python train_sft.py \
  > runs/synthetic_v1_13h/synthetic_v1_13h.log 2>&1 &
```

Current training defaults use response-only loss and align the answer format with evaluation:

```text
RESPONSE_ONLY_LOSS=1
TRAIN_FINAL_ANSWER_PREFILL=1
INFERENCE_FINAL_ANSWER_PREFILL=1
INFERENCE_STOP_AFTER_BOXED=1
FINAL_ANSWER_PREFILL_TEXT="</think>\nThe final answer is \boxed{"
```

This masks system/user prompt tokens from the SFT loss and trains the assistant side to close `<think>`, enter `The final answer is \boxed{...}`, and then evaluate with the same final-answer prefill. Set any of these flags to `0` to restore the older behavior for comparison.

Monitor:

```bash
tail -f runs/synthetic_v1_13h/synthetic_v1_13h.log
```

## W&B Tracking

W&B is available in the project venv. A long synthetic run with W&B, periodic eval, and checkpoints can be started with:

```bash
bash scripts/start_synthetic_v1_wandb.sh
```

Default tracking settings:

```text
WANDB_ENABLED=1
WANDB_PROJECT=llm-infer-sft
WANDB_RUN_NAME=synthetic_v1_wandb_long
WANDB_MODE=online
EVAL_EVERY_STEPS=100
CHECKPOINT_EVERY_STEPS=100
```

W&B logs dense training curves for `train/loss` and `train/lr`, plus periodic eval curves for `eval/accuracy`, `eval/boxed_rate`, and `eval_family/<family>/accuracy`. `EVAL_EVERY_STEPS` controls how often evaluation points are added to the curves.

The machine must be logged in to W&B for online runs. To force offline logging:

```bash
WANDB_MODE=offline bash scripts/start_synthetic_v1_wandb.sh
```

Offline runs can be uploaded later:

```bash
.venv/bin/wandb sync runs/synthetic_v1_wandb_long/wandb/wandb/offline-run-*
```

## Checkpoints

New runs save resumable checkpoints by default:

```text
CHECKPOINT_EVERY_STEPS=100
KEEP_LAST_CHECKPOINTS=2
```

Each checkpoint contains the LoRA adapter plus optimizer, scheduler, epoch, batch index, global step, and RNG state.

Resume from the latest checkpoint:

```bash
RESUME_FROM_CHECKPOINT=latest \
CHECKPOINT_DIR=/data2/yaoxuran/llm_infer/runs/synthetic_v1_13h/checkpoints \
...same training env vars... \
.venv/bin/python train_sft.py
```

## Evaluate Adapter

Run evaluation without retraining:

```bash
GPU_ID=6 \
ADAPTER_DIR=/data2/yaoxuran/llm_infer/runs/synthetic_v1_13h/adapter_best \
TRAIN_CSV=/data2/yaoxuran/llm_infer/data/train_plus_synthetic_v1.csv \
VAL_MAX_SAMPLES=100 EVAL_MAX_NEW_TOKENS=128 \
EVAL_OUTPUT_DIR=/data2/yaoxuran/llm_infer/runs/synthetic_v1_13h/eval \
EVAL_LABEL=synthetic_v1_100 \
.venv/bin/python eval_adapter.py
```

By default eval now pre-fills `</think>\nThe final answer is \boxed{` and stops after the first complete boxed span, so escaped answer-internal braces do not terminate generation early. This makes the saved `decoded` field directly parseable as a boxed final answer and avoids spending the full token budget inside long reasoning. For diagnostic comparison with the old behavior, run with:

```bash
INFERENCE_FINAL_ANSWER_PREFILL=0 INFERENCE_STOP_AFTER_BOXED=0 ...
```

## Outputs

Final adapter:

```text
runs/<run_name>/adapter/
```

Best validation adapter:

```text
runs/<run_name>/adapter_best/
```

Evaluation details:

```text
runs/<run_name>/eval/eval_<step>.jsonl
runs/<run_name>/eval/eval_<step>.csv
runs/<run_name>/eval/eval_<step>_summary.json
```

Each eval record includes `prompt`, `gold`, raw `decoded` output, extracted `pred`, task `family`, `correct`, and whether a boxed answer was parsed. Set `VAL_MAX_SAMPLES=100` or higher for a more stable estimate; set `EVAL_SAVE_DETAILS=0` to disable per-sample files.

Submission zip:

```text
runs/<run_name>/artifacts/submission.zip
```
