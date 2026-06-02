# Nemotron SFT Project Progress

Last updated: 2026-06-02 17:10:29 CST

| Stage | Status | Evidence | Next action |
| --- | --- | --- | --- |
| M1 Official demo reproduction | Done | `output_adapter/adapter_config.json`, `output_adapter/adapter_model.safetensors`, and `submission.zip` exist. | Keep zip contents minimal. |
| M2 Tokenizer + SFT formatting | Done | `train_sft.py` loads tokenizer and prints formatted examples. | Keep prompt/answer format stable. |
| M3 Smoke training/package | Done | Smoke run on 2026-06-01 completed training, save, and zip packaging. | Scale up to a longer run. |
| M4 Local validation | Done | Smoke run evaluated 2 validation samples before and during training; generation no longer blocks. | Expand validation sample count for a broader check. |
| M5 Full SFT baseline | In progress | Rank32 20-step baseline completed, rank16 20-step baseline completed, and rank16 `NUM_EPOCHS=1` formal run is live on GPU 6 (`PID 2739700`). Currently around optimizer step 120/535 and batch 960/4277. Speed is ~4.8-5.1 s/batch, ~39-41 s/optimizer step including eval overhead. GPU 6: ~76 GB / 81 GB. | Continue to epoch end; estimated training finish around 22:10-22:15 CST, final zip around 22:20-22:35 CST. |
| M6 Synthetic data | Retraining | `generate_synthetic_data.py` produced verified `data/synthetic_v1.csv` (6,000 rows, 1,000/family), `data/synthetic_v1_metadata.jsonl`, `data/synthetic_v1_report.json`, and `data/train_plus_synthetic_v1.csv` (15,500 rows total). Old synthetic-mix SFT completed at `MAX_TRAIN_STEPS=1100`; adapter: `runs/synthetic_v1_13h/adapter_best`; zip: `runs/synthetic_v1_13h/artifacts/submission.zip`. Old-format 100-sample eval: `11/100` accuracy and `11/100` boxed rate. Fixed-format eval on the same adapter: `29/100` accuracy and `100/100` boxed rate. | Run a new synthetic SFT with response-only loss plus final-answer prefill/boxed stop, then evaluate on 100+ samples. |
| M7 RLVR/GRPO | Not started | No RL pipeline found. | Defer. |

## Active Issue

Synthetic v1 eval accuracy is low because most generations do not reach a final boxed answer. On the 100-sample eval, only 11/100 outputs contained a complete `\boxed{...}` answer, and all 11 boxed predictions were correct. The other 89 samples were unboxed/truncated, so the main failure mode is output format and generation length rather than wrong answers inside boxed outputs. The previous Nemotron-H generation cache issue remains covered by the `use_cache=False` fallback, and the gradient-checkpointing/evaluation interaction has been fixed.

## Synthetic v1 Eval Diagnosis

Eval command label: `synthetic_v1_100`

Outputs:

- Summary: `runs/synthetic_v1_13h/eval/eval_synthetic_v1_100_summary.json`
- JSONL details: `runs/synthetic_v1_13h/eval/eval_synthetic_v1_100.jsonl`
- CSV details: `runs/synthetic_v1_13h/eval/eval_synthetic_v1_100.csv`
- Log: `runs/synthetic_v1_13h/eval/synthetic_v1_100.eval.log`

Metrics:

- Overall accuracy: `11/100 = 0.11`
- Boxed answer rate: `11/100 = 0.11`
- `numeral_system`: `11/16 = 0.6875`, boxed `11/16`
- `bit_manipulation`, `equation_symbol_transformation`, `gravity_formula`, `text_decryption`, and `unit_conversion`: all `0%`, boxed `0%`

Root causes:

- Inference prompt ends with the chat-template generation prompt, which for this model leads into `<think>` and encourages long reasoning. Training examples use assistant content like `The final answer is \boxed{answer}.`, which renders as a short assistant answer with an empty think block in the template. This train/inference mismatch makes eval spend tokens in reasoning instead of final-answer emission.
- `EVAL_MAX_NEW_TOKENS=128` is too short for the current generation behavior. Non-numeral tasks often consume the whole budget while reasoning and are truncated before `\boxed{...}`. One numeral sample reached `So answer: \boxed` and was also cut before the boxed value.
- Training did not show eval improvement: `runs/synthetic_v1_13h/synthetic_v1_13h.log` stayed at `2/20` accuracy and `2/20` boxed rate from early evals through step 1100.
- The run stopped at `MAX_TRAIN_STEPS=1100` during epoch 1/2, so it did not complete the configured two epochs.
- Current SFT labels train on the whole prompt plus answer (`labels = input_ids.copy()`), so answer-format supervision is diluted by long prompt tokens. Response-only labels should give a stronger signal for producing the final boxed answer.

Recommended next actions:

- Fixed eval inference now pre-fills `</think>\nThe final answer is \boxed{` and stops after the first generated `}`. On the old adapter this raised boxed rate from `11/100` to `100/100` and accuracy from `11/100` to `29/100`.
- Fixed training now uses response-only loss by default, masking system/user prompt tokens and supervising only the assistant final-answer tokens.
- Next run should retrain with these defaults and then re-evaluate on 100+ samples. Remaining hard families after format fixing are `equation_symbol_transformation` and `gravity_formula`, both still `0%` on the old adapter.

## Update Log

| Time | Update |
| --- | --- |
| 2026-06-01 10:36:54 CST | Created progress table. Current focus: fix generation cache error, verify baseline eval, then resume SFT run. |
| 2026-06-01 10:36:54 CST | Patched `train_sft.py` with generation fallback, smoke-mode env overrides, and early-stop support. Next: GPU smoke run. |
| 2026-06-01 11:08:38 CST | Smoke run passed baseline generation, then failed at first backward because evaluation re-enabled gradient checkpointing. Patched evaluation to leave checkpointing disabled and set smoke `NUM_WORKERS=0`. |
| 2026-06-01 11:18:43 CST | Smoke rerun succeeded: baseline eval, 2 training steps, adapter save, and `submission.zip` packaging all completed. |
| 2026-06-01 14:08:01 CST | Mini baseline succeeded for 5 optimizer steps with finite loss and submission packaging. This confirms the training path is stable past smoke. |
| 2026-06-01 14:54:50 CST | Starting full-data 20-step baseline on GPU 6. Outputs: `output_adapter_baseline_20/`, zip artifacts: `output_adapter_baseline_20_artifacts/`, log: `baseline_20.log`. |
| 2026-06-01 15:19:26 CST | Full-data baseline with `BATCH_SIZE=4` OOMed during training. Patched default full-run batch size to 1 and will use gradient accumulation for the next run. |
| 2026-06-01 15:26:16 CST | `BATCH_SIZE=1` full-data run is stable through step 5/20. Current speed is about 40 seconds per optimizer step; expected 20-step run completion is roughly 15:45 CST including eval/save/zip. |
| 2026-06-01 15:42:09 CST | Rank32 full-data 20-step baseline completed. Final step20 eval was 0/5 accuracy and 0/5 boxed rate; adapter and zip were produced under `output_adapter_baseline_20_b1*`. Starting rank16 test next. |
| 2026-06-01 15:58:58 CST | Rank16 `BATCH_SIZE=2` run is stable past step 14/20. Speed is about 40-42 seconds per optimizer step, but each step covers 16 examples, so one epoch is estimated at about 6-6.5 hours. |
| 2026-06-01 16:13:00 CST | Launched formal rank16 1-epoch baseline on GPU 6 with `BATCH_SIZE=2`, `GRAD_ACCUM_STEPS=8`, and `ENABLE_BASELINE_EVAL=0`. PID: `2739700`. First step-100 eval should land in about 1h10m if speed holds. |
| 2026-06-01 16:15:10 CST | Confirmed formal run is training: GPU 6 uses about 75.5 GB, process `2739700` is live, and log has reached optimizer step 5. |
| 2026-06-01 16:24:00 CST | Formal run at optimizer step 10/535 (1.9%), batch 84/4277. Speed ~5.0 s/batch, ~40 s/step. Loss warmup phase: 2.73→2.91→3.76→2.94→2.41→2.64→2.62→3.50→2.62→2.47. LR at 6.62e-05 (still warming up). GPU 6: 76 GB, 21% util. ETA full epoch ~22:10 CST. |
| 2026-06-01 16:35:54 CST | Formal run reached optimizer step 36. Current code is single-GPU only (`CUDA_VISIBLE_DEVICES` is forced to one GPU and `device_map={"": 0}`), so simply exposing 2 GPUs will not speed this run without changing the launcher/model placement. |
| 2026-06-01 17:26:08 CST | Formal run passed step-100 eval and resumed training. Latest observed progress: step 110/535, batch 883/4277. Speed is ~4.8-5.0 s/batch, ~39-41 s/optimizer step; step-100 eval took ~2m12s. ETA for 1 epoch training completion is ~22:10-22:20 CST, with adapter save/zip shortly after. |
| 2026-06-01 17:32:55 CST | Formal run reached about step 120/535 and batch 960/4277 (22.4%). Effective speed is ~4.8-5.1 s/batch and ~40.8 s/optimizer step including the step-100 eval overhead. ETA for 1 epoch training completion is ~22:10-22:15 CST; final adapter save/zip should follow by ~22:20-22:35 CST. |
| 2026-06-01 18:07:52 CST | Started M6 synthetic data. Added `generate_synthetic_data.py`; generated `synthetic_v1` with 6,000 verified answer-only rows across six task families and `train_plus_synthetic_v1.csv` with 15,500 total rows. Patched `train_sft.py` to force `prompt`/`answer` string schema so mixed synthetic answers load reliably. |
| 2026-06-01 19:10:19 CST | Launched synthetic-mix SFT on GPU 5. Command uses `TRAIN_CSV=data/train_plus_synthetic_v1.csv`, `LORA_RANK=16`, `BATCH_SIZE=2`, `GRAD_ACCUM_STEPS=8`, `NUM_EPOCHS=2`, `MAX_TRAIN_STEPS=1100`, and `ENABLE_BASELINE_EVAL=0`. Run is in training loop with GPU 5 around 71 GB used. |
| 2026-06-01 19:39:17 CST | Added resumable checkpoint support to `train_sft.py`: saves LoRA adapter plus optimizer/scheduler/global step/epoch/batch/RNG state under `CHECKPOINT_DIR`, supports `RESUME_FROM_CHECKPOINT=latest` or a checkpoint path, and keeps the most recent checkpoints via `KEEP_LAST_CHECKPOINTS`. Current already-running synthetic process will not inherit this code until restarted. |
| 2026-06-02 15:26:53 CST | Completed 100-sample adapter eval for `runs/synthetic_v1_13h/adapter_best`; outputs saved under `runs/synthetic_v1_13h/eval/`. Result: `11/100` accuracy, `11/100` boxed rate, with all correct/boxed examples coming from `numeral_system`. |
| 2026-06-02 16:41:10 CST | Diagnosed low synthetic eval accuracy. Main issue is missing/truncated final boxed answers caused by train/inference format mismatch and too-short generation budget; secondary issue is whole-sequence SFT loss diluting answer supervision. |
| 2026-06-02 16:58:40 CST | Patched `train_sft.py` and `eval_adapter.py`: added response-only SFT labels, final-answer prefill, and stop-after-boxed generation. Added README notes for the new flags. |
| 2026-06-02 17:00:30 CST | Ran fixed-format eval on old `runs/synthetic_v1_13h/adapter_best`: `29/100` accuracy, `100/100` boxed rate. Family results: `numeral_system 15/16`, `unit_conversion 9/14`, `text_decryption 3/22`, `bit_manipulation 2/15`, `equation_symbol_transformation 0/17`, `gravity_formula 0/16`. |
| 2026-06-02 17:10:29 CST | Smoke training with new response-only/final-answer format passed: 2 optimizer steps, checkpoints, final adapter, and `submission.zip` were saved successfully. |
