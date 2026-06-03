# Nemotron SFT Project Progress

Last updated: 2026-06-03 10:59:00 CST

| Stage | Status | Evidence | Next action |
| --- | --- | --- | --- |
| M1 Official demo reproduction | Done | `output_adapter/adapter_config.json`, `output_adapter/adapter_model.safetensors`, and `submission.zip` exist. | Keep zip contents minimal. |
| M2 Tokenizer + SFT formatting | Done | `train_sft.py` loads tokenizer and prints formatted examples. | Keep prompt/answer format stable. |
| M3 Smoke training/package | Done | Smoke run on 2026-06-01 completed training, save, and zip packaging. | Scale up to a longer run. |
| M4 Local validation | Done | Smoke run evaluated 2 validation samples before and during training; generation no longer blocks. | Expand validation sample count for a broader check. |
| M5 Full SFT baseline | Archived | Earlier smoke/mini/baseline/rank test artifacts were removed on 2026-06-02 to avoid mixing old-code results with the current response-only/DDP runs. | No action unless a fresh non-synthetic baseline is needed. |
| M6 Synthetic data | Running | `synthetic_v1` produced 6,000 rows but later validation found solvability bugs. `synthetic_v2` fixed visibility/numeric consistency and powers the current no-brace v2 DDP run. As of 2026-06-03 10:50 CST, that run is still live on GPUs 5/6 and reached step-1900 eval at `51/100` accuracy with `100/100` boxed rate. A new `synthetic_v3` dataset has been generated with weak-family upweighting and brace-safe answer support: `data/synthetic_v3.csv`, `data/synthetic_v3_metadata.jsonl`, `data/synthetic_v3_report.json`, and `data/train_plus_synthetic_v3.csv`. | Let the current v2 run continue until the next decision point, then start a short v3 warm-start experiment using the updated brace-safe `train_sft.py`. |
| M7 RLVR/GRPO | Not started | No RL pipeline found. | Defer. |

## Active Issue

The format issue has mostly been fixed: response-only loss, final-answer prefill, and stop-after-boxed generation moved boxed rate to `100/100` on both the old adapter fixed-format eval and the current step-100 eval. The active issue is now data quality and task solvability. `synthetic_v1` contains a systematic `equation_symbol_transformation` bug where many final-query symbols are never shown in the example inputs, making the mapping impossible to infer. `synthetic_v2` fixes this and also aligns displayed numeric inputs with computed numeric answers.

Current recommendation: use `data/train_plus_synthetic_v2_no_brace_answers.csv` for the current boxed-format formal training. The live v1 run has been stopped after preserving step-100 artifacts; use its results only as a short trend probe and do not treat v1 `equation_symbol_transformation` scores as a clean undertraining signal. Do not continue the unfiltered `data/train_plus_synthetic_v2.csv` run until brace-safe boxed formatting, stopping, extraction, and unescaping are implemented.

Brace-answer handling note: `data/train_plus_synthetic_v2.csv` still contains `173` official `equation_symbol_transformation` rows whose gold answers include `{` or `}`. These rows are not from `synthetic_v2`, but the current `\boxed{answer}` format, `StopAfterBoxClose`, and `extract_boxed()` parser cannot safely represent or recover answers with internal braces. For the next clean v2 baseline, these official brace-answer rows should be filtered out rather than used as noisy supervision/eval. In a later data/code version, add them back by implementing brace-safe answer formatting, stopping, extraction, and unescaping.

Current formal run:

- Run root: `runs/synthetic_v2_no_brace_response_only_ddp_10000/`
- Log: `runs/synthetic_v2_no_brace_response_only_ddp_10000/synthetic_v2_no_brace_response_only_ddp_10000.log`
- PID file: `runs/synthetic_v2_no_brace_response_only_ddp_10000/synthetic_v2_no_brace_response_only_ddp_10000.pid`
- W&B: `https://wandb.ai/yaozhonger7-shantou-university/llm-infer-sft/runs/di0xzgvq`
- Init adapter: none; this run starts clean from the base model with a new LoRA adapter.
- Output adapter: `runs/synthetic_v2_no_brace_response_only_ddp_10000/adapter`
- Checkpoints: `runs/synthetic_v2_no_brace_response_only_ddp_10000/checkpoints`, saved every 100 optimizer steps, keeping last 10.
- Eval outputs: `runs/synthetic_v2_no_brace_response_only_ddp_10000/eval`
- Main parameters: `MAX_TRAIN_STEPS=10000`, `NUM_EPOCHS=12`, `BATCH_SIZE=2`, `GRAD_ACCUM_STEPS=4`, `LORA_RANK=16`, `VAL_MAX_SAMPLES=100`, `EVAL_EVERY_STEPS=100`, `DISABLE_CUDNN_SDP=1`.

Note: the no-brace v2 run intentionally does not resume from v1 adapters/checkpoints or the stopped unfiltered v2 attempt, because those data/run states are not clean baselines. New checkpoints from this run include optimizer/scheduler/global-step state and can be resumed with `RESUME_FROM_CHECKPOINT=latest`. Latest finished evals are saved under `runs/synthetic_v2_no_brace_response_only_ddp_10000/eval/`; best so far is `step-1900` with `51/100` accuracy and `100/100` boxed rate.

GPU status at 2026-06-03 10:50 CST: there is no fully free H100. GPUs 5/6 are occupied by the live v2 DDP training processes (`train_sft.py`, about 79-80 GB each). GPUs 0/1 have about 65 GB free, and GPUs 2/3/4/7 have less free memory. The current single-card eval/training path usually needs roughly a full H100, so avoid launching another 30B eval on those partial cards unless model loading is changed to use multi-card sharding or lower-memory inference.

## Synthetic v3 Data Fix

Files:

- Synthetic only: `data/synthetic_v3.csv`
- Metadata: `data/synthetic_v3_metadata.jsonl`
- Report: `data/synthetic_v3_report.json`
- Combined train: `data/train_plus_synthetic_v3.csv`

Generator/data changes:

- `generate_synthetic_data.py` now defaults to `synthetic_v3`.
- Added `--family-counts` so weak families can be upweighted without inflating already-solved tasks.
- Default v3 synthetic counts are `bit_manipulation=3000`, `equation_symbol_transformation=3000`, `gravity_formula=3000`, `text_decryption=1500`, `numeral_system=500`, and `unit_conversion=500`.
- `bit_manipulation` now uses a canonical, uniquely identifiable XOR/rotation/bit-reversal rule space and validates that the example set identifies exactly one rule in that candidate space.
- `equation_symbol_transformation` now mixes character substitution with official-like numeric operator rules. Character substitution can include brace answers because `train_sft.py` now escapes boxed answers.
- `gravity_formula` now uses more examples and sometimes includes simple anchor times to make hidden-`g` inference easier.
- `train_sft.py` now uses brace-safe boxed answer escaping, stopping, extraction, and unescaping; eval summaries also include `source_summary` and `source_family_summary`.

Validation/audit results:

- `data/synthetic_v3.csv`: `11,500` synthetic rows.
- `data/train_plus_synthetic_v3.csv`: `21,000` total rows, `9,500` official + `11,500` synthetic.
- Combined family counts: `bit_manipulation=4602`, `equation_symbol_transformation=4555`, `gravity_formula=4597`, `text_decryption=3076`, `numeral_system=2076`, `unit_conversion=2094`.
- v3 intentionally restores brace-answer coverage: `train_plus_synthetic_v3.csv` has `548` equation answers containing `{` or `}`. Use it only with the updated brace-safe `train_sft.py`; do not train it with older code.

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
- The current response-only DDP v1 run reached step-100 eval at `32/100` accuracy and `100/100` boxed rate. Family results: `numeral_system 13/16`, `unit_conversion 11/14`, `text_decryption 6/22`, `bit_manipulation 2/15`, `equation_symbol_transformation 0/17`, `gravity_formula 0/16`.
- Step-100 details show errors are no longer caused by missing boxed answers. `gravity_formula` predictions are often numerically close but outside tolerance; `text_decryption` partially improves; `bit_manipulation` remains hard; `equation_symbol_transformation` is not cleanly interpretable because v1 has many impossible samples.
- Next formal run should use `data/train_plus_synthetic_v2.csv` with the same response-only/final-answer settings.

## Synthetic v2 Data Fix

Files:

- Synthetic only: `data/synthetic_v2.csv`
- Metadata: `data/synthetic_v2_metadata.jsonl`
- Report: `data/synthetic_v2_report.json`
- Combined train: `data/train_plus_synthetic_v2.csv`
- Current combined train, brace-answer rows removed: `data/train_plus_synthetic_v2_no_brace_answers.csv`
- Filter report: `data/train_plus_synthetic_v2_no_brace_answers_report.json`

Generator changes in `generate_synthetic_data.py`:

- Added versioned output support; default version is now `synthetic_v2`.
- Added built-in validation before writing files.
- Fixed `equation_symbol_transformation` so every unique query character appears in at least one example input.
- Fixed `text_decryption` so encrypted query characters, including those from `the`, are visible in encrypted examples.
- Fixed `gravity_formula` and `unit_conversion` so displayed 1-decimal inputs are also the values used to compute displayed answers.

Validation results:

- `data/synthetic_v1.csv`: equation missing-visible-mapping `587/1000`; text missing-visible-mapping `3/1000`.
- `data/train_plus_synthetic_v1.csv`: equation missing-visible-mapping `1545/2555`; text missing-visible-mapping `974/2576` by visibility heuristic, mostly from official/v1 mixed rows.
- Current validation split from `train_plus_synthetic_v1.csv`: equation missing-visible-mapping `154/255`.
- `data/synthetic_v2.csv`: equation missing-visible-mapping `0/1000`; text missing-visible-mapping `0/1000`; numeric display consistency errors `0`.
- `data/train_plus_synthetic_v2.csv`: `15,500` rows total, source counts `official_train=9500`, `synthetic_v2=6000`; family counts remain balanced with `1000` synthetic rows per family.
- Known combined-data caveat: `train_plus_synthetic_v2.csv` has `173` official `equation_symbol_transformation` answers containing `{` or `}` (`158` in train, `15` in val for the current split). The current run uses `train_plus_synthetic_v2_no_brace_answers.csv`, which removes these 173 rows and keeps 15,327 rows. Future version should restore them after adding brace-safe boxed parsing/formatting and a stop criterion that does not stop at an answer-internal `}`.

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
| 2026-06-02 18:36:00 CST | Added and smoke-tested DDP support in `train_sft.py`. Rank32 DDP smoke OOMed at optimizer step; rank16 DDP smoke passed 2 optimizer steps, checkpointing, final adapter save, and zip packaging. |
| 2026-06-02 18:45:00 CST | Removed old-code or low-value artifacts: earlier smoke/mini/baseline/rank tests, old single-card W&B long run, empty long run, DDP smoke outputs, and root 2-step test checkpoints. Preserved `runs/synthetic_v1_13h/` for comparison and as adapter initialization. |
| 2026-06-02 18:48:27 CST | Launched formal response-only DDP retraining on GPUs 5/6 from `runs/synthetic_v1_13h/adapter_best`. Run root: `runs/synthetic_v1_response_only_ddp_13h/`; PID: `797914`; W&B: `https://wandb.ai/yaozhonger7-shantou-university/llm-infer-sft/runs/jwt02l7y`; checkpoint every 100 optimizer steps, keep last 3. |
| 2026-06-02 18:49:00 CST | First DDP launch failed while loading old `adapter_best`: PEFT entered tensor-parallel adapter sharding after early distributed init and hit `ImportError: EmbeddingParallel` with the installed transformers version. Failed log preserved as `runs/synthetic_v1_response_only_ddp_13h/synthetic_v1_response_only_ddp_13h.failed_1848.log`. |
| 2026-06-02 18:50:40 CST | Patched `train_sft.py` to delay `dist.init_process_group()` until after LoRA adapter loading, then verified `py_compile`. This avoids PEFT's incompatible TP-sharding path while preserving DDP training. |
| 2026-06-02 18:52:30 CST | Relaunched formal response-only DDP run on GPUs 5/6. PID: `1008354`; W&B: `https://wandb.ai/yaozhonger7-shantou-university/llm-infer-sft/runs/59qyyvuj`. The run loaded `runs/synthetic_v1_13h/adapter_best` successfully and entered `[7/8] Training`; GPU 5/6 memory was about 70 GB / 66 GB. |
| 2026-06-02 18:54:30 CST | Confirmed the DDP run is computing: log reached optimizer step 3 with losses `1.7301 -> 1.5449 -> 1.3988`; GPU 5/6 memory was about 80 GB / 78 GB with nonzero utilization. |
| 2026-06-02 19:19:01 CST | Stopped the short `runs/synthetic_v1_response_only_ddp_13h/` run by SIGTERM at optimizer step ~63. It had not reached `CHECKPOINT_EVERY_STEPS=100` or `EVAL_EVERY_STEPS=100`, so no useful checkpoint/eval was produced. |
| 2026-06-02 19:20:44 CST | Started a fresh 10000-step response-only DDP run on GPUs 5/6 from `runs/synthetic_v1_13h/adapter_best`. Run root: `runs/synthetic_v1_response_only_ddp_10000/`; torchrun PID file: `runs/synthetic_v1_response_only_ddp_10000/synthetic_v1_response_only_ddp_10000.pid`; W&B: `https://wandb.ai/yaozhonger7-shantou-university/llm-infer-sft/runs/13pk1w1w`; checkpoints/eval every 100 optimizer steps, keeping last 10 checkpoints. |
| 2026-06-02 19:23:30 CST | Confirmed the 10000-step run entered `[7/8] Training`: log reached optimizer step 2, GPU 5/6 memory was about 79.6 GB / 78.1 GB, and the run reported `Total optimizer steps: 10476` with early stop at `MAX_TRAIN_STEPS=10000`. |
| 2026-06-02 19:34:34 CST | The first 10000-step attempt failed at optimizer step ~29 in `loss.backward()` with `RuntimeError: Expected mha_graph.execute(...).is_good()`, not OOM. Failed log preserved as `runs/synthetic_v1_response_only_ddp_10000/synthetic_v1_response_only_ddp_10000.failed_mha_1934.log`. |
| 2026-06-02 19:58:44 CST | Added `DISABLE_CUDNN_SDP` to `train_sft.py` and restarted the 10000-step run with `DISABLE_CUDNN_SDP=1`. New torchrun PID: `630931`; W&B: `https://wandb.ai/yaozhonger7-shantou-university/llm-infer-sft/runs/ykl0u58e`; output root remains `runs/synthetic_v1_response_only_ddp_10000/`. |
| 2026-06-02 20:04:30 CST | Confirmed the restarted run is active: log reached optimizer step ~9 with no runtime errors, and GPU 5/6 remain loaded around 80 GB / 78 GB. No checkpoint/eval yet; first save/eval is at step 100. |
| 2026-06-02 20:41:00 CST | Step-100 eval for `runs/synthetic_v1_response_only_ddp_10000/` completed: `32/100` accuracy and `100/100` boxed rate. Family results: `numeral_system 13/16`, `unit_conversion 11/14`, `text_decryption 6/22`, `bit_manipulation 2/15`, `equation_symbol_transformation 0/17`, `gravity_formula 0/16`. The run saved `adapter_best` and `checkpoints/checkpoint-000100`, then continued training. |
| 2026-06-02 20:54:00 CST | Audited `generate_synthetic_data.py` and found `synthetic_v1` solvability bugs. Patched generator for `synthetic_v2`: versioned outputs, built-in validation, guaranteed visible equation query mappings, visible text query cipher chars, and numeric answer/display consistency. Generated `data/synthetic_v2.csv`, `data/synthetic_v2_metadata.jsonl`, `data/synthetic_v2_report.json`, and `data/train_plus_synthetic_v2.csv`; independent checks found equation missing-visible-mapping `0/1000`, text missing-visible-mapping `0/1000`, and numeric display consistency errors `0`. |
| 2026-06-02 21:01:42 CST | Stopped the live `synthetic_v1_response_only_ddp_10000` run after preserving step-100 eval/checkpoint. Decision: do not fall back to raw-only `data/train.csv`; use fixed `data/train_plus_synthetic_v2.csv` for the next formal run because it preserves the 9,500 official rows plus 6,000 validated balanced synthetic rows. |
| 2026-06-02 21:03:47 CST | Started clean v2 response-only DDP run on GPUs 5/6. Run root: `runs/synthetic_v2_response_only_ddp_10000/`; PID: `127425`; W&B: `https://wandb.ai/yaozhonger7-shantou-university/llm-infer-sft/runs/r80304mq`; `RESUME_FROM_CHECKPOINT=(none)`, so it starts from the base model with a new LoRA adapter. |
| 2026-06-02 21:07:00 CST | Confirmed v2 run is active: it entered `[7/8] Training`, reached optimizer step 2 with losses `2.8341 -> 2.7281`, and GPU 5/6 were both loaded with active compute. First v2 eval/checkpoint is expected at step 100. |
| 2026-06-02 21:11:58 CST | Checked `synthetic_v2` and found all six generated families present with `1000` rows each and no synthetic answer/visibility/length issues. Found a combined-data caveat: `173` official `equation_symbol_transformation` rows in `train_plus_synthetic_v2.csv` contain `{` or `}` in the gold answer, which current boxed formatting/parsing cannot safely handle. Decision: temporarily filter these for the next clean v2 baseline; add them back in a later version after implementing brace-safe boxed formatting, stopping, extraction, and unescaping. |
| 2026-06-02 21:14:00 CST | Created filtered combined data `data/train_plus_synthetic_v2_no_brace_answers.csv`: removed 173 official `equation_symbol_transformation` brace-answer rows, kept 15,327 rows, and left 0 remaining answers containing `{` or `}`. Filter report: `data/train_plus_synthetic_v2_no_brace_answers_report.json`. |
| 2026-06-02 21:15:33 CST | Stopped the unfiltered v2 run `runs/synthetic_v2_response_only_ddp_10000/` by SIGTERM at optimizer step 29. It had not reached step-100 checkpoint/eval, so it should not be treated as a usable training result. |
| 2026-06-02 21:19:05 CST | Started clean no-brace v2 response-only DDP run on GPUs 5/6. Run root: `runs/synthetic_v2_no_brace_response_only_ddp_10000/`; torchrun PID file: `runs/synthetic_v2_no_brace_response_only_ddp_10000/synthetic_v2_no_brace_response_only_ddp_10000.pid`; W&B: `https://wandb.ai/yaozhonger7-shantou-university/llm-infer-sft/runs/di0xzgvq`; `RESUME_FROM_CHECKPOINT=(none)`. |
| 2026-06-02 21:21:32 CST | Confirmed no-brace v2 run is active: it loaded 15,327 rows, entered `[7/8] Training`, reached optimizer step 2, and GPUs 5/6 are both loaded with active compute. No early RuntimeError/OOM/`mha_graph` issue observed. First checkpoint/eval remains step 100. |
| 2026-06-03 10:17:43 CST | The no-brace v2 run has produced intermediate eval results. Latest finished eval is `eval_step-1800` with `50/100` accuracy and `100/100` boxed rate; `step-1600` is the current best checkpoint at `50/100`. Eval artifacts are under `runs/synthetic_v2_no_brace_response_only_ddp_10000/eval/`, and training is still ongoing. |
