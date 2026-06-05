# Nemotron SFT Project Progress

Last updated: 2026-06-05 18:58 CST

| Stage | Status | Evidence | Next action |
| --- | --- | --- | --- |
| M1 Official demo reproduction | Done | `output_adapter/adapter_config.json`, `output_adapter/adapter_model.safetensors`, and `submission.zip` exist. | Keep zip contents minimal. |
| M2 Tokenizer + SFT formatting | Done | `train_sft.py` loads tokenizer and prints formatted examples. | Keep prompt/answer format stable. |
| M3 Smoke training/package | Done | Smoke run on 2026-06-01 completed training, save, and zip packaging. | Scale up to a longer run. |
| M4 Local validation | Done | Smoke run evaluated 2 validation samples before and during training; generation no longer blocks. | Expand validation sample count for a broader check. |
| M5 Full SFT baseline | Archived | Earlier smoke/mini/baseline/rank test artifacts were removed on 2026-06-02 to avoid mixing old-code results with the current response-only/DDP runs. | No action unless a fresh non-synthetic baseline is needed. |
| M6 Synthetic data | Running | `synthetic_v1` produced 6,000 rows but later validation found solvability bugs. `synthetic_v2` fixed visibility/numeric consistency and produced the clean no-brace v2 DDP baseline through step-2600 eval (`48/100`, best observed step-1900 `51/100`, `100/100` boxed). `synthetic_v3` adds weak-family upweighting and brace-safe answer support; the active v3 run is now 5/6 DDP at `runs/synthetic_v3_ddp56_from_gpu6_latest/`, resumed from GPU 6 `checkpoint-000050`. | Let the v3 DDP run reach step-100 eval/checkpoint, then compare against the v2 no-brace baseline and full-val v2 best result. |
| M7 RLVR/GRPO | Not started | No RL pipeline found. | Defer. |
| M8 Solver-guided CoT v5 | Running | `data/train_plus_synthetic_v5.csv` has `19748/21000` CoT rows after verified imports from `extern/nemotron`; formal GPU3 run launched at `runs/cot_v5_gpu3_from_base_raw/`. | Monitor checkpoint-000050, then step-200 eval; do not resume old escaped-format checkpoints. |

## Planned bit/equation repair: solver-guided CoT v5 (2026-06-05 17:36 CST)

Current diagnosis: continuing v4 training alone is unlikely to fix `bit_manipulation` and `equation_symbol_transformation`, because their remaining errors are mostly rule-space mismatch and partial unobservability, not just insufficient SFT steps.

### Public references to inspect

Clone the Progress Prize / solver reference into the project workspace when available:

```bash
mkdir -p /data2/yaoxuran/llm_infer/extern
git clone --depth 1 https://github.com/tonghuikang/nemotron.git /data2/yaoxuran/llm_infer/extern/nemotron
```

If full clone is slow, use sparse checkout:

```bash
mkdir -p /data2/yaoxuran/llm_infer/extern
git clone --depth 1 --filter=blob:none --sparse https://github.com/tonghuikang/nemotron.git /data2/yaoxuran/llm_infer/extern/nemotron
cd /data2/yaoxuran/llm_infer/extern/nemotron
git sparse-checkout set reasoners
```

Files to inspect first:

- `reasoners/bit_manipulation.py`
- `reasoners/equation_numeric.py`
- any other `equation` / `transformation` reasoner
- any trace / CoT generation utilities

### Why current synthetic data is insufficient

`bit_manipulation`: current v3/v4 synthetic rules cover mostly whole-byte `xor`, `rotl_xor`, and `reverse_xor`. Public solver evidence suggests official bit tasks are often per-output-bit boolean rules over input-bit columns, including identity, NOT, constants, AND/OR/XOR, and mixed negated-operand forms. Therefore the current synthetic bit distribution covers only a narrow easy subset of official tasks.

`equation_symbol_transformation`: current numeric equation synthetic rules are limited to several two-digit operations such as `absdiff`, `sum_mod`, `prod_mod`, `cross_sum`, `swap_concat`, and `outer_inner_abs`. Public solver evidence suggests official numeric transformation tasks include richer operations: concat/reverse concat, addition/subtraction/absdiff, multiplication, +/-1, division/modulo, reversed operands/results, sign encoded by operator prefix/suffix, digit-level rules, and determinant-like rules. This explains length-changing gold answers such as `631` or `62`.

### v5 implementation plan

1. Port or wrap the reference bit/equation solvers locally instead of trusting free-form model reasoning.
2. Run the solvers on official `data/train.csv` bit/equation rows and produce an audit with:
   - total rows by family
   - uniquely solved rows
   - solver prediction equals gold
   - ambiguous rows
   - unobserved-symbol / insufficient-example rows
   - unsupported-rule rows
3. Generate CoT only for rows where `solver_pred == gold` and the rule is uniquely supported by visible examples.
4. Do not fabricate CoT for ambiguous or information-missing rows; that would leak gold answers and teach hallucinated reasoning.
5. Build `data/train_plus_synthetic_v5.csv` by combining:
   - official rows with solver-verified CoT where available
   - existing v4 CoT for already solved families
   - new synthetic bit/equation rows generated from the expanded solver rule space
6. Retrain from base with the raw-answer boxed format fix already in `train_sft.py`; do not resume from old escaped-format CoT checkpoints.
7. Only after raw-answer v5 SFT has nonzero bit/equation sampling success, run a small GRPO/RLVR pilot. Prefer GRPO over PPO because the task has a direct verifier and PPO adds value-model complexity. If GRPO groups have almost no correct samples, go back to solver/synthetic coverage rather than scaling RL.

Success criteria:

- measurable increase on fixed full v4/v5 validation for `bit_manipulation` and `equation_symbol_transformation`
- no regression on `gravity_formula`, `numeral_system`, and `unit_conversion`
- boxed rate remains near 100%
- answer strings containing `{`, `}`, or `\` are emitted raw and score correctly under the official metric

### v5 import result from `extern/nemotron` (2026-06-05 17:44 CST)

User cloned the Progress Prize repo to `extern/nemotron/`. It contains `reasoning/*.txt` for all 9,500 official train rows plus `problems.jsonl` with granular categories/status/submissions.

Added `scripts/build_v5_from_nemotron.py` and generated:

- `data/train_plus_synthetic_v5.csv`
- `data/train_plus_synthetic_v5_report.json`

Import policy:

- only official-train rows in `{bit_manipulation, equation_symbol_transformation, text_decryption}` are replaced/filled from `extern/nemotron`
- a trace is imported only when its final answer matches the local gold under `official_metric.verify()`
- all `\boxed` lines are stripped from imported CoT, so training still uses `train_sft.py`'s single canonical final answer target
- text-decryption traces are compacted by removing full Wonderland dictionary candidate enumeration and keeping the decrypt + best-match evidence

Coverage improvement:

| Family | v4 official CoT | v5 official CoT | Added/fixed source |
| --- | ---: | ---: | --- |
| `bit_manipulation` | `86/1602` | `1364/1602` | per-output-bit boolean solver (`I/NOT/C0/C1/AND/OR/XOR/*-NOT`) |
| `equation_symbol_transformation` | `46/1555` | `634/1555` | numeric + limited symbolic solver |
| `text_decryption` | `605/1576` | `1576/1576` | substitution + Wonderland dictionary solver |

Overall CoT coverage improved from v4 `16909/21000 = 0.805` to v5 `19748/21000 = 0.940`.

External verified imports by granular category:

- `bit_manipulation`: imported `1364`, rejected `238`
- `cipher`: imported `1576`
- `equation_numeric_deduce`: imported `541`, rejected `55`
- `equation_numeric_guess`: imported `21`, rejected `115`
- `cryptarithm_deduce`: imported `59`, rejected `600`
- `cryptarithm_guess`: imported `13`, rejected `151`

Length sanity after compaction:

- official bit CoT: p50 `243` chars, p99 `418`, max `427`
- official equation CoT: p50 `347`, p99 `643`, max `752`
- official text CoT: p50 `732`, p99 `5652`, max `8597` (still long for rare multi-unknown cases; training truncation must preserve useful tail/final answer)

Action taken: added tail-preserving CoT truncation in `train_sft.py`, smoke-tested v5 raw-answer training on GPU3, then launched the formal raw-answer CoT v5 run from base.

Formal run:

- Run dir: `runs/cot_v5_gpu3_from_base_raw/`
- PID: `3990225`
- Log: `runs/cot_v5_gpu3_from_base_raw/cot_v5_gpu3_from_base_raw.log`
- W&B: `https://wandb.ai/yaozhonger7-shantou-university/llm-infer-sft/runs/yzlj23lm`
- Train CSV: `data/train_plus_synthetic_v5.csv`
- Config: single GPU3, `MAX_SEQ_LEN=896`, `LORA_RANK=16`, `GRAD_ACCUM_STEPS=8`, response-only loss, raw answers inside `\boxed{...}`
- Checkpoints: every 50 optimizer steps under `runs/cot_v5_gpu3_from_base_raw/checkpoints/`
- Eval: every 200 optimizer steps on 48 validation samples under `runs/cot_v5_gpu3_from_base_raw/eval/`

Observed after launch: model loaded, LoRA attached (`440,069,120` trainable params), dataset prepared (`21000` rows, `19748` with CoT), and training entered epoch 0. Next action is to monitor first checkpoint at step 50 and first eval at step 200. Do **not** resume from old escaped-format CoT checkpoints.

## 官方评测标准 (Official Metric — 来自 Kaggle,请勿改动)

下面是 Kaggle "NVIDIA Nemotron Metric" notebook
(`https://www.kaggle.com/code/metric/nvidia-nemotron-metric`) 的官方评分规则。
这是排行榜实际使用的打分方式,**作为唯一基准,任何本地 eval 都必须与之逐字对齐,不要修改这套规则**。
本地副本已固化在 `official_metric.py`(逐字拷贝),并接入 `train_sft.py` 的 `evaluate()`。

指标:**Accuracy** = 答对题数 / 总题数。

答案提取 `extract_final_answer(text)`(优先级):

1. 取最后一个非空的 `\boxed{...}` 内容(每个 `\boxed{` 取到下一个 `\boxed{` 或文末之前的最后一个 `}`,以兼容答案里含 `}`)。
2. 无 boxed 时,匹配 `The final answer is: ...` / `Final answer is: ...` / `Final answer: ...`(大小写不敏感)。
3. 仍无,取文本中**最后一个数字** `-?\d+(?:\.\d+)?`。
4. 仍无,取最后一个非空行;文本为 `None` 返回 `NOT_FOUND`。

判分 `verify(gold, pred)`(先 `strip()`):

1. **gold 是二进制串**(完全匹配 `[01]+`)→ 严格字符串比较(`.lower()`)。这一条保护 `bit_manipulation`,避免把二进制当成数字落进容差。
2. 否则尝试 `float(gold)`、`float(pred)` → `math.isclose(gold, pred, rel_tol=1e-2, abs_tol=1e-5)`。**数值比较不分 family,对所有能转成浮点的答案都生效。**
3. 否则 case-insensitive 字符串比较(`pred.lower() == gold.lower()`)。

官方推理参数(`vLLM`,贪心):

| 参数 | 值 |
| --- | --- |
| max_lora_rank | 32 |
| max_tokens | 7680 |
| top_p | 1.0 |
| temperature | 0.0 |
| max_num_seqs | 64 |
| gpu_memory_utilization | 0.85 |
| max_model_len | 8192 |

官方推理 prompt(注意与本地训练格式的差异):**无 system prompt**,user 内容为
`item.prompt + "\nPlease put your final answer inside \`\\boxed{}\`. For example: \`\\boxed{your answer}\`"`,
经 chat template `add_generation_prompt=True, enable_thinking=True` 渲染。

本地对齐状态(2026-06-03 21:31 CST):

- 新增 `official_metric.py`:逐字拷贝官方 `extract_final_answer` 与 `verify`,纯 stdlib。**请勿在此文件加入本地启发式,唯一目的是与官方逐字一致。**
- `train_sft.py` 的 `evaluate()` 已改用官方 `extract_final_answer` + `verify` 打分;旧的 `numeric_equal(rel_tol=1e-3)` 与 `NUMERIC_FAMILIES` 不再参与打分(仅 `boxed_pred` 作为 boxed 覆盖率诊断保留)。
- 新增 `scripts/rescore_official.py`:用官方 metric 重算历史 `eval_*.jsonl`(基于已存的 `decoded`,无需重跑模型)。
- 与旧本地评分的关键差异:旧的 `rel_tol=1e-3` 比官方 `1e-2` 严 10 倍,且只对 `{gravity_formula, unit_conversion}` 做数值比较、字符串比较大小写敏感、boxed 缺失直接判 0,这些都在系统性低报。用官方 metric 重打分后,v3 DDP `step-100` 从本地 `38/100` 升到官方 `50/100`(`gravity` 由 `2/21` → `13/21`)。
- 尚未对齐项(后续单独处理):本地训练/eval 仍用 SYSTEM_PROMPT、prefill `</think>`、HF `generate` 且不追加官方那句 boxed 指令,与官方 prompt 分布不一致;线上分数可能与本地有偏差。

### 历史 eval 官方重算结果 (2026-06-03 21:31 CST)

用 `scripts/rescore_official.py` 基于已存 `decoded` 对历史 eval 重新打分(无需重跑模型)。

v2 no-brace run (`runs/synthetic_v2_no_brace_response_only_ddp_10000/`):**训练确有提升,旧本地 metric 系统性低报约 10 分。**

| 信号 | 官方 | 旧本地 |
| --- | --- | --- |
| 100 样本最佳 (step-2400) | `62/100 = 0.62` | `50/100 = 0.50` |
| 100 样本曲线 | step-100 `0.40` → step-2400 `0.62`(随训练上升) | 0.32 → 0.50 |
| **全量 val (1530 条) v2-best** | **`892/1530 = 0.583`** | `747/1530 = 0.488` |

全量 val 分 family(v2-best):`unit 259/259 (1.00)`、`numeral 217/257 (0.84)`、`gravity 156/259 (0.60)`、`text 138/257 (0.54)`、`bit 89/260 (0.34)`、`equation 33/238 (0.14)`。`gravity` 在旧 metric 下只有 ~0.10,纯属容差误杀;`equation`/`bit` 是真难。

v3 DDP run (`runs/synthetic_v3_ddp56_from_gpu6_latest/`):官方重算 step-100..500 = `0.50 / 0.53 / 0.53 / 0.49 / 0.46`,平到略降——从 v2-best 暖启已在天花板附近,未见增益。

重要:**v2 与 v3 的 100 样本 val 分布不同,绝对分不可直接比较。** v2 val 偏易(`numeral 22 + unit 17` 共 39 易题),v3 val 偏难(`bit 27 + equation 26 + gravity 21` 共 74 难题),这是 v3 看起来更低的主因。要对比模型须固定同一 val 集。

## CoT 诊断与标准推理生成 (2026-06-03 21:31 CST)

诊断:`equation`(0.14)与 `bit`(0.34)卡死的根因不是数据不可解,而是**训练把模型训成了"零推理一步出答案"**。

- eval 输出实测为 `</think>\nThe final answer is \boxed{...}`——think 块为空(还被 prefill `</think>` 强制跳过),模型对着 6~14 个例子直接蒙一串 bit / 符号。
- bit 要先辨识 `XOR 0xF7` / `rotl+xor` / `reverse+xor` 这类规则再逐位算;equation 要先建逐字符映射或识别运算符规则再套用。**无草稿空间 = 必败。** 而 `unit`/`numeral`/`gravity` 是近单步运算,一步出答案也能到 0.6~1.0——难易差异恰好就是"需不需要多步推理"。
- 这也与官方推理严重不一致:官方 `enable_thinking=True` + `max_tokens=7680`,**预期模型思考几千 token**。当前 SFT + eval prefill 等于关掉了模型的 reasoning。

成果(标准 CoT 生成器,确定性、teacher 级、自带答案自检):

- 新增 `cot_builders.py`:从 prompt 反解析 examples + query,用 metadata 的 `rule_payload` 生成「归纳规则 → 在样例上验算复现 → 应用到 query 并展示每步算式」的推理链。已实现 `bit_manipulation`(xor / rotl_xor / reverse_xor)与 `equation_symbol_transformation`(symbol_substitution / numeric_operator_rules)。**CoT 推导规则但不直接抄 payload,让模型学"从例子归纳"。**
- 新增 `scripts/preview_cot.py`:抽样某 family、生成 CoT、自检 CoT 复现 gold 并打印。bit + equation 各 600 条自检 **600/600 通过**。
- 训练目标拟改为 `<think>{cot}</think>\nThe final answer is \boxed{answer}.`;CoT 既可用于新生成数据,也可回填到现有 `data/train_plus_synthetic_v3.csv`。

## v4 数据集(CoT 回填)与 v3 可解性审计 (2026-06-03 21:31 CST)

新增 `scripts/build_v4_dataset.py`:对 `train_plus_synthetic_v3.csv` 每行**仅用 prompt 里的例子解题**(不依赖隐藏 payload)、生成 CoT、并用官方 `verify()` 自检 CoT 复现 gold。"能从自身例子解出且对上 gold"即可解性自检。

产物:

- `data/train_plus_synthetic_v4.csv`:v3 全列 + `cot` + `has_cot`,21000 行,其中 **16909 行(80.5%)带 CoT**。其余 `cot=""`、`has_cot=False`,保留为 answer-only(不丢数据)。
- `data/train_plus_synthetic_v4_report.json`:按 source×family 的审计明细。
- `cot_builders.py` 扩展为全 6 family,均支持 example-only 规则推断;numeric equation 增加**歧义检测**(与例子一致的规则若在 query 上不一致即判 ambiguous)。

### v3 可解性审计结论(= 抽查结果)

合成数据(synthetic_v3)自身质量:

| family | example-only 可解率 | 问题 |
| --- | --- | --- |
| bit / gravity / numeral / text / unit | 100% | 无 |
| equation | 97% | **95/3000 numeric 行歧义**:运算符在所给例子下不唯一可确定(`gen_numeric_equation` 只保证每个运算符出现一次,未保证唯一可解)→ v3 生成器待修 |

官方训练数据(official_train)—— **暴露合成与官方分布严重不匹配**:

| family | 可解率 | 含义 |
| --- | --- | --- |
| gravity / numeral / unit | 100% | 与合成同构,已解决 |
| text | 38% | 大量 query 字母从未在例子中出现(单替换无法解),官方任务本身部分不可解 |
| **bit** | **5%** | 官方规则多在我们的 xor/rotl/reverse 候选空间之外(AND/OR/NOT/majority/shift)。**合成 bit 只覆盖了真实分布的窄易子集。** |
| **equation** | **3%** | 官方规则更丰富、输出变长(gold 如 `631`/`62`),且大量 query 符号/运算符未在例子出现。合成 6 条 2 位规则远不能覆盖。 |

推论:`bit`/`equation` 长期卡死,除了"无 CoT"之外,**更深层原因是合成数据没有覆盖官方任务的真实规则空间**。后续要么扩充生成器规则空间以匹配官方,要么承认这两族官方样本部分不可解。

### 官方 bit 规则空间逆向(2026-06-03 21:31 CST)

为评估"扩充合成 bit 匹配官方"是否值得,对官方 bit 做了规则空间逆向:

| 模型 | 覆盖(预测==gold) |
| --- | --- |
| 位置换 + XOR(单输入位线性) | 18% |
| 完整 GF(2) 仿射(任意线性) | 27%,且"可解"行里也仅 52% 命中 |
| 旋转/反转/移位 ∘ AND/OR/XOR(17664 候选,含非线性) | 12% |

结论:**官方 bit 主要是多级非线性组合(majority/choice/组合),且 8 个样例常不足以唯一确定规则——连无限算力也解不出。** 现有线性合成 bit 几乎不覆盖官方分布,**扩充以匹配官方是低产出的深坑,已放弃该方向。** official `equation` 同理(规则更丰富、输出变长、符号常缺席)。

### 已修复:equation 数值歧义

`gen_numeric_equation` 现保证每个运算符的同号样例在 6 个候选规则里**唯一确定**(逐个加判别性样例直到唯一)。验证:修复后 equation example-only 自检 500/500,ambiguous `0`(原 ~95/3000);样例数均值 4.5、最多 8。注意:现有 `data/train_plus_synthetic_v3/v4.csv` 仍含旧的 95 条歧义行,需重生成才会清除。

待办(尚未执行,需确认后进行):

1. ~~扩充 bit/equation 规则空间~~ —— 已评估为低产出,放弃。改为承认 bit/equation 官方样本部分不可解,接受其较低天花板。
2. 改训练格式用上 v4 CoT:response-only loss 覆盖 think 段;**去掉 eval 的 `</think>` prefill,`enable_thinking=True`,放开 `max_new_tokens` 向官方 7680 靠**。这是当前最高确定性赢面,直接惠及可解族(gravity/numeral/unit/text 及 bit/equation 的可解子集)。
3. 注意序列变长对显存/`max_model_len=8192` 的影响。

## CoT 训练 + thinking-eval 对齐 (任务 2,2026-06-03 22:31 CST)

代码改动(均已 `py_compile` + GPU smoke 验证):

- `train_sft.py`:`format_answer_text(answer, cot)` 把 CoT 注入 `<think>` 块,训练目标变为 `<think>\n{cot}\n</think>\nThe final answer is \boxed{answer}.`;`format_training_parts`/`SFTDataset`/数据加载贯通 v4 的 `cot` 列(无 `cot` 列时回落 answer-only)。response-only loss 自然覆盖 think 段。
- chat template 实测:`add_generation_prompt=True` 渲染到 `<think>\n`(打开 think),故只需把 CoT 接在 `</think>` 之前;**推理侧 `INFERENCE_FINAL_ANSWER_PREFILL=0` 即去掉 `</think>` prefill,模型自然进入 thinking**,与官方 `enable_thinking=True` 对齐。
- `MAX_SEQ_LEN` 需设 1024:带 CoT 的训练序列 p99≈781、max≈884,旧的 512 会截断约 25%。

实践发现:**HF `generate` 的 thinking-mode eval 很慢**(单序列、未训练模型 rambling 到上限,实测 ~150s/样本;官方用 vLLM 批量)。故 eval 配置降本:`VAL_MAX_SAMPLES=48`、`EVAL_MAX_NEW_TOKENS=512`、`EVAL_EVERY_STEPS=200`、关闭 baseline eval。训练后模型出 boxed 即 `StopAfterBoxClose` 早停会更快。

smoke(GPU 2,v4 数据,thinking eval):格式正确(CoT 在 think 块内)、模型/LoRA 加载、thinking 生成正常、训练步 loss=0.94 有限。已验证通过并停止。

### 已停 v3,从 base 起训 CoT 版本

- 已 SIGTERM 停止 v3 DDP(原 PID 2273771),GPU 5/6 释放。
- 启动脚本:`scripts/launch_cot_v4_ddp.sh`。
- Run:`runs/cot_v4_ddp56_from_base/`,torchrun PID `2265360`,W&B `https://wandb.ai/yaozhonger7-shantou-university/llm-infer-sft`(run 名 `cot_v4_ddp56_from_base`)。
- 数据:`data/train_plus_synthetic_v4.csv`(16909/21000 行带 CoT)。
- 关键参数:`CUDA_VISIBLE_DEVICES=5,6`、`RESUME_FROM_CHECKPOINT=(空,从 base)`、`MAX_SEQ_LEN=1024`、`LORA_RANK=16`、`BATCH_SIZE=1`、`GRAD_ACCUM_STEPS=4`、`NUM_EPOCHS=3`、`MAX_TRAIN_STEPS=10000`、`TRAIN_FINAL_ANSWER_PREFILL=1`、`INFERENCE_FINAL_ANSWER_PREFILL=0`、`EVAL_EVERY_STEPS=200`、`EVAL_MAX_NEW_TOKENS=512`、`VAL_MAX_SAMPLES=48`、`CHECKPOINT_EVERY_STEPS=100`、`DISABLE_CUDNN_SDP=1`。
- 状态:已崩溃退出。主日志显示 rank0 在 `loss.backward()` 处 CUDA OOM,时间 `2026-06-03 23:06:04`,当时约 optimizer step 97、batch 391/9452,距首个 `checkpoint-000100` 还差 3 step;`checkpoints/` 和 `eval/` 均为空,没有可恢复/可评分产物。
- 直接原因:`MAX_SEQ_LEN=1024` 的 CoT 训练显存高于旧 `seq_len=512` 假设,而训练阶段仍关闭 gradient checkpointing。OOM 时本训练进程已占约 `77.93 GiB`,PyTorch allocated `75.85 GiB`,GPU 只剩 `494.75 MiB`,再申请 `462 MiB` 失败。日志里显示的 `GPU 0` 是 DDP rank0 的本地可见卡,对应 `CUDA_VISIBLE_DEVICES=5,6` 里的物理 GPU 5。
- 建议重跑:先给 CoT 长序列训练打开 gradient checkpointing(或降低 `MAX_SEQ_LEN`/LoRA rank),并把首个保存点临时降到 `CHECKPOINT_EVERY_STEPS=50` 以免再次在 step100 前无产物退出。
- 用户判断 OOM 可能是偶发现象,要求删除重算/省显存改动并重跑。已停止临时 GC 试跑 `runs/cot_v4_ddp56_from_base_gc/`(SIGTERM,约 step 12,无 checkpoint),恢复非 gradient-checkpointing 训练路径。
- 非 gradient-checkpointing 重跑 `runs/cot_v4_ddp56_from_base_retry/` 复现同一 OOM:2026-06-04 12:16 CST 仍在 rank0 `loss.backward()`、optimizer step 97、batch 391/9452 崩溃。不是偶发;同一 shuffle 顺序下 rank0 的 batch 391 是 `official_train` 的 `text_decryption` 样本,训练长度 `922` tokens、prompt `212` tokens、response/CoT label `710` tokens、CoT `1846` chars、answer `knight follows in garden`。训练集长度分布:p99 `775`,max `922`;仅 `text_decryption` 有 `>=850` 的长样本(21 条,`>=900` 的 3 条)。rank0 已在 batch 318 跑过 `853` tokens 样本,说明非重算 80GB 的临界点约在 900-token CoT 样本附近。
- 该重跑已保存可恢复点:`runs/cot_v4_ddp56_from_base_retry/checkpoints/checkpoint-000050/`。后续如继续非重算,必须降低峰值:缩短/截断极少数超长 CoT,降低 `MAX_SEQ_LEN`,或降低 LoRA rank;否则会在同一位置复现。临时 `retry_alloc` 验证 run 未进入训练,因为当时 GPU 5/6 被外部 `rwj` eval 进程各占约 19-20GB,加载阶段即 OOM,这与 step97 训练 OOM 是另一个资源冲突。
- 最小修复已改为训练阶段 `TRAIN_USE_CACHE=0`(默认),eval/generate 仍显式 `use_cache=True`;这不是 gradient checkpointing/重算。已启动 30 分钟 GPU 5/6 watcher,PID `3738651`,日志 `runs/cot_v4_ddp56_from_base_resume_nocache.watch.log`;当 GPU 5/6 无超过 2GB 的 compute 进程时,会自动从 `checkpoint-000050` 续跑到 `runs/cot_v4_ddp56_from_base_resume_nocache/`。当前等待外部进程:GPU5 `2230459`约 4.8GB,GPU6 `2221794`约 21GB。
- 按“固定 max_seq_len 截断”方案处理长样本:修改 `SFTDataset` 的超长样本路径,保留 prompt 和最终 `</think>...\\boxed{answer}` 尾部,只截断中间 CoT。验证最长 `train_idx=784` 从 `922` tokens 被压到 `896`,且 `\\boxed{knight follows in garden}` 仍保留。未启用 gradient checkpointing/重算。
- 直接用 GPU5 跑 `runs/cot_v4_gpu5_resume_trunc896/` 仍在 batch 203 OOM,但当时 GPU5 还有外部 PID `2230459` 占 `4.66GB`,训练只差 `20MB`;同位置样本长度仅 `243` tokens,不是长 CoT 本身。GPU5 当前资源冲突不适合硬跑 896。
- 已按用户要求切到 GPU3 单卡续训:`runs/cot_v4_gpu3_resume_trunc896/`,PID `2525161`,W&B `https://wandb.ai/yaozhonger7-shantou-university/llm-infer-sft/runs/ec3arh9x`,从 `checkpoint-000050` 续训,`MAX_SEQ_LEN=896`,`GRAD_ACCUM_STEPS=8`,`TRAIN_USE_CACHE=0`,GPU3 仅有小常驻进程。已越过 GPU5 崩溃点 batch 203,观察到 batch 300+ 正常训练;单卡顺序中 922-token 样本位于 batch 782。该旧 escaped-format v4 训练后来已停止,不要再继续它;当前正式方向是 raw-answer v5。
- 重要:这是**首个 thinking 模式 + 官方 metric 对齐**的 run,其 eval 数才是与排行榜可比的真实基线;不要再拿它和历史 skip-thinking eval 直接比。

## boxed 转义与官方 metric 对齐修复 + 全量 eval (2026-06-05 16:58 CST)

### 根因复核(已用实测样本证实)

CoT run step-1000 eval 里有一类"模型其实算对了却判错"的样本,根因是**训练目标对答案做了转义**:`escape_boxed_answer` 把答案里的 `\ { }` 写成 `\\ \{ \}` 再放进 `\boxed{}`。模型忠实学会输出转义形态,但 `evaluate()` 用的是**官方 `official_metric.extract_final_answer`,它不做 unescape**,直接按字面比对 → 判错。

- 实测样本:gold `\![<_`,模型 decoded 推理过程正确(`... to get '\![<_'`),但 `\boxed{\\![<_}` 是双反斜杠 → 官方提取 `\\![<_` ≠ gold。**线上排行榜用的就是官方 metric,这分是真丢的。**
- `official_metric.py` 的 docstring 明确写了官方期望**原样字面**写法(`the model writes them literally, producing \boxed{}52} for "}52"`)——转义方案与官方设计直接冲突。

### 修复(`train_sft.py`,已 `py_compile` + 单测验证)

- **去掉转义**:`format_answer_text`、`format_example` 现在把**原始答案**写进 `\boxed{...}`。模型输出的字符 = 官方提取回来的字符,逐字一致。
- **本地解析对齐官方**:删除旧的 `escape_boxed_answer`/`unescape_boxed_answer`/`find_boxed_spans`(基于括号深度且把 `\` 当转义符,会破坏原始答案)。新增 `_boxed_spans_official`,与官方"每个 `\boxed{` 取到下一个 `\boxed{` 或文末前最后一个 `}`"完全同构;`extract_boxed`(仅用作 boxed 覆盖率诊断)改用它且不 unescape。
- **新终止符**:生成早停 `StopAfterBoxClose` 与 `truncate_after_first_boxed` 改用 `}.` 作为答案终止符(训练目标恒以 `}.` 收尾)。**全数据集 0 条答案包含子串 `}.`**,故对含不平衡括号的答案(如 `+}`、`{17`)也安全;旧的括号深度扫描会把这些答案截断。
- `test_boxed_answer.py` 重写,断言原始答案(含 `{ } \` 与不平衡括号)经 `\boxed{ans}.` 后能被**官方 metric** 逐字取回;5 个用例全过。

### 量化依据(`data/train_plus_synthetic_v4.csv`,21000 行)

- 含 `{` 或 `}` 的答案:**548**;含 `\` 的:**287**(几乎全在 equation 族)。
- 原始答案 `\boxed{ans}.` → **官方提取器:0 处不一致**(完美往返);→ 旧的括号深度首闭合:**527 处被破坏**。证实"原始写法 + 官方贪婪提取"才是对的。

### 影响与后续

- **现有 CoT 模型/checkpoint 是用旧(转义)格式训练的**,仍会输出转义字符;此修复**只惠及之后重训/续训的 run**。建议用修复后的代码重训(或从 base 重训 CoT v4),以拿回 equation 及含括号/反斜杠答案的可解子集分数。
- equation/bit 的官方天花板(可解率 3%/5%)不变;此修复只捞回"算对了被转义判错"的那部分,主要落在 equation 可解子集。

### 全量 official eval(进行中)

为确立真实基线(48 样本 val 噪声太大、且偏难),对 **best CoT checkpoint = `checkpoints/checkpoint-001000`(step-1000,48-val `0.5208` 为该 run 最佳)** 跑全量 official eval:

- 配置:`ADAPTER_DIR=.../checkpoints/checkpoint-001000`、`TRAIN_CSV=data/train_plus_synthetic_v4.csv`、`VAL_MAX_SAMPLES=0`(全量,**v4 全 val = 2097 条**,注意不是 v2 的 1530)、`INFERENCE_FINAL_ANSWER_PREFILL=0`(thinking 模式)、`EVAL_MAX_NEW_TOKENS=1024`、`INFERENCE_STOP_AFTER_BOXED=1`、官方 metric。
- **2 卡数据并行**(应用户要求用 GPU5+6 提速):`eval_adapter.py` 新增 `NUM_SHARDS`/`SHARD_INDEX`,用 `gather_every` 把 val 交错二分。为腾出 GPU5,先 SIGTERM 了 GPU5 上另一 CoT 训练 `cot_v4_gpu5_resume_trunc896`(PID 1383983,step 479,**可从 `checkpoint-000450` 续训**)。
  - shard0:GPU5,PID `1636773`,1049 条,日志/结果 `eval/cot_step1000_fullval_official_s0.log` / `eval/eval_cot_step1000_fullval_official_s0_summary.json`。
  - shard1:GPU6,PID `1636775`,1048 条,日志/结果 `..._s1.log` / `..._s1_summary.json`。
- 预计 ~3–5h(2 卡并行;thinking 模式逐条生成,bit/equation 常跑满 token 预算)。两半跑完后合并两个 jsonl 计总分 + 分 family。**结果待回填。**
- 重要:此 eval 评的是**当前(转义格式)模型**,因此**包含转义罚分**,是当前真实基线;转义修复的增益要等用新代码重训后才体现。

## Active Issue

The format issue has mostly been fixed: response-only loss, final-answer prefill, and stop-after-boxed generation moved boxed rate to `100/100` on both the old adapter fixed-format eval and the current step-100 eval. The active issue is now data quality and task solvability. `synthetic_v1` contains a systematic `equation_symbol_transformation` bug where many final-query symbols are never shown in the example inputs, making the mapping impossible to infer. `synthetic_v2` fixes this and also aligns displayed numeric inputs with computed numeric answers.

Current recommendation: keep `data/train_plus_synthetic_v2_no_brace_answers.csv` results as the clean v2 baseline, and use `data/train_plus_synthetic_v3.csv` for new brace-safe experiments. The live v1 run has been stopped after preserving step-100 artifacts; use its results only as a short trend probe and do not treat v1 `equation_symbol_transformation` scores as a clean undertraining signal.

Brace-answer handling note: `train_sft.py` now escapes answers before writing them into `\boxed{...}`, stops only after the first complete boxed span, and unescapes the parsed answer before scoring. This restores support for gold answers containing `{`, `}`, or `\`. The filtered v2 no-brace data remains useful only as a historical clean baseline; new runs can use v3's full brace-answer coverage.

Latest v2 baseline run:

- Run root: `runs/synthetic_v2_no_brace_response_only_ddp_10000/`
- Log: `runs/synthetic_v2_no_brace_response_only_ddp_10000/synthetic_v2_no_brace_response_only_ddp_10000.log`
- PID file: `runs/synthetic_v2_no_brace_response_only_ddp_10000/synthetic_v2_no_brace_response_only_ddp_10000.pid`
- W&B: `https://wandb.ai/yaozhonger7-shantou-university/llm-infer-sft/runs/di0xzgvq`
- Init adapter: none; this run starts clean from the base model with a new LoRA adapter.
- Output adapter: `runs/synthetic_v2_no_brace_response_only_ddp_10000/adapter`
- Checkpoints: `runs/synthetic_v2_no_brace_response_only_ddp_10000/checkpoints`, saved every 100 optimizer steps, keeping last 10.
- Eval outputs: `runs/synthetic_v2_no_brace_response_only_ddp_10000/eval`
- Main parameters: `MAX_TRAIN_STEPS=10000`, `NUM_EPOCHS=12`, `BATCH_SIZE=2`, `GRAD_ACCUM_STEPS=4`, `LORA_RANK=16`, `VAL_MAX_SAMPLES=100`, `EVAL_EVERY_STEPS=100`, `DISABLE_CUDNN_SDP=1`.

Note: the no-brace v2 run intentionally did not resume from v1 adapters/checkpoints or the stopped unfiltered v2 attempt, because those data/run states were not clean baselines. New checkpoints from this run include optimizer/scheduler/global-step state and can be resumed with `RESUME_FROM_CHECKPOINT=latest`. Latest finished evals are saved under `runs/synthetic_v2_no_brace_response_only_ddp_10000/eval/`; best observed eval is `step-1900` with `51/100` accuracy and `100/100` boxed rate, while the latest observed eval is `step-2600` with `48/100` and `100/100` boxed. The log later shows a SIGTERM after step ~2620.

Current v3 training run:

- Pre-DDP single-GPU warm-start: `runs/synthetic_v3_gpu6_v2best_ckpt50/`; stopped after saving `checkpoints/checkpoint-000050`; W&B `https://wandb.ai/yaozhonger7-shantou-university/llm-infer-sft/runs/upzx02so`.
- Active DDP run: `runs/synthetic_v3_ddp56_from_gpu6_latest/`
- Torchrun PID: `2273771`; workers observed on GPUs 5/6: `2273996`, `2273997`
- PID file: `runs/synthetic_v3_ddp56_from_gpu6_latest/synthetic_v3_ddp56_from_gpu6_latest.pid`
- Log: `runs/synthetic_v3_ddp56_from_gpu6_latest/synthetic_v3_ddp56_from_gpu6_latest.log`
- W&B: `https://wandb.ai/yaozhonger7-shantou-university/llm-infer-sft/runs/6m0vixz5`
- Init adapter: `runs/synthetic_v2_no_brace_response_only_ddp_10000/adapter_best`, then resumed from `runs/synthetic_v3_gpu6_v2best_ckpt50/checkpoints/checkpoint-000050`
- Data: `data/train_plus_synthetic_v3.csv`
- Main parameters: `CUDA_VISIBLE_DEVICES=5,6`, `BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=4`, `NUM_EPOCHS=3`, `MAX_TRAIN_STEPS=10000`, `CHECKPOINT_EVERY_STEPS=50`, `KEEP_LAST_CHECKPOINTS=20`, `VAL_MAX_SAMPLES=100`, `EVAL_EVERY_STEPS=100`, `DISABLE_CUDNN_SDP=1`.

Switch status at 2026-06-03 17:35 CST: full-val v2 best eval completed with `747/1530 = 0.4882` accuracy and `1528/1530 = 0.9987` boxed rate. `scripts/watch_v3_ddp_after_eval.sh` stopped the single-GPU run and launched DDP. The first DDP resume attempt exposed a CUDA RNG restore device mismatch; `train_sft.py` now restores CUDA RNG states as CPU ByteTensors and tolerates incompatible optimizer/scheduler state by continuing with fresh optimizer/scheduler state. The relaunched DDP run resumed from `epoch=0`, `next_batch_step=200`, `global_step=50`, and was observed training at `step=59` with both GPUs loaded.

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
- v3 intentionally restores brace-answer coverage: `train_plus_synthetic_v3.csv` has `548` equation answers containing `{` or `}` and `783` answers containing `{`, `}`, or `\`. Use it only with the updated brace-safe `train_sft.py`; do not train it with older code.
- Brace-safe regression coverage: `test_boxed_answer.py` checks escaping/unescaping, internal escaped braces, truncation after the first complete boxed span, and ignoring unclosed boxes.
- GPU smoke coverage: `runs/brace_safe_v3_smoke_gpu6_retry/` ran 2 optimizer steps on GPU 6 from `runs/synthetic_v2_no_brace_response_only_ddp_10000/adapter_best`, using `data/train_plus_synthetic_v3.csv`. It saved checkpoints, final adapter, `submission.zip`, and eval details; eval was `0/2` accuracy with `2/2` boxed rate, sufficient only as a path check.

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

- Fixed eval inference now pre-fills `</think>\nThe final answer is \boxed{` and stops after the first complete boxed span. On the old adapter this raised boxed rate from `11/100` to `100/100` and accuracy from `11/100` to `29/100`.
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
- Known combined-data caveat: `train_plus_synthetic_v2.csv` has `173` official `equation_symbol_transformation` answers containing `{` or `}` (`158` in train, `15` in val for the current split). The current v2 baseline uses `train_plus_synthetic_v2_no_brace_answers.csv`, which removes these 173 rows and keeps 15,327 rows. Brace-safe parsing/formatting has since been added in `train_sft.py`; use v3 for restored brace-answer coverage.

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
| 2026-06-03 16:17:12 CST | Fixed the stale brace-answer project notes and added `test_boxed_answer.py` to lock brace-safe boxed escaping/parsing. Verified `py_compile` and `python -m unittest test_boxed_answer.py`. A previous v3 2-GPU warm-start attempt at `runs/synthetic_v3_warmstart_v2best_600/` OOMed early; a corrected single-GPU 6 smoke at `runs/brace_safe_v3_smoke_gpu6_retry/` completed 2 optimizer steps, checkpointing, eval, adapter save, and `submission.zip` packaging with `2/2` boxed rate. GPU 6 is free again after the smoke. |
| 2026-06-03 16:25:45 CST | Started the requested v3 training on GPU 6: `runs/synthetic_v3_gpu6_v2best_ckpt50/`, PID `2486933`, W&B `upzx02so`, warm-starting from v2 no-brace `adapter_best`, with checkpoint every 50 steps and eval every 100 steps. The run entered training and reached optimizer step 7 without OOM. Started watcher PID `2707788` to wait for GPU 5 full-val eval PID `1154476`, then stop the single-GPU run after a complete checkpoint and launch 5/6 DDP at `runs/synthetic_v3_ddp56_from_gpu6_latest/`. |
| 2026-06-05 16:58:00 CST | Fixed boxed/official-metric misalignment in `train_sft.py`: training now writes RAW answers into `\boxed{...}` (no `\\`/`\{`/`\}` escaping), and local extract/truncate/stop are aligned to the official greedy extractor with a `}.` terminator (no answer contains `}.`, so unbalanced-brace answers like `+}`/`{17` survive). Measured: raw target → official extractor 0/21000 mismatches vs 527 for the old depth scan; 548 answers contain braces, 287 contain backslash. Rewrote `test_boxed_answer.py` (5 tests pass) + `py_compile`. Existing CoT checkpoints were trained with the old escaped format, so the fix only benefits future re-training. |
| 2026-06-05 16:58:30 CST | Launched full-val official eval of best CoT checkpoint `checkpoints/checkpoint-001000` (step-1000, 48-val best 0.5208). Full v4 val = 2097 rows, thinking mode (`INFERENCE_FINAL_ANSWER_PREFILL=0`, `EVAL_MAX_NEW_TOKENS=1024`), official metric. Baselines the CURRENT (escaped) model so it includes the escaping penalty. |
| 2026-06-05 17:20:00 CST | Per request, switched the full-val eval to 2-GPU data parallel on GPU5+6. Added `NUM_SHARDS`/`SHARD_INDEX` (gather_every split) to `eval_adapter.py`. Stopped (SIGTERM) the other CoT training on GPU5 `cot_v4_gpu5_resume_trunc896` (PID 1383983, step 479, resumable from `checkpoint-000450`) to free the card. Eval now: shard0 GPU5 PID `1636773` (1049 rows), shard1 GPU6 PID `1636775` (1048 rows); merge both jsonl when done. ETA ~3-5h. The old escaped-format GPU3 training `cot_v4_gpu3_resume_trunc896` was later stopped; do not continue it. |
| 2026-06-05 18:58:00 CST | Launched formal raw-answer CoT v5 training from base on GPU3. Run root `runs/cot_v5_gpu3_from_base_raw/`, PID `3990225`, W&B `yzlj23lm`, train CSV `data/train_plus_synthetic_v5.csv`. Verified it loaded the model, attached LoRA, entered training, and reached optimizer step 2 with finite loss. First checkpoint is step 50; first eval is step 200. |
| 2026-06-03 17:35:24 CST | GPU 5 full-val v2 best eval completed: `747/1530 = 0.4882` accuracy, `1528/1530 = 0.9987` boxed. Watcher stopped the GPU 6 single run after `checkpoint-000050` and launched v3 DDP on GPUs 5/6 under `runs/synthetic_v3_ddp56_from_gpu6_latest/`. Patched `train_sft.py` to restore CUDA RNG states as CPU ByteTensors and tolerate optimizer/scheduler restore mismatch; relaunched DDP outside the sandbox. The active DDP run resumed at `global_step=50` and was observed training at `step=59`. |
