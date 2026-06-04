# 官方信息汇总 — NVIDIA Nemotron Model Reasoning Challenge

> 本文件收录从官方来源(Kaggle 比赛页、HuggingFace 模型卡、NVIDIA 文档)整理的官方信息,
> 作为训练/评测对齐的事实基准。**标注 [已确认] 的来自官方页面;[待核实] 的来自搜索摘要,
> 需用 Kaggle 页面正文逐字核对后再依赖。** 采分相关的逐字规则见 `official_metric.py`。
>
> 最后更新:2026-06-03

## 来源

- 比赛主页:https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- 评分 notebook:https://www.kaggle.com/code/metric/nvidia-nemotron-metric (已逐字拷贝到 `official_metric.py`)
- 模型页(Kaggle):https://www.kaggle.com/models/metric/nemotron-3-nano-30b-a3b-bf16
- HF 模型卡:https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
- NVIDIA NIM 文档:https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-30b-a3b
- build.nvidia.com 模型卡:https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard

---

## 1. 比赛规则与时间线

- **提交物** [已确认]:LoRA adapter,**rank ≤ 32**,打包成 `submission.zip`,基于 Nemotron-3-Nano-30B base model。
  zip 内只放 `adapter_config.json` + `adapter_model.safetensors`,不要塞 base 权重。
- **工具自由** [已确认]:可用任意框架(HF / Unsloth / Axolotl / TRL 等),只要产出兼容的 LoRA adapter
  (必须含 `adapter_config.json`)。也可用 prompting / 数据筛选 / 合成数据 / RL / 轻量微调等任意方法。
- **评测基础设施** [已确认]:Google Cloud **G4 VM,搭载 NVIDIA RTX PRO 6000 Blackwell Server Edition GPU**
  (单卡约 96GB 显存)。**注意:Overview/Evaluation 页未明示单次评测的运行时间上限与每天提交次数**,
  仅给了推理参数(见 §3);如需硬上限仍要看 Code Requirements / Rules。
- **获奖资格** [已确认]:必须公开 Kaggle notebook + solution write-up(方法、数据、技术),否则不计奖。

### 时间线(UTC 23:59) [已确认]

| 事件 | 日期 |
| --- | --- |
| Start | 2026-03-16 |
| Midpoint Cut-off(Open Progress Prize 评定)| 2026-04-09(已过)|
| **Entry / Team Merger 截止** | **2026-06-08** |
| **最终提交截止 / 比赛结束** | **2026-06-15** |

> ⚠️ 今天 2026-06-03,距最终提交 12 天;**Entry 截止 6/8 仅剩 5 天**,需先在比赛页 accept rules 才能参赛。
> 组委会保留调整时间线的权利。

### 奖项 [已确认]

- **最终榜**:1st `$25,000 + 5×DGX Spark`;2nd `$15,000 + 2×DGX Spark`;3rd `$5,000 + 1×DGX Spark`
  (共 8 台 DGX Spark,按名次 cascade;每人最多 1 台)。
- **Open Progress Prize**(中期,4/9 榜首):`$5,000 + 1×DGX Spark`(已评定)。
- **Open Contribution Awards**(各 1×DGX Spark,**仅最终榜 top 10% 可参评**,需通过表单提交):
  - Best Data/Synthetic Data Method
  - Best RL Method
  - Best Fine-tuning Method
  > 我们走的是合成数据 + 微调路线,若进 top 10%,可冲 **Best Data/Synthetic** 与 **Best Fine-tuning** 两个分项。

### 引用 [已确认]

Jamil C Semaan, Jean-Francois Puget, Christof Henkel, Yi Dong, Addison Howard, Ashley Oldacre,
Ryan Holbrook, Chris Alexiuk, Rebecca Kao. *NVIDIA Nemotron Model Reasoning Challenge.* Kaggle, 2026.

---

## 2. 基础模型(Nemotron-3-Nano-30B-A3B)[已确认 / HF + NVIDIA]

- **架构**:Mamba2-Transformer 混合 MoE。30B 总参,**3.5B 激活**。23 个 Mamba-2 + MoE 层 + 6 个 Attention 层。
  - 推论:代码里 import `mamba_ssm` 即因此;LoRA target module 用官方
    `r".*\.(in_proj|out_proj|up_proj|down_proj)$"` 是正确的,不要换成 q/k/v_proj。
- **Chat template**:ChatML 风格,带 `<think>` 标签。
  - `enable_thinking=True`(默认):assistant 以 `<think>\n` 开启推理 trace。
  - `enable_thinking=False`:assistant 为空的 `<think></think>`,精度在难题上略降。
  - 也可在 system prompt 放 `/think` 指令开启思考。
  - 输出区分 `reasoning_content`(think 内)与 `content`(最终答案)。
- **Context length**:HF 默认 256k(262144),最高可扩到 1M。
- **License**:NVIDIA Nemotron Open Model License。
- **官方推荐推理参数**(注意:这是模型卡推荐,**不是比赛评测设置**):
  reasoning 任务 temp=1.0 / top_p=1.0;reasoning-off 用 greedy;reasoning 开启时 max_tokens 设高(~10k)。

---

## 3. 比赛评测设置(以 `official_metric.py` 与 metric notebook 为准)[已确认]

- **指标**:Accuracy = 答对 / 总数。
- **答案提取/判分**:见 `official_metric.py`(逐字拷贝)。boxed 优先,数值容差 `rel_tol=1e-2, abs_tol=1e-5`,二进制串严格比较。
- **推理设置**:vLLM,**greedy(temperature=0.0)**,`top_p=1.0`,`max_tokens=7680`,`max_model_len=8192`,
  `max_lora_rank=32`,`max_num_seqs=64`,`gpu_memory_utilization=0.85`。
- **推理 prompt**:**无 system prompt**,user 内容为
  `item.prompt + "\nPlease put your final answer inside \`\\boxed{}\`. For example: \`\\boxed{your answer}\`"`,
  经 chat template `add_generation_prompt=True, enable_thinking=True` 渲染。

> ⚠️ **关键差异**:HF 模型卡推荐 reasoning 用 temp=1.0,但**比赛评测固定 temp=0.0(greedy)+ enable_thinking=True**。
> 以比赛 metric notebook 为准——这是 greedy 配 thinking 的特意设定。

---

## 3.5 数据集(Data 页)[已确认 / 2026-06-03 复制比赛页正文]

数据集 = 一批逻辑推理谜题,需识别并应用底层 transformation rule,涵盖 bit manipulation、
代数方程等多个 domain。**数据集 License:CC BY 4.0(Attribution 4.0 International)。**

文件与字段:

- **`train.csv`**(3.07 MB):`id`、`prompt`(谜题描述,含 input-output 示例 + 待解实例)、`answer`(标准答案)。
- **`test.csv`**(样例 1.46 kB,仅 3 行):只有 `id`、`prompt`,**无 `answer`**。
  > ⭐ **提交评分时,这个样例 test.csv 会被替换成"several hundred problems"(数百题)的真实隐藏测试集。**

### 关键认知:in-context 规则归纳,不是记忆

每条 prompt **完全自包含**:先给同一隐藏规则下的若干 input→output 示例,再问一个待解实例。例:

```
In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers...
Here are some examples of input -> output:
01010001 -> 11011101
00001001 -> 01101101
...(共 8 例)
Now, determine the output for: 00110100
```
答案:`10010111`。

含义:

- 隐藏测试集是**新规则的新实例**,但模型**始终能在 prompt 里看到该规则的示例**。
- 因此真正要练的能力是 **in-context 规则归纳 + 精确计算**,而不是背特定规则。
  → 合成数据应覆盖"任意 family 下随机生成新规则 + 自带示例"的分布,提升归纳泛化,而非把训练里出现过的具体规则记下来。
- Data 页未明示隐藏测试是否引入**训练 6 族之外的新 family**;样例 test.csv 两行属于已知的
  bit_manipulation / text_decryption。**保守假设:同族新规则为主,但不排除新 family**,合成数据保持多样性更稳妥。

---

## 4. 与当前训练方针的冲突检查(2026-06-03)

- **LoRA target / rank**:与官方一致(rank ≤ 32,官方 module pattern)。✅
- **boxed 答案格式**:训练对齐评测 boxed 输出。✅
- **本地 eval prompt 与评测不一致** ⚠️(已在 `PROJECT_PROGRESS.md` 记录):
  本地训练/eval 仍用 SYSTEM_PROMPT、prefill `</think>`、HF `generate`,且未追加官方那句 boxed 指令;
  与官方「无 system prompt + 追加 boxed 指令 + enable_thinking=True」的分布不同,线上分数可能有偏差。
  → 待办:让本地 eval 路径逐字对齐官方 prompt 构造。

---

## 5. 仍缺、需从 Kaggle 页面正文补充的官方信息

> Kaggle 页面是 JS 渲染的,WebFetch 读不到正文,需手动复制粘贴。

1. ~~**Data 页**~~ — ✅ 已补(见 3.5)。仍有一个开放问题:隐藏测试是否含训练 6 族之外的新 family。
2. ~~**Evaluation 页**~~ — ✅ 已补(见 §3、§1)。仍缺:**单次评测运行时间上限、每天提交次数**(本页未明示)。
3. ~~**Overview 页**~~ — ✅ 已补(任务定义见 §3.5;允许方法见 §1)。
4. **Rules 页正文(法律条款)** — 外部数据是否允许、允许的预训练模型/许可、能否换 base model 的细则。
   (Overview 已说"任意框架/方法"和"基于 Nemotron-3-Nano-30B base",但正式 Rules 仍建议核对。)
