# Training Log

最后更新:2026-06-06 14:01 CST。每个正式训练一行。结果一律用**官方 metric**(`official_metric.py`)。详细背景见 `PROJECT_PROGRESS.md`。

约定:
- **48-val** = 训练中 48 条噪声 val(快但不稳);**full-val** = 全量官方 val(947 条 official-only,或旧的 1530/2097 混合 val,口径已注明)。
- 最有意义的是 **vLLM + 官方配置(贪心, max_tokens=7680, 官方 prompt)** 的 full-val 分。

## 汇总表

| # | Run | 日期 | 数据 | 核心改动(vs 上一版) | 最佳结果(官方 metric) | 状态 |
|---|---|---|---|---|---|---|
| 1 | `synthetic_v1_13h` | 06-01 | v1 (15.5k) | 首个合成混合 SFT;整序列 loss、短 eval 预算 | 100-val **0.11**(修复 eval 后 0.29) | 停 |
| 2 | `synthetic_v1_response_only_ddp_10000` | 06-02 | v1 | **response-only loss** + final-answer prefill + stop-after-boxed;DDP | step-100 100-val **0.32**(boxed 100%) | 停(数据有 bug) |
| 3 | `synthetic_v2_no_brace_response_only_ddp_10000` | 06-02/03 | v2 no-brace (15.3k) | v2 修可解性(equation 可见映射、text 可见密文、数值显示一致);去 brace 答案 | **full-val 892/1530 = 0.583**;100-val best 0.62 | 停(基线) |
| 4 | `synthetic_v3_ddp56_from_gpu6_latest` | 06-03 | v3 (21k) | brace-safe boxed;弱族上采样(bit/equation/gravity ×3000);从 v2-best 暖启 | 100-val 0.50–0.53,**平台无增益** | 停 |
| 5 | `cot_v4_gpu3_resume_trunc896` | 06-03/05 | v4 (21k, 80.5% CoT) | **CoT 训练**(`<think>…</think>`)+ **thinking-mode eval**(去 prefill, `enable_thinking=True`);官方 metric 对齐;长样本尾部截断。**仍是转义答案格式** | 48-val best 0.52;**full-val(2097)0.590**(转义罚分内) | 停(转义格式弃用) |
| 6 | `cot_v5_gpu3_from_base_raw` | 06-05/06 | v5 (21k, **94% CoT**) | **raw 答案进 `\boxed{}`**(去转义,与官方逐字对齐);**官方 prompt 对齐**;从 nemotron 导入 bit/equation/text 已验证 CoT;从 base 重训 | **vLLM full-official(947, t7680)606/947 = 0.640**(boxed 0.937) | **训练中(~step1430, epoch0 61%)** |

## 当前最强基线明细 — Run #6 v5 step-1000

vLLM + 官方配置(贪心 / max_tokens=7680 / 官方 prompt+metric),全量 947 条 official-only val:

- **总 acc 0.640**(606/947),boxed 0.937,撞 token 上限仅 6.7%。
- 分 family:`unit 1.000` · `numeral 1.000` · `gravity 0.981` · `text 0.554` · `equation 0.174` · `bit 0.125`。
- **token 预算敏感**:同 checkpoint,7680 → 0.640,1024 → 0.517,更短 → 0.378(CoT 需草稿空间,本地短预算会系统性低估)。
- 瓶颈在 **bit / equation**——官方规则空间超出合成覆盖、8 例常不唯一可解,属结构性天花板,非欠训。
- 48-val 监控曲线(噪声大,仅供趋势)steps 200→1400:`0.396 / 0.479 / 0.500 / 0.458 / 0.542 / 0.521 / 0.542`,已进 0.50–0.54 平台。

## 关键经验(踩坑 & 定论)

- **本地 eval 必须逐字对齐官方 metric**:旧本地 `rel_tol=1e-3` + 只对部分族数值比较 + 大小写敏感,系统性低报约 10 分(v2 50→62)。
- **答案要 raw 写入 `\boxed{}`**,不要转义:官方提取器不 unescape,转义会把算对的判错(主要伤 equation/含括号答案)。终止符用 `}.`(全数据集无答案含此子串)。
- **CoT + thinking 推理是刚需**:bit/equation/text 多步任务,"零推理一步出答案"必败;官方 `enable_thinking=True` + 7680 token 就是要模型思考几千 token。
- **评测必须固定同一 val 集 + 同一 token 预算**才能比模型;v2/v3/v4/v5 的 val 口径不同,绝对分不可直接逐分比。
- **bit/equation 官方天花板低**(可解率 ~5% / ~3%),扩合成规则匹配官方已评估为低产出,放弃;接受其低分。
- **OOM 教训**:CoT 长序列(`MAX_SEQ_LEN`≥896)显存高,长 text_decryption 样本(~900 token)是临界点;用尾部截断保留 prompt + `\boxed{answer}.`,不开 gradient checkpointing。
