# v6 合成数据集计划 —— 对齐 `extern/nemotron/reasoners`

最后更新:2026-06-06 · 状态:**待审阅(OK 后开始生成)**

目标:让合成数据的**规则空间**和**CoT 格式**都与官方参考 solver(`extern/nemotron/reasoners/*.py`)对齐,
重点修复 `bit_manipulation` 与 `equation_symbol_transformation`,且不回退其它家族。

---

## 1. 当前 v5 数据缺陷分析(已实测)

数据源:`data/train_plus_synthetic_v5.csv`(21000 行 = 9500 官方 + 11500 synthetic_v3)。

### 1.1 CoT 覆盖空洞(按家族)

| 家族 | 有 CoT / 总数 | 空洞 | 说明 |
|---|---:|---:|---|
| gravity_formula | 4597 / 4597 | 0 | 健康 |
| numeral_system | 2076 / 2076 | 0 | 健康 |
| unit_conversion | 2094 / 2094 | 0 | 健康 |
| text_decryption | 3076 / 3076 | 0 | 健康(cipher solver 100% 解出) |
| **bit_manipulation** | 4364 / 4602 | **238** | 官方 238 行 solver 未解(`hypothesis_formed`/`rule_unknown`) |
| **equation_symbol_transformation** | 3541 / 4555 | **1014** | 最大空洞,见下 |

### 1.2 bit_manipulation 的"规则空间错配"(核心问题,非步数问题)

- 官方 bit 规则 = **逐输出位布尔函数**,语法见 `bit_manipulation.py`:
  `I{j}`(恒等)/ `NOT{j}` / `C0` `C1`(常量)/ 对称二元 `XOR/OR/AND{a}{b}` / 非对称二元 `XOR-NOT/OR-NOT/AND-NOT{a}{b}`(第二操作数取反),各位之间常有 **stride +1** 的旋转结构。
- 实测 1364 条官方 bit trace:
  - **1062 条(78%)** 至少用到一个**双输入位**运算(AND/OR/XOR/*-NOT)。
  - 仅 302 条(22%)只用 I/NOT/Constant。
  - 运算频率(按 trace):XOR 24% · AND 20% · XOR-NOT 16% · OR-NOT 15% · OR 10% · AND-NOT 7%。
- 而当前 3000 条 synthetic bit **100%** 是 `xor_*` / `rotl*_xor_*` / `reverse_xor_*`。
  代数展开后这三类**全部落在 `{I, NOT}` 子空间**(XOR-mask=逐位 I 或 NOT;rotl=移位后的 I/NOT;reverse=反序的 I/NOT),
  **从不**在两个不同输入位间做布尔运算。
- **结论:合成 bit 只覆盖官方 22% 的子空间,占主体的 78%(双输入位)一条没有。**

### 1.3 equation 的覆盖与缺陷

官方 `equation_symbol_transformation` = 两个官方子类:
- **数值类**(`equation_numeric_*`,732 行):`12-34 = ...` 形式,reasoner=`equation_numeric.py`。solver 解出 561/732(77%)。
- **符号类**(`cryptarithm_*`,823 行):`%|*"| = %|"|` 形式,reasoner=`cryptarithm.py`。solver 仅解出 **65/823(8%)**。

当前 synthetic equation 的 op 覆盖(实测 1664 条 numeric_operator_rules):
仅 6 种 digit-wise mod10 / swap-concat 运算(digit 乘法 mod10、交换拼接、abs digit diff、outer/inner abs diff、add mod10、cross-add mod10)。
对比 `equation_numeric.py` 的**完整 op 空间**(约 29 个基础 op + 反转组合 + 符号格式),缺失极多 → 见 §2.2。

### 1.4 已发现的 CoT 文本缺陷(来自参考 trace 本身,需在 v6 清洗)

1. **花括号尾巴**:`cryptarithm.py:160` 生成 `output: 【X】-> 【{X}】`,把答案包进 `{ }`,且是 CoT 最后一行。
   `verify("{X}","X")=False`,模型若模仿会判错。v5 里 72 条受影响。**v6 必须剥掉 `-> 【{...}】` 尾巴。**
2. **前导零丢失**:数值绝对差兜底时,gold=`03` 但 CoT 结论 `【3】`。v5 里 3 条。
   v6 合成时 gold = solver 输出(自洽),并对结果行按位数补齐,避免该问题。

> 已核验**正确**、无需改:1364 条官方 bit 逐位重组 == gold;synthetic bit 引用的样例真实存在、query 一致;
> synthetic equation 内部算术自洽。即:现有数据没有"答案错误",问题在**规则空间覆盖**与**少量格式瑕疵**。

---

## 2. 与 reasoners 对齐 —— 哪些 CoT 没被覆盖

七个 reasoner 与本地家族映射,以及"参考 solver 能解出的比例"(= 可用官方 CoT 上限):

| reasoner | 本地家族 | 官方行 | solver 解出 | 未覆盖 | 备注 |
|---|---|---:|---:|---:|---|
| `gravity.py` | gravity_formula | 1597 | 1597 | 0 | d=k·t²,长乘长除,完整 |
| `numeral.py` | numeral_system | 1576 | 1576 | 0 | Arabic→Roman,完整 |
| `unit_conversion.py` | unit_conversion | 1594 | 1594 | 0 | 线性 factor,完整 |
| `cipher.py` | text_decryption | 1576 | 1576 | 0 | 替换密码+Wonderland 词典,完整 |
| `bit_manipulation.py` | bit_manipulation | 1602 | 1364 | 238 | 规则语法完整,238 行多解/无解 |
| `equation_numeric.py` | equation(数值) | 732 | 561 | 171 | op 空间见 §2.2,完整 |
| `cryptarithm.py` | equation(符号) | 823 | **65** | **758** | **只支持 concatenation**,符号替换+算术不支持 |

**关键洞察:**
- **bit / 数值 equation / gravity / numeral / unit / cipher** 六类的 reasoner **规则空间完整**
  → 可以"采样规则 → 调 reasoner 生成 CoT",合成数据与官方**同格式同分布**。
- **符号 equation(cryptarithm)** 的 reasoner 只会拼接,758/823 官方行属于
  "符号经未知替换后再做选择/算术"——reasoner 标 `rule_unknown`,**无法用它生成 CoT**。
  这是真正难啃、且 reasoner 帮不上忙的部分。

### 2.1 `bit_manipulation.py` 完整规则空间(v6 bit 采样依据)
- 一元:`I{j}`,`NOT{j}`;常量:`C0`,`C1`
- 对称二元(两不同位):`XOR{a,b}`,`OR{a,b}`,`AND{a,b}`
- 非对称二元:`XOR-NOT{a,b}`,`OR-NOT{a,b}`,`AND-NOT{a,b}`(= op(a, ¬b))
- 跨位结构:solver 偏好 stride +1 的连续段(旋转式),并从左右两端外推填中间位。

### 2.2 `equation_numeric.py` 完整 op 空间(v6 numeric 采样依据)
- common:concatenation、reverse concatenation、addition、absolute difference、negated absolute difference、subtraction(a−b)、reverse subtraction(b−a)、multiplication
- rare:multiply±1、add±1、sub±1、max mod min、integer division(a/b)、modulo(a mod b)、reverse division(b/a)、reverse modulo(b mod a)
- 两位数 digit 级:digit abs diff、digit add/sub mod10、cross multiply(±rev)、digit multiply(±rev)、digit sum diff/sum、digit product diff/sum、determinant、abs determinant
- 正交修饰:`reversed operands` × `reversed result`(4 组合) + 符号格式 `prefix` / `neg_suffix` / `neg_prefix`(把符号当正负号编码,产生 `17/`、`-53`、`*53` 这类变长 gold)

### 2.3 `cryptarithm.py` 仅覆盖
- forward concatenation(A1A2B1B2)与 reverse concatenation(B1B2A1A2),按 operator 分组,未知 operator 默认 fwd。
- **不支持**:符号↔数字替换、符号选择/删除、符号算术 → 758 行官方符号题在此之外。

---

## 3. v6 生成计划

总思路:**"采样 ground-truth 规则 → 用样例喂 reasoner → reasoner 解出并验证 pred==gold → 复用 `build_v5` 的 `compact_reasoning` 压缩 → 入库"**。
这样合成 CoT 与导入的官方 CoT **逐字同格式**,且分布可控对齐官方。

新增脚本:`scripts/build_v6_synthetic.py`(import `extern.nemotron.reasoners.*`,复用 `official_metric.verify` 与 `build_v5_from_nemotron.compact_reasoning`)。

### 3.1 bit_manipulation(优先级最高)
- 采样一个**带 stride 结构的逐位规则向量**,family 按官方频率加权(XOR/AND/*-NOT 为主,I/NOT/C 兜底)。
- 随机 8–10 个 8-bit 输入算输出 → 构 `Problem` → 调 `reasoning_bit_manipulation`。
- 仅保留 solver 复原答案 == 采样 ground-truth 的行(自洽过滤;丢弃多解/退化样例后重采)。
- 目标量:**~3500 行**,替换现有全 XOR-mask 的 3000 行。验收:双输入位运算占比 ≈ 官方(~78%)。

### 3.2 equation —— 数值子类
- 遍历 §2.2 全 op × {正常/反操作数/反结果} × {无/prefix/suffix/neg 符号格式}。
- 每个 op 配比生成,**重点补**:multiplication、division/modulo、reversed 组合、符号正负号格式(变长 gold)。
- 调 `reasoning_equation_numeric`,`verify(gold,pred)` 通过才入库。
- 目标量:**~2500 行**(覆盖全 op,均衡配比)。

### 3.3 equation —— 符号子类(cryptarithm)

**结构已查清**(见 `investigators/cryptarithm_deduce.py`):`AB op CD = result`,
每个符号 = 唯一数字 0–9(实测 per-problem 操作数字母 ≤10,确认是数字替换),
operator 符号 = 算术运算(add / abs_diff / mul / concat / rev_concat,可扩 sub/rsub),
通过回溯搜索求"符号→数字 + operator→运算"的一致解,再套用到 query。

**官方行可解性实测(823 行,这是硬上限):**
| 方法 | 解对(==gold) | 自信但错 | 歧义/无解 |
|---|---:|---:|---:|
| 参考 solver(共识法,5 op) | 99 | 134 | 590 |
| 我的严格唯一解 solver(7 op) | 49 | 17 | 740 |

→ 官方符号题**本质欠定**(≤4 条样例 vs ~9–10 个未知数字),参考团队因此标 687 行 `rule_unknown`。
**诚实结论:官方符号行可验证生成的 CoT 很少,绝大多数无法在不泄漏答案的前提下给 CoT。**

**方案(已采纳"尝试推理 CoT",但坚持 gold 验证、绝不编造):**
- **官方行**:用我**严格唯一解**的本地 solver(零自信错误优先)求解,仅当答案被样例唯一确定且 `verify(gold,pred)==True` 才生成 CoT。
  实际**新增 30 行**(其余唯一可解行已在 v5 有 CoT;欠定行一律留空)。solver 支持一题多运算符(实测官方行 `108067c3` 含 `#`=mul/`]`=abs_diff/`|`=add 三运算符,全部一致推出并命中 gold)。
- **合成行(真正的覆盖增益)**:我**构造**唯一可解的算术 cryptarithm —— 先定一个完整符号→数字双射 + operator→运算,
  再生成足够多样例使映射被唯一确定,然后用 solver 跑出 CoT。可量产、自洽、教"符号替换+算术"推理。**实际 900 行**。
- **concatenation 子类**:用 `reasoning_cryptarithm` 生成 fwd/rev,**实际 400 行**(剥掉 `-> 【{X}】` 尾巴)。

### 3.4 CoT 清洗(对所有家族统一)
- 剥掉 `cryptarithm` 的 `-> 【{X}】` 花括号尾巴(全库统一,含 v5 遗留的 72 条官方,实测最终 0 条残留)。
- 数值前导零:合成行 gold=solver 输出,天然自洽;**官方导入行**仍残留 5 条(gold `03` vs CoT `【3】`),
  但 `verify("03","3")==True`(官方 metric 对前导零宽容),判分安全,未强改。
- 沿用 `clean_lines`:删 `\boxed` 行、`<|im_end|>`,并把残留 `\boxed`→`boxed`。

### 3.5 组装 v6
- 基线 = `data/train_plus_synthetic_v5.csv` 的**官方 9500 行原样保留**(含已验证官方 CoT)。
- 替换其中的 synthetic 部分:bit/equation 用上面新生成的;gravity/numeral/unit/text 的 synthetic 保留不动。
- 输出:`data/train_plus_synthetic_v6.csv` + `data/train_plus_synthetic_v6_report.json`。

---

## 4. 验收标准
- bit:synthetic 中"双输入位运算"trace 占比落在 70–80%(对齐官方),CoT 逐位重组 == gold 100%。
- equation 数值:覆盖 §2.2 全部 op,含变长 gold;`verify` 自洽率 100%。
- 全量:无 `-> 【{...}】` 花括号尾巴;无前导零不一致;boxed 率不受影响。
- 不回退 gravity / numeral / unit / text_decryption(行数与 CoT 不变)。
- 训练后固定验证集上 bit / equation 子项准确率上升,其余家族不降。

---

## 4b. 生成结果(2026-06-06 已完成)

产物:`data/train_plus_synthetic_v6.csv`(22300 行)· `data/train_plus_synthetic_v6_report.json`
· `data/synthetic_v6.csv`(7300 新合成)· `scripts/build_v6_synthetic.py`。

| 项目 | 结果 |
|---|---|
| 新合成 bit | 3500;逐位重组==gold **100%**;双输入位运算占比 **79%**(对齐官方 78%);9 个 family 全覆盖(XOR/OR/AND/XOR-NOT/OR-NOT/AND-NOT/mask/const) |
| 新合成 numeric eq | 2500;覆盖 `equation_numeric.py` 全 op(含乘除/反转/digit级/determinant);`verify` 自洽 100% |
| 新合成 symbolic concat | 400(fwd/rev);花括号尾巴已剥 |
| 新合成 symbolic arith | 900(构造唯一可解的符号→数字+算术);solver 验证 + gold 入 CoT 100% |
| 官方符号 verified 补 CoT | +30 行(严格唯一解 + gold 验证) |
| 花括号尾巴 | 全库 **0**(含修掉 v5 遗留的 72 条官方) |
| CoT 覆盖 | equation 3541→**4466**;bit 4364→**4864** |
| 回退检查 | gravity/numeral/unit/text 行数与 CoT **不变** |
| 已知残留 | 5 条官方数值前导零(metric 宽容,verify 通过);238 官方 bit + ~889 官方 equation 仍无 CoT(欠定题,诚实留空,未编造) |

## 5. 待你拍板的开放问题(生成时的取舍记录)
1. **符号 equation Tier B**(§3.3):保留现有 substitution 启发式生成器,还是暂时不合成该子空间?
2. **总量与配比**:bit ~3500 / 数值 equation ~2500 / 符号 concat ~800 是否合适?是否需要更大?
3. **是否替换** v5 里的 synthetic bit/equation(推荐替换),还是**追加**到 v5 之上?
4. 是否需要我同时把 v5 现存的 72 条花括号尾巴 + 3 条前导零**就地修一版 v5.1**,以便和 v6 做对照实验?
