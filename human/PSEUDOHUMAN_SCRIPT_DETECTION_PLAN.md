# Human 数据“伪人 / 脚本刷取”检测指标方案

## 目标

在现有 `duplicate identity + RT + 重复题 + 方差` 规则之外，新增一套针对 **乱填、脚本化模板作答、伪随机作答、批量刷取** 的行为检测层。

这套方案不追求“宣称 100% 识别所有伪人”，而是追求：

1. **尽量清掉明显乱填/脚本数据**
2. **每个删除都有可解释证据**
3. **保留 review 层，避免单一指标误杀真实被试**

---

## 总体框架：三层拦截

### 第一层：硬删除（已有规则）

这些规则继续保留，作为基础清洗：

- 重复身份簇（`duplicate_identity_drop`）
- 结构异常（题数不对、重复题数不对、repeat gap 太小等）
- 全部答案相同（`all_same_response_flag`）
- 重复题严重不一致（`repeat_severe_count >= 2`）
- 可用 RT 比例过低（`low_rt_usable_flag`）
- `manual_fraud_session_ids.csv` 人工硬删除

### 第二层：行为异常 soft / hard flags（新增）

核心思想是把“伪人/脚本”分成三类来检验：

1. **周期/模板型**
2. **伪随机型**
3. **批量刷取型**

每一类先产生若干指标，再转成 flag。

### 第三层：综合 ultra-clean 拦截（新增）

新增一个更严格的导出层：

- `suspicious_participants_review.csv`：所有命中行为异常的人，供人工复核
- `participants_ultra_cleaned.csv`：在 `final_clean` 基础上，再扣掉行为异常明显者
- `final_ultra_clean_wide.csv / final_ultra_clean_long.csv`：ultra-clean 终版

判定逻辑：

- 命中 **行为 hard flag**，直接进入 ultra-clean 排除
- 或者 **行为 soft flag 数量 >= 2**，也进入 ultra-clean 排除

---

## 第二层细分：三类“伪人 / 脚本”检测

## A. 周期 / 模板型作答

这类被试不是随机乱点，而是按某种机械模板填写，例如：

- `0121012101210`
- `0123456543210`
- 固定 2-8 长度循环
- 明显的上升、下降、锯齿式轨迹

### A1. 最佳周期匹配率

字段：

- `sequence_best_cycle_lag`
- `sequence_best_cycle_match_rate`

做法：

- 在 non-repeat 75 题上，检查 lag = 2 到 8
- 计算“当前题答案是否等于 lag 题之前答案”的匹配比例
- 取最佳 lag 与最佳匹配率

解释：

- 若某人按固定模板循环填，匹配率会异常高

对应 flag：

- `pattern_cycle_flag`
- `pattern_cycle_hard_flag`

### A2. 主导 bigram 比例

字段：

- `sequence_dominant_bigram_prop`

做法：

- 统计相邻答案对（bigram）中，最常见那一种占全部 bigram 的比例

解释：

- 如果答案不断重复相同的短跳转，比如 `0->1`, `1->2`, `2->1`，会导致 bigram 高度集中

对应 flag：

- `pattern_bigram_flag`

### A3. 低复杂度（Lempel-Ziv complexity）

字段：

- `sequence_lz_complexity`
- `sequence_lz_complexity_norm`

做法：

- 把整条作答序列离散化后计算 LZ complexity
- 再除以 trial 数得到归一化复杂度

解释：

- 脚本模板通常“压缩率高、复杂度低”
- 真实被试虽然也可能有短模式，但整条序列通常不会过于机械

对应 flag：

- `pattern_low_complexity_flag`

### A4. 长单调轨迹比例

字段：

- `sequence_longest_monotonic_run`
- `sequence_longest_monotonic_run_prop`

做法：

- 计算最长连续单调段（持续升、持续降，允许局部相等）

解释：

- 对无关句子逐题评分时，出现很长的“连续上升/连续下降”通常不自然
- 这种指标对 `0123456543210` 一类模板很有用

对应 flag：

- `pattern_template_walk_flag`

---

## B. 伪随机型作答

这类被试看起来不像固定模板，但其实像“机器随机填”：

- 分布很散
- 过于均匀
- 与句子内容几乎无关

### B1. 响应熵

字段：

- `response_entropy_norm`

做法：

- 计算被试答案分布的归一化熵

解释：

- 伪随机往往会把不同选项铺得更均匀，熵更高
- 这个指标单独用不够安全，但和下一条结合后很有力

### B2. leave-one-out item consensus correlation

字段：

- `leave_one_out_item_consensus_corr`

做法：

- 对每个 `condition + template_id`，先算“去掉当前被试后”的其他人均值
- 再计算该被试自己的逐题评分，与这个 leave-one-out item mean 之间的相关

解释：

- 真正认真作答的人，不一定和大众完全一致，但通常和句子内容结构有一定相关
- 纯随机或伪随机被试，和 item consensus 的相关往往异常低，甚至接近 0 或负相关

对应 flag：

- `pseudo_random_flag`
- `pseudo_random_hard_flag`

当前实现的逻辑是：

- **低 consensus correlation + 高熵** → `pseudo_random_flag`
- **极低 consensus correlation + 高熵** → `pseudo_random_hard_flag`

---

## C. 批量刷取 / 复制型作答

这类不是单个人自己乱填，而是一批人答案互相很像，像脚本批量生成、复制粘贴，或同一套行为模板。

### C1. 最大同伴相关

字段：

- `peer_max_correlation`

做法：

- 在同一 condition 内，对每个被试与其他人按 `template_id` 对齐
- 计算逐题评分相关
- 取该被试与任一同伴的最大相关

对应 flag：

- `batch_high_similarity_flag`

### C2. 最大精确匹配比例

字段：

- `peer_max_exact_match_prop`

做法：

- 在同一 condition 内，对齐同一模板后，计算两个被试逐题答案完全一致的比例
- 取最大值

对应 flag：

- `batch_exact_match_flag`
- `script_clone_hard_flag`

解释：

- 如果两人题目顺序一致、答案高度一致，尤其在 75 题上大面积完全相同，就很可疑

### C3. 高相似邻居数

字段：

- `peer_high_similarity_neighbor_count`

做法：

- 在同一 condition 内，若两人相关或精确匹配比例高于本 condition 的极端阈值，则记为“高相似邻居”
- 统计每个被试有多少这样的邻居

对应 flag：

- `batch_cluster_flag`

解释：

- 单独和一个人像，可能只是巧合
- 如果和多个被试都异常像，就更接近“批量刷取簇”

---

## 阈值思想

本方案尽量避免用死阈值一刀切，而是采用：

- **绝对阈值 floor**（防止阈值太松）
- **condition 内分位数阈值**（适应 3pt / 7pt / slider 三种量表）

例如：

- 周期匹配率：有固定软/硬阈值，也参考极端高分位
- 低复杂度：用低分位数
- 高熵：用高分位数
- 低 consensus correlation：用低分位数
- 高 peer similarity：用极端高分位数 + floor

这样可以兼顾：

- 不同量表尺度差异
- 数据自身分布
- 可解释性

---

## 当前代码里新增的核心字段

### 序列型

- `sequence_best_cycle_lag`
- `sequence_best_cycle_match_rate`
- `sequence_dominant_bigram_prop`
- `sequence_lz_complexity`
- `sequence_lz_complexity_norm`
- `sequence_longest_monotonic_run`
- `sequence_longest_monotonic_run_prop`
- `response_entropy_norm`

### 伪随机 / item 一致性

- `leave_one_out_item_consensus_corr`

### 批量刷取

- `peer_max_correlation`
- `peer_max_exact_match_prop`
- `peer_high_similarity_neighbor_count`
- `peer_similarity_corr_threshold`
- `peer_similarity_exact_match_threshold`

### 行为 flag

- `pattern_cycle_flag`
- `pattern_cycle_hard_flag`
- `pattern_bigram_flag`
- `pattern_low_complexity_flag`
- `pattern_template_walk_flag`
- `pseudo_random_flag`
- `pseudo_random_hard_flag`
- `batch_high_similarity_flag`
- `batch_exact_match_flag`
- `batch_cluster_flag`
- `script_clone_hard_flag`
- `bot_behavior_soft_flag_count`
- `bot_behavior_hard_flag`
- `suspicious_behavior_review_flag`
- `ultra_clean_additional_exclusion_flag`

---

## 输出文件设计

运行：

```bash
cd /home/sigurd/.openclaw/workspaces/engineer/projects/hate/human
uv run python clean_human_data.py --output-dir outputs_ultraclean_validation
```

新增输出：

- `participant_qc_master.csv`
  - 现在会包含全部行为检测指标和 flags
- `suspicious_participants_review.csv`
  - 所有命中行为 review flag 的被试
- `participants_ultra_cleaned.csv`
  - ultra-clean 版本的 participant 表
- `final_ultra_clean_wide.csv`
- `final_ultra_clean_wide.xlsx`
- `final_ultra_clean_long.csv`
- `final_ultra_clean_long.xlsx`

现有输出仍保留：

- `participants_hard_cleaned.csv`
- `participants_strict_cleaned.csv`
- `final_clean_wide.csv`
- `final_clean_long.csv`

也就是说，现在可以并行保留两套结果：

1. `final_clean_*`：原有主结果
2. `final_ultra_clean_*`：更严的 ultra-clean 结果

---

## 当前实现建议的使用方式

### 用于正式主分析

如果你希望结果更稳、更保守：

- 主文建议用 `final_clean_*`
- 敏感性分析 / robustness check 用 `final_ultra_clean_*`

### 用于组会展示

可以直接讲成一句话：

> 我们把清洗从“身份、RT、重复题、低方差”扩展到了“行为模式层”，新增了针对脚本模板、伪随机作答、批量刷取的检测模块，并输出了可人工复核的 suspicious 表与 ultra-clean 版本结果。

---

## 当前实现边界

这套规则已经能抓住：

- 固定循环模板
- 低复杂度模板
- 异常长单调轨迹
- 高熵且与 item consensus 几乎无关的伪随机作答
- 同条件内与别人过度相似的复制型作答

但也要明确：

1. **它不是 100% 真伪判别器**
2. 任何单一指标都不建议单独作为“删除依据”
3. 最稳的策略仍然是：
   - hard rule 删除明显异常
   - suspicious review 人工复核
   - ultra-clean 作为更严格版本做稳健性分析

---

## 本次代码改动文件

- `human/human_cleaning/config.py`
- `human/human_cleaning/pipeline.py`
- `human/clean_human_data.py`

---

## 验证状态

已在独立目录完成一次实际运行验证：

- 输出目录：`human/outputs_ultraclean_validation/`
- 成功生成：
  - `participant_qc_master.csv`
  - `suspicious_participants_review.csv`
  - `participants_ultra_cleaned.csv`
  - `final_ultra_clean_wide.csv`
  - `final_ultra_clean_long.csv`
  - 以及对应 xlsx 和 summary report

当前这版验证结果显示：

- `suspicious_behavior_review_flag` 命中：`77`
- `ultra_clean_additional_exclusion_flag` 命中：`18`
- `final_clean_wide.csv`：`667` 个 session
- `final_ultra_clean_wide.csv`：`665` 个 session

说明：

- 在当前数据上，这套新增行为检测是有实际作用的
- 但它并没有“粗暴地大面积删人”，而是相对克制地只切掉更可疑的一小部分
