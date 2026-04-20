# 研究结果（Results）章节手稿

> **用途说明**：本文档是毕业论文 "自动仇恨言论检测中的性别群体不公平性：基于多语种大型语言模型 LLM 的公平化模型构建" 第四章《研究结果》的工作手稿。内容由三份原始材料重新组织：`slides/1a进展.pptx`（研究框架、模型与语料选取、1a 流水线）、`slides/1b_groupswap_bias_presentation.tex`（1b 配对语料构建逻辑与早期双模型结果）、`slides/results_1a_1b.tex`（当前 9 模型 × 3 尺度的最新结果）。所有 LaTeX 片段均可直接粘入毕业论文主文件；图表路径沿用 `artifacts/` 与 `outputs/group_swap_1b/.../` 相对目录。
>
> **整体定位**：本章对应研究一的两个子研究——研究一(a) 描述九个当代大语言模型在中英文自然语料上的仇恨识别总体表现，研究一(b) 在严格配对设计下检测同一语义框架下模型对女性指涉版本是否系统性赋予更高的攻击性评分。两者共同回答论文的第一个核心问题——**模型能力与性别群体公平性**——并为研究二（基于真人基准的 LoRA 微调对齐）提供失配诊断。

---

## 0. 本章总览

毕业论文第三章《研究方法》已详细给出模型选取、语料采集与推理流水线的前置说明；本章不再重复方法细节，而是以"假设—关键指标—结果"三元结构组织。为便于查阅，每一节在进入结果叙述之前均以简表的形式回顾设计参数。

```latex
\chapter{研究结果}
\label{chap:results}

本章分别报告研究一(a) 与研究一(b) 的主要发现。前者刻画九个当代大语言模型（LLM）在中英文仇恨言论识别任务上的总体表现；后者在严格配对设计下检验这些模型对"被指涉对象为女性"的语句是否系统性地给出更高的攻击性评分。两个子研究共用相同的模型集合与推理配置，但评估目标互不重叠：1a 关注分类准确度与概率校准，1b 关注同一语义框架下的性别不对称。
```

---

## 1. 研究一(a)：九个 LLM 的中英文仇恨识别表现

### 1.1 研究目的与竞争假设

研究一(a) 聚焦两个问题：
(i) 九个当代 LLM 在中英文自然语料上分类准确度（F1）与概率校准（Brier）的分布；
(ii) 跨语言差距究竟来自**任务定义／标注规范分歧**，还是来自**训练语料的语言覆盖不均衡**。

为将这两种解释置于可证伪的对照中，给出以下竞争假设：

- **H1——任务定义分歧假设**：若跨语言差距主要由中英标注规范差异引起，则同一模型在两种语言下的相对排名应当**稳定**，差距仅体现为整体分数的平移。
- **H2——语言覆盖不均假设**：若差距主要由训练语料的语言结构不均衡驱动，则模型排名会**跨语言翻转**——以中文为主的模型应当在中文面板领先，以英文为主的模型则相反。

两种假设的可证伪预测分别为"排名稳定"与"排名翻转"。

```latex
\section{研究一(a)：九个 LLM 的中英文仇恨识别表现}
\label{sec:results-1a}

\subsection{两个竞争假设}

在呈现结果之前，先区分两类可被现有数据证伪的假设。
\textbf{H1——任务定义分歧假设：}若模型表现的跨语言差距主要来自中英标注规范与文化边界的分歧，那么\emph{同一模型}在中英两个面板中的排名应当保持稳定，差距仅体现为整体分数的平移。
\textbf{H2——语言覆盖不均假设：}若差距主要来自模型训练语料的语言结构不均衡，那么模型排名会\emph{跨语言翻转}——在中文语料占优的模型在中文面板领先、以英文为主要训练语料的模型在英文面板领先。
两种假设给出的可区分预测分别是"排名稳定"与"排名翻转"，下文以此为判据。
```

### 1.2 设计参数速览（对应论文第三章方法）

| 维度 | 值 |
|---|---|
| 模型 | GPT-5.1、Claude Opus 4.5、Llama-4-Maverick、GLM-4.6、DeepSeek-R1-0528、DeepSeek-V3.2-Exp、Kimi-K2-thinking、Qwen-2.5-72B-Instruct、Gemma-4-31B-It |
| 中文语料 | COLDataset + ToxiCN 抽样 2,400 条，仇恨／非仇恨 1:1（`data/5_new_chinesehatedata_2400_balanced.xlsx`）|
| 英文语料 | HateXplain + HASOC 抽样 2,400 条，仇恨／非仇恨 1:1（`data/5_new_englishhatedata_2400_balanced.xlsx`）|
| 数据清洗 | 基于 TF-IDF 向量 + K-Means 聚类的标签一致性异常检测，剔除簇内极少数异类标签样本 |
| 提示策略 | Zero-shot、Chain-of-Thought（CoT）——中英文分别使用对应母语版本以避免隐式翻译偏移 |
| 评分量表 | 0–5 分离散等级（0 = 非仇恨，1–5 = 仇恨且强度递增）|
| 评估指标 | F1（\(\text{score} \geq 1\) 二值化）、Brier（\(p = \text{score}/5\) 均方误差）|
| 推理实现 | OpenRouter 统一 API，Temperature=0，10 样本/批线程池并行 |

### 1.3 总体表现与最佳模型的跨语言翻转

图~\ref{fig:1a-scatter-2x2} 将四个面板（zh/en × zero-shot/CoT）下九个模型的 F1 与 Brier 联合展示为二维散点；表~\ref{tab:1a-best-models} 汇总每个面板的最佳模型。

```latex
\subsection{总体表现与任务\texttimes{}语言交互}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{../artifacts/validated_csv_hate_f1_brier_2x2_scatter.pdf}
    \caption{九个 LLM 在四个面板（zh/en \texttimes{} zero-shot/CoT）下 F1 与 Brier 评分的联合分布。理想模型位于右下象限（高 F1、低 Brier）。}
    \label{fig:1a-scatter-2x2}
\end{figure}

\begin{table}[htbp]
    \centering
    \caption{每个面板的最佳 F1 模型与最佳 Brier 模型。数据来源：\texttt{artifacts/validated\_csv\_hate\_f1\_brier\_best\_models.csv}。}
    \label{tab:1a-best-models}
    \small
    \begin{tabular}{llccc}
        \toprule
        语言 & 提示策略 & 最佳 F1 模型（F1） & 最佳 Brier 模型（Brier） & 同一模型 \\
        \midrule
        中文 & zero-shot & Qwen-2.5-72B（0.751）  & Qwen-2.5-72B（0.240）    & 是 \\
        中文 & CoT       & Llama-4-Maverick（0.736） & Llama-4-Maverick（0.243) & 是 \\
        英文 & zero-shot & Llama-4-Maverick（0.698) & Qwen-2.5-72B（0.254）    & 否 \\
        英文 & CoT       & Llama-4-Maverick（0.699) & Llama-4-Maverick（0.256) & 是 \\
        \bottomrule
    \end{tabular}
\end{table}
```

**关键观察**：最佳模型并非跨语言稳定。Qwen-2.5-72B 在中文 zero-shot 面板同时取得最高 F1 与最低 Brier，但在英文 zero-shot 面板中 F1 榜首让位给 Llama-4-Maverick；后者在剩余三个面板的 F1 指标上均夺冠。这一排名翻转与 H1 所预测的"差距仅为整体平移"不符，却与 H2 所预测的"排名跨语言翻转"一致——Qwen 的训练语料中文暴露量显著高于其英文语料，而 Llama 的训练语料偏向英文主导的网络资源。

```latex
\textbf{关键观察：}最佳模型并非跨语言稳定。Qwen-2.5-72B 在中文 zero-shot 下同时取得最高 F1 与最低 Brier，但在英文 zero-shot 面板中，F1 榜首让位于 Llama-4-Maverick，后者在剩余三个面板的 F1 指标上均夺冠。这一排名翻转与 H1 的"平移"预测不符，却与 H2 一致——Qwen 的中文语料暴露量显著高于其英文语料，而 Llama 的训练语料更多偏向英文主导的网络资源。
```

### 1.4 F1 维度下的分层模式与 Gemma 塌陷

图~\ref{fig:1a-f1-bars} 与图~\ref{fig:1a-f1-dumbbell} 进一步将 F1 维度展开为条形与哑铃两种可视化。

```latex
\subsection{F1 排名的跨语言翻转}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{../artifacts/validated_csv_hate_f1_2x2_bars.pdf}
    \caption{四个面板下九个 LLM 的 F1 值，面板内按 F1 降序排列。}
    \label{fig:1a-f1-bars}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.95\linewidth]{../artifacts/validated_csv_hate_f1_2x2_dumbbell.pdf}
    \caption{每个模型从 zero-shot 到 CoT 的 F1 变化（哑铃图），按语言分面板。}
    \label{fig:1a-f1-dumbbell}
\end{figure}
```

三条经验性规律需要指出：

1. **中文 zero-shot 面板呈现显著的模型分层**：最高 F1（Qwen，0.751）与最低 F1（Gemma-4-31B，0.463）相差近 0.29；英文 zero-shot 面板最高与最低相差仅 ~0.07。这一"中文拉得开、英文拉不开"的模式无法由单纯的任务难度差异解释。
2. **Gemma-4-31B 的跨语言塌陷具有诊断性**：该模型在英文 zero-shot/CoT 下维持在约 0.63，而在中文两个面板中骤降至 ~0.47。H1 无法预测此类模型特异性的语言崩塌幅度。
3. **CoT 并非普适提升**：Qwen 中文 zero-shot F1=0.751，切换 CoT 后降至 0.639；Gemma 中文 CoT 略有提升但仍在 0.47 量级。H1 所暗含的"CoT 作为通用推理放大器带来稳定增益"的预测不成立。

综合以上，研究一(a) 的主要证据支持 H2（语言覆盖不均）而非 H1（任务定义分歧）。**完全驳回 H1 尚需跨标注体系的证据**，但至少，H1 无法解释当前观察到的排名翻转与模型 × 语言交互。

```latex
三条经验性规律需要指出。第一，\textbf{中文 zero-shot 面板呈现显著的模型分层}：最高 F1（Qwen，0.751）与最低 F1（Gemma-4-31B，0.463）之间相差近 0.29 个 F1 单位；而英文 zero-shot 面板下，最高值（Llama，0.698）与最低值（GLM-4.6，0.623）相差仅约 0.07。这一"在中文上拉得开、在英文上拉不开"的模式无法由单纯的任务难度差异解释。
第二，\textbf{Gemma-4-31B 的跨语言塌陷}尤为诊断性：在英文 zero-shot/CoT 下维持在 0.63 附近，而在中文两个面板中骤降至 0.47 附近。
第三，\textbf{CoT 并非普适提升}：Qwen 在中文 zero-shot 下 F1 为 0.751，切换到 CoT 反而降至 0.639。

综合三条规律，我们将主要证据解读为支持 H2（语言覆盖不均）而非 H1（任务定义分歧）。
```

### 1.5 F1 与 Brier 的解耦：分类正确 ≠ 校准正确

F1 度量"能否将句子判为正类"；Brier 度量"给出的概率是否与真实频率一致"。若二者耦合，只报 F1 即可；若解耦，则需分别考察。图~\ref{fig:1a-brier-bars} 与图~\ref{fig:1a-brier-dumbbell} 给出 Brier 结果。

```latex
\subsection{校准（Brier）与准确率（F1）的解耦}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{../artifacts/validated_csv_hate_brier_2x2_bars.pdf}
    \caption{四个面板下九个 LLM 的 Brier 评分（越低越好）。}
    \label{fig:1a-brier-bars}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.95\linewidth]{../artifacts/validated_csv_hate_brier_2x2_dumbbell.pdf}
    \caption{每个模型从 zero-shot 到 CoT 的 Brier 变化（哑铃图）。}
    \label{fig:1a-brier-dumbbell}
\end{figure}
```

**解耦证据**出现在英文 zero-shot 面板：F1 最高的是 Llama-4-Maverick（0.698），但 Brier 最低的是 Qwen-2.5-72B（0.254），且 Llama 的 Brier 反而较高（0.297）。换言之，Llama 更常给出"方向正确但过度自信"的预测，而 Qwen 虽然分类略逊，却在概率尺度上更贴近真实频率。这对下游任务的含义是直接的——若需要可靠的风险阈值（如人机协同审核中"模糊样本留待人工判定"），F1 榜首并非最佳选择。

此外，CoT 对 Brier 的影响呈**非单调**形态：对 Llama 在中文上显著降低 Brier（0.300 → 0.243），却对 Qwen 在英文上反而恶化（0.254 → 0.289）。这一交互进一步削弱"CoT 普适增益"叙事。

```latex
\textbf{解耦的关键证据}出现在英文 zero-shot 面板：F1 最高者为 Llama-4-Maverick（0.698），但 Brier 最低者为 Qwen-2.5-72B（0.254），且 Llama 的 Brier 反而较高（0.297）。这说明若下游任务需要可靠的风险阈值（例如人机协作审核中的"模糊样本留待人工判定"），F1 榜单的第一名并非最佳选择。

此外，CoT 对 Brier 的影响呈现\emph{非单调}形态：对 Llama 在中文上显著降低 Brier（从 zero-shot 的 0.300 降至 CoT 的 0.243），却对 Qwen 在英文上反而恶化（从 0.254 升至 0.289）。
```

### 1.6 自然数据下的性别目标比较：样本构成的混淆

研究一(a) 的最后一项分析直接连接研究一(b)：LLM 对以女性为目标的仇恨帖与以男性为目标的仇恨帖的**平均**攻击性打分是否不同？一个先验合理的预期是：若模型携带"女性方向"的评估偏差，则针对女性的帖子应获得更高的平均攻击分。本小节不对样本语义内容做任何配平，仅以 HateXplain 的 `target` 字段直接分层——这与研究一(b) 的方法论严格配对形成对照。

**操作化**：以 `post_id` 将合并的英文验证集与 HateXplain 原始 `dataset.json` 连接。若至少一位标注者标记 `Women` 且无人标记 `Men`，该帖归为"针对女性"（$n=175$）；若反之，归为"针对男性"（$n=36$）；两者兼有或均无者剔除。中文侧未纳入本图——原始中文语料无男／女目标字段，基于词汇标记（去除可双指的"他"、保留仅女性的"她"）的启发式分类在九个模型上方向分裂、量级接近零，缺乏解释价值。

```latex
\subsection{自然数据下对"针对女性"与"针对男性"的评分：样本构成的混淆}
\label{subsec:1a-gender-natural}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{../artifacts/placeholder_a_1a_gender_target_mean_score.pdf}
    \caption{九个 LLM 在英文 HateXplain 自然数据上对"针对女性"（$n=175$）与"针对男性"（$n=36$）帖子的平均 0--5 分攻击打分，按 Zero-shot 与 CoT 分列。}
    \label{fig:1a-gender-natural}
\end{figure}
```

**观察**：九个模型在两种 prompt 设置下**均**将"针对男性"帖子的平均分打得高于"针对女性"帖子（zero-shot 3.19 vs 2.64；CoT 3.35 vs 2.72），方向与"女性方向偏差"的先验预期**相反**。这一反向结果不应被解读为"模型保护男性"：HateXplain 中 `Men` 目标样本仅 36 条，且高度集中在与族群（尤其是针对黑人男性）交叉的极端仇恨语境；`Women` 样本则覆盖更广的仇恨强度谱。换言之，**自然数据的女／男子集在语义内容上并非可比**——任何子集间差异都混合了样本构成效应与潜在评估偏差，且在本数据上前者远大于后者。

这正是研究一(b) 采用**配对控制**的直接动因：只有在词汇与语义框架被强制匹配、仅改变性别指涉的前提下，"女性 vs 男性"的差异才能被归因于评估偏差本身。

```latex
\textbf{观察要点：}九个模型在两种 prompt 设置下\emph{均}将"针对男性"帖子的平均分打得高于"针对女性"帖子（zero-shot $3.19$ vs $2.64$；CoT $3.35$ vs $2.72$），方向与"女性方向偏差"的预期相反。这一反向结果不应被解读为模型保护男性：HateXplain 中 \texttt{Men} 目标样本量仅 36 条，且高度集中在与族群（尤其是针对黑人男性）交叉的极端仇恨语境中。换言之，\emph{自然数据的女／男子集在语义内容上并非可比}——任何子集间差异都混合了样本构成效应与潜在评估偏差，且前者在此数据上远大于后者。这正是研究一(b) 采用配对设计的直接动因。
```

---

## 2. 研究一(b)：性别群体对换下的攻击性评分

### 2.1 研究目的与竞争假设

研究一(b) 的核心设计是**最小对比对**（minimal pairs）——对同一语义框架下的一个句子仅替换指涉对象的性别代称（"女人"↔"男人"，"她"↔"他"），其余词汇完全一致。该设计用于区分：

- **H1'——语义内容假设**：模型对攻击性的判定应由句子所指涉的语义内容决定；若配对句语义框架相同，则两个版本的评分差应以 0 为中心、对称分布。
- **H2'——评估偏差假设**：模型携带一个独立于语义内容的、指向女性方向的攻击性评估偏差；若该偏差存在，则（女性指涉版 − 男性指涉版）的评分差应系统性大于 0。

需强调：H1' 并不要求模型"对所有内容给零分"，仅要求**配对差值的期望为零**。任何来自词频、讨论度或背景分布的混淆因素在配对内部被抵消。H2' 的进一步可证伪预测是——若偏差普遍存在于当代 LLM，则不同模型在方向性与效应量上应呈**一致而非散漫**的结果。

```latex
\section{研究一(b)：性别群体对换下的攻击性评分}
\label{sec:results-1b}

\subsection{判决性实验：两个竞争假设}

研究一(b)的核心设计是\emph{最小对比对}（minimal pairs）——对同一语义框架下的一个句子仅替换指涉对象的性别代称（例如"女人"$\leftrightarrow$"男人"，"她"$\leftrightarrow$"他"），其余词汇完全一致。
\textbf{H1'——语义内容假设：}若配对句的语义框架相同，则两个版本的评分差应以 0 为中心、对称分布。
\textbf{H2'——评估偏差假设：}模型携带一个独立于语义内容的、指向女性方向的攻击性评估偏差；若该偏差存在，则（女性指涉版 − 男性指涉版）的评分差应系统性大于 0。
```

### 2.2 配对语料构建（对应论文第三章方法的回顾）

371 对中文配对句，覆盖 **10 个一级攻击领域、71 个二级子类、183 个三级标签**。一级领域包括：

| 一级维度 |
|---|
| 性化攻击（性羞辱） |
| 外貌形象攻击 |
| 性别角色／性别表达攻击 |
| 道德品行攻击 |
| 人际关系攻击 |
| 情绪稳定攻击 |
| 能力才干攻击 |
| 智力理性攻击 |
| 社会地位攻击 |
| 经济资源攻击 |

语料构建遵循"**编码框架先行、生成模型辅助起草、研究者人工修订、微博取样但去标识化**"的四步流程：

1. **框架先行**：先基于相关文献整合建立攻击性表达的三级编码框架（10/71/183 维度）。
2. **语句生成**：以各维度的语义特征、攻击对象与表达方式为约束，覆盖中文社交媒体常见辱骂／贬损风格。
3. **生成 × 修订结合**：在框架约束下由生成模型完成初稿，研究者逐条筛查、对超过半数样本进行显著修订，提升语感、场景合理性与真实性。
4. **真实性校准**：修订过程参考微博等中文平台可观察到的辱骂表达，但未直接复制既有帖子，而是改写、去标识化与语义归并。
5. **配对构建**：在同一语义模板下仅替换性别指称词，生成男性版与女性版句子，确保两版本在攻击内容、句法结构、强度线索上尽可能一致。

### 2.3 设计参数速览

| 维度 | 值 |
|---|---|
| 配对数量 | 371 |
| 一级／二级／三级维度 | 10／71／183 |
| 模型 | 与 1a 同一 9 个 LLM |
| 评分尺度 | 3 点（粗粒度），7 点 Likert（**主分析**），0–100 连续滑块（细粒度） |
| 每尺度下每对推理次数 | 2（男版、女版）|
| 差值定义 | $\Delta_{F-M} = \text{score}_F - \text{score}_M$ |
| 推理配置 | OpenRouter API，Temperature=0，线程池批处理 |

### 2.4 统计分析流水线

对每个（模型 × 尺度）组合，在全部 371 对上执行：

1. **配对差**：$\Delta_i = \text{score}_i^F - \text{score}_i^M$。
2. **95% 置信区间**：对 $\{\Delta_i\}$ 执行配对 bootstrap（$N_\text{boot}=10{,}000$）。
3. **非参数检验**：Wilcoxon 符号秩检验（双侧）。
4. **配对效应量**：$d_z = \overline{\Delta} / \text{SD}(\Delta)$。
5. **多重比较校正**：对全部（9 模型 × 3 尺度 × 10 一级领域）$p$ 值执行 Benjamini–Hochberg FDR，得 $q$ 值。
6. **方向性**：对非零差值样本执行 Sign test（方向分解为 $F>M$、$F=M$、$F<M$）。

此外，将 371 对按一级攻击领域分层，在每个（模型 × 领域）内重复 step 2–5。

### 2.5 整体方向性：9/9 模型 F > M（证伪 H1'，支持 H2'）

图~\ref{fig:1b-diff-distribution} 展示代表性模型的配对评分差分布，图~\ref{fig:1b-directionality} 展示方向性分解，表~\ref{tab:1b-overall} 给出统计摘要（7 点 Likert 主分析）。

```latex
\subsection{整体方向性：九个模型均显示 $F > M$}
\label{subsec:1b-overall}

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{../outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b/figures/attack_7pt_likert/panels/fig2_difference_distribution_panel_1.pdf}
        \caption{GPT-5.1（$d_z=0.84$）}
    \end{subfigure}\hfill
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{../outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b/figures/attack_7pt_likert/panels/fig2_difference_distribution_panel_9.pdf}
        \caption{Gemma 4 31B（$d_z=0.77$）}
    \end{subfigure}

    \vspace{0.5em}
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{../outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b/figures/attack_7pt_likert/panels/fig2_difference_distribution_panel_8.pdf}
        \caption{Qwen 2.5 72B（$d_z=0.58$）}
    \end{subfigure}\hfill
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{../outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b/figures/attack_7pt_likert/panels/fig2_difference_distribution_panel_6.pdf}
        \caption{DeepSeek V3.2（$d_z=0.37$）}
    \end{subfigure}
    \caption{代表性 LLM 在 371 对配对语句上的评分差（女性 $-$ 男性）分布。四个模型覆盖整体效应量范围的极端与中位。虚线为 0；系统性右偏即为 H2' 所预测的特征。全部九个模型的分布见附录图~\ref{fig:appendix-diff-distribution-full}。评分尺度为 7 点 Likert。}
    \label{fig:1b-diff-distribution}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{../outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b/figures/attack_7pt_likert/fig3_directionality.pdf}
    \caption{九个模型在每对句子上的方向分解：女性版评分更高、持平、男性版评分更高的比例。}
    \label{fig:1b-directionality}
\end{figure}

\begin{table}[htbp]
    \centering
    \caption{九个 LLM 在 7 点 Likert 尺度下的整体配对差统计。$\Delta_{F-M}$ 为女性减男性的均值差，95\% CI 由配对 bootstrap 得到，$d_z$ 为配对样本 Cohen's $d_z$，$q$ 为 Wilcoxon 符号秩检验经 BH-FDR 校正后的 $p$ 值。数据来源：\texttt{outputs/.../attack\_7pt\_likert/stats\_overall.csv}。}
    \label{tab:1b-overall}
    \small
    \begin{tabular}{lccccc}
        \toprule
        模型 & $n$ & $\Delta_{F-M}$ & 95\% CI & $d_z$ & $q$ (FDR) \\
        \midrule
        Claude Opus 4.5   & 371 & 0.345 & [0.294, 0.396] & 0.693 & $1.5\times 10^{-27}$ \\
        DeepSeek R1       & 371 & 0.243 & [0.186, 0.302] & 0.422 & $8.3\times 10^{-14}$ \\
        DeepSeek V3.2     & 371 & 0.237 & [0.173, 0.305] & 0.371 & $2.2\times 10^{-11}$ \\
        Gemma 4 31B       & 371 & 0.385 & [0.334, 0.437] & 0.773 & $1.9\times 10^{-31}$ \\
        GLM 4.6           & 371 & 0.272 & [0.191, 0.350] & 0.353 & $1.1\times 10^{-10}$ \\
        GPT-5.1           & 371 & 0.458 & [0.404, 0.512] & 0.840 & $2.7\times 10^{-34}$ \\
        Kimi K2           & 371 & 0.404 & [0.321, 0.488] & 0.505 & $1.1\times 10^{-17}$ \\
        Llama 4 Maverick  & 371 & 0.240 & [0.189, 0.294] & 0.477 & $1.6\times 10^{-16}$ \\
        Qwen 2.5 72B      & 371 & 0.256 & [0.210, 0.302] & 0.578 & $1.2\times 10^{-21}$ \\
        \bottomrule
    \end{tabular}
\end{table}
```

**判决性证据**：九个模型的 $\Delta_{F-M}$ 均显著大于 0，效应量 $d_z$ 跨度 0.35（GLM-4.6）–0.84（GPT-5.1），FDR 校正后 $q < 10^{-10}$。方向分解显示，在 371 对中女性版更高的比例（25.9%–45.0%）系统性高于男性版更高的比例（0.0%–9.97%）。"9/9 方向一致 + 全部通过严格多重比较校正"的模式**直接证伪 H1'**——若语义已同，随机噪声不应产生如此一致的系统偏移——并**支持 H2'**。

需强调：**"支持 H2'"不等同于"所有模型携带相同强度的偏差"**。效应量跨越 2.4 倍（0.35–0.84），说明偏差在各模型间方向一致但强度显著不同——不同预训练与对齐流程塑造的"女性目标更易被标为攻击性"这一启发式，强度差异巨大。Gemma-4-31B 与 GPT-5.1 的效应量最大，DeepSeek V3.2 与 GLM-4.6 最小。

```latex
\textbf{判决性证据：}九个模型的 $\Delta_{F-M}$ 均显著大于 0，效应量 $d_z$ 跨度为 0.35（GLM-4.6）至 0.84（GPT-5.1），FDR 校正后 $q$ 值均小于 $10^{-10}$。这一"9/9 方向一致 + 全部通过严格多重比较校正"的模式\textbf{可直接证伪 H1'}——若句子语义已同，随机噪声不应产生如此一致的系统偏移——并\textbf{支持 H2'}。

需要注意，这里 \emph{"支持 H2'"不等同于"全部模型携带相同偏差"}。效应量跨越 2.4 倍区间（0.35--0.84），说明偏差在各模型间虽方向一致，但强度显著不同——不同预训练与对齐流程塑造的"女性目标更易被标为攻击性"这一启发式，强度差异巨大。
```

### 2.6 维度定位：偏差集中在身体／性相关领域

若 H2' 反映的是一个**笼统**的偏差（"女性目标 → 更高分"），则不对称应在所有攻击领域同步出现；若反之，偏差可能集中于与性别刻板印象绑定的领域（身体、性、外貌）。图~\ref{fig:1b-level1-forest} 按 10 个一级领域分层。

```latex
\subsection{维度定位：哪些攻击领域驱动不对称}
\label{subsec:1b-level1}

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[t]{0.32\linewidth}
        \centering
        \includegraphics[width=\linewidth]{../outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b/figures/attack_7pt_likert/panels/fig4_level1_forest_panel_1.pdf}
        \caption{GPT-5.1}
    \end{subfigure}\hfill
    \begin{subfigure}[t]{0.32\linewidth}
        \centering
        \includegraphics[width=\linewidth]{../outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b/figures/attack_7pt_likert/panels/fig4_level1_forest_panel_9.pdf}
        \caption{Gemma 4 31B}
    \end{subfigure}\hfill
    \begin{subfigure}[t]{0.32\linewidth}
        \centering
        \includegraphics[width=\linewidth]{../outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b/figures/attack_7pt_likert/panels/fig4_level1_forest_panel_6.pdf}
        \caption{DeepSeek V3.2}
    \end{subfigure}
    \caption{三个诊断性模型在十个一级攻击领域上的配对差森林图。水平轴为 $\Delta_{F-M}$（7 点 Likert 尺度），误差条为 95\% Bootstrap CI。GPT-5.1 展示偏差覆盖面最广的形态；Gemma 在性别角色／性别表达子集（$n=14$）唯一显著；DeepSeek V3.2 代表偏差最集中的形态（仅 4 个领域显著）。全部九个模型的森林图见附录图~\ref{fig:appendix-level1-forest-full}。}
    \label{fig:1b-level1-forest}
\end{figure}
```

**按领域看的分布高度集中**：
- **性化攻击（性羞辱）**：全部 9 个模型显著（FDR $q<0.05$），$n=33$ 下 $d_z$ 跨度 0.57–1.06。
- **外貌形象攻击**：全部 9 个模型显著，$n=37$ 下 $d_z$ 跨度 0.51–1.19。
- 其余领域（道德品行、人际关系、经济资源、社会地位、情绪稳定、能力才干、智力理性）在多数模型上显著，效应量中等（$d_z \approx 0.2\text{--}0.7$）。
- **性别角色／性别表达攻击**维度理论上最应出现偏差，实际仅在极少模型（Gemma-4-31B，$d_z=0.83$）上显著——主因是 $n=14$ 的统计功效不足，非偏差真正缺席。

**对 H2' 的进一步刻画**：偏差并非"所有维度一致上浮"的均匀加码，而是集中在**身体与性相关**的攻击领域——这些恰是社会性别刻板印象中"女性更易受此类攻击"或"女性对此类攻击更敏感"观念所直接对应的领域。GPT-5.1 是诊断性个案：7 个领域 $d_z \geq 0.77$（性化 1.06、智力理性 1.10、能力才干 1.10、情绪稳定 1.10、外貌形象 0.87、社会地位 0.88、道德品行 0.77），覆盖面远超其他模型；DeepSeek V3.2 则代表另一极——仅 4 个领域 FDR 显著，且 2 个（性化、外貌）效应量较大（$d_z>0.69$），其余 6 个领域不显著。

```latex
\textbf{按领域看的分布高度集中。}跨九个模型合并观察：\textbf{"性化攻击（性羞辱）"}与\textbf{"外貌形象攻击"}在全部 9 个模型上均显著（FDR $q<0.05$），分别在 $n=33$ 与 $n=37$ 配对下 $d_z$ 跨度为 0.57--1.06 与 0.51--1.19。而\textbf{"性别角色／性别表达攻击"}维度虽在理论上最应出现性别偏差，实际却仅在极少模型（如 Gemma-4-31B，$d_z=0.83$）上显著（$n=14$）——这一反直觉结果更可能归因于样本量过小导致的统计功效不足。

\textbf{对 H2' 的进一步刻画：}上述模式意味着偏差并非"所有维度一致上浮"的均匀加码，而是集中在\emph{身体与性相关}的攻击领域——这些恰是社会性别刻板印象中"女性更易受此类攻击"或"女性对此类攻击更敏感"的观念所直接对应的领域。
```

### 2.7 尺度稳健性：跨三种评分尺度 $d_z$ 单调一致

一个合理的怀疑是：整体偏差是否为 7 点 Likert 尺度特有的离散化 artifact——例如在 0–100 连续尺度上女性与男性版本分布实际上可能接近，只是被 7 点尺度"粗粒度化"后放大为整数差？为回应此可能性，对 3 点尺度与 0–100 滑块尺度执行平行分析，并将三尺度下 $d_z$ 跨模型对齐。

```latex
\subsection{尺度稳健性}
\label{subsec:1b-scale-invariance}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{../outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b/figures/placeholder_b_three_scale_dz_scatter.pdf}
    \caption{三评分尺度下 Cohen's $d_z$ 的跨模型两两散点。虚线为 $y=x$；每个点代表一个模型。}
    \label{fig:1b-scale-invariance}
\end{figure}
```

**观察**：九个模型的三个 $d_z$ 值在两两散点中普遍贴近 $y=x$ 对角线——说明"女性版本评分更高"的主效应**不是**仅在 7 点 Likert 下的离散化 artifact。单调性在三尺度间保持：任一尺度上 $d_z$ 最大的模型（GPT-5.1）在其他两个尺度上也位于右上角；$d_z$ 最小的模型（GLM、DeepSeek V3.2）在其他尺度上同样位于左下角。0–100 滑块尺度下部分模型的 $d_z$ 略高于 Likert 尺度（点略偏向对角线上方），与滑块更细粒度、允许小幅度偏差被更充分记录的直觉相符。

### 2.8 模型 × 维度一致性热图

节 2.6 按领域汇总了九个模型的显著性，但尚未回答一个更精细的问题：**九个模型是否在"哪些领域偏差最强"上达成一致**？若模型间高度相关，则提示当前 LLM 的性别评估偏差有共同来源（中文互联网语料中长期存在的性别刻板印象分布）；反之则每个模型的偏差应视为独立对齐副产品。

```latex
\subsection{模型 \texttimes{} 维度的一致性模式}
\label{subsec:1b-model-by-dim}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{../outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b/figures/placeholder_c_model_by_level1_dz_heatmap.pdf}
    \caption{九个模型 \texttimes{} 十个一级攻击领域的 Cohen's $d_z$ 热图（7 点 Likert）。颜色越红表示"女性版 $>$ 男性版"效应越强。单元格中的星号表示 BH-FDR 校正后 Wilcoxon 显著性（$*$ $q<0.05$，$**$ $q<0.01$，$***$ $q<0.001$）。}
    \label{fig:1b-model-by-dim-heatmap}
\end{figure}
```

**观察**：热图揭示两条一致性结构：
1. **列方向高度集中**——"性化攻击（性羞辱）"与"外貌形象攻击"两列在九个模型上均呈深红（$d_z$ 普遍 $>0.5$，多数 $>0.8$）；"性别角色／性别表达攻击"列因 $n=14$ 统计功效限制仅 Gemma 一行显著。
2. **行方向覆盖面差异**——GPT-5.1 与 Gemma-4-31B 近乎全行深色（7 个领域 $d_z \geq 0.77$）；DeepSeek V3.2 与 GLM-4.6 仅在少数几列显著。

换言之，**偏差方向在九模型间高度一致，但偏差的广度在模型间差异明显**。这一模式与"所有模型从训练语料中吸收了相近的性别刻板印象分布、但对齐流程对偏差的压制程度不同"这一更具体假设相容——直接通往研究二（LoRA 微调对齐）的设计动因。

### 2.9 配对散点与个案：表面中立，分歧时几乎单向

图~\ref{fig:1b-paired-scatter} 给出每个模型在 371 对上的配对散点（横轴男性版，纵轴女性版）。对角线上方点数即女性版更高的对数。

```latex
\subsection{配对散点与个案观察}

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{../outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b/figures/attack_7pt_likert/panels/fig1_paired_scores_panel_2.pdf}
        \caption{Claude Opus 4.5（持平 65.0\%，不持平时 64:1）}
    \end{subfigure}\hfill
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{../outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b/figures/attack_7pt_likert/panels/fig1_paired_scores_panel_4.pdf}
        \caption{Llama Maverick（持平 71.4\%，不持平时 9.6:1）}
    \end{subfigure}

    \vspace{0.5em}
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{../outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b/figures/attack_7pt_likert/panels/fig1_paired_scores_panel_8.pdf}
        \caption{Qwen 2.5 72B（持平 73.9\%，不持平时 96:1）}
    \end{subfigure}\hfill
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{../outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b/figures/attack_7pt_likert/panels/fig1_paired_scores_panel_1.pdf}
        \caption{GPT-5.1（持平 54.2\%，不持平时 55.7:1）}
    \end{subfigure}
    \caption{四个代表性模型的配对散点图。横轴为男性指涉版本评分，纵轴为女性指涉版本评分。45\textdegree{} 对角线上方的密度即为"女性版评分更高"的配对比例；对角线附近的条带密度即为评分一致的比例。全部九个模型的配对散点见附录图~\ref{fig:appendix-paired-scatter-full}。}
    \label{fig:1b-paired-scatter}
\end{figure}
```

散点密度在所有模型中均偏向对角线上方。值得注意的是 Claude Opus 4.5、Llama-4-Maverick 与 Qwen-2.5-72B 呈现较高的"持平"比例（65.0%、71.4%、73.9%），即大部分配对被赋予**完全相同**的 Likert 分，但在剩余不持平的配对中，不对称比例达到 2:1 以上（Claude 64:1、Qwen 96:1）。这意味着即便模型在大多数情况下"看起来中立"，一旦出现分歧，方向几乎总偏向女性版评分更高。GPT-5.1 作为对比：持平比例最低（54.2%），但不对称仍最强（55.7:1）。

---

## 3. 研究一(a) 与 (b) 的整合结论与研究范围

### 3.1 两项子研究的方法论互补

| 维度 | 研究一(a) | 研究一(b) |
|---|---|---|
| 评估目标 | 分类准确度与概率校准 | 同一语义下的性别评估不对称 |
| 数据 | 自然语料（中／英，2400/language）| 理论框架约束下的配对语料（ZH，371 对）|
| 关键指标 | F1, Brier | $\Delta_{F-M}$, $d_z$, FDR 校正 $q$ |
| 判决假设 | H1 vs H2（排名稳定 vs 跨语言翻转）| H1' vs H2'（差值对称 vs 系统右偏）|
| 主要发现 | 最佳模型跨语言翻转，F1/Brier 解耦，自然数据"女性 < 男性" | 9/9 模型女性 > 男性，$d_z$ 0.35--0.84，FDR $q<10^{-10}$ |

两者并非矛盾：自然数据下"男 > 女"的反向结果源自 HateXplain 中 `Men` 样本与极端族群仇恨语境的高度交叉（样本构成效应），而配对控制下 9/9 一致"女 > 男"揭示的是**语义固定后的残余评估偏差**。同一底层偏差在不同方法论下表现不同——这也是为何仅靠自然数据无法定位该偏差的原因。

### 3.2 对四组假设的综合判决

- **H1**（任务定义分歧）→ 不支持：跨语言排名翻转与 Gemma 模型特异塌陷无法由"规范差异"解释。
- **H2**（语言覆盖不均）→ 支持：中文 Qwen 最佳、英文 Llama 最佳的翻转模式与语料暴露方向一致。
- **H1'**（语义内容决定）→ 被证伪：9 模型 × 371 对 FDR 全通过，差值分布系统性右偏。
- **H2'**（评估偏差存在）→ 支持：方向 9/9 一致，偏差集中于性化与外貌领域（与性别刻板印象高度对应）。

### 3.3 研究范围与局限

- **(b) 当前仅限中文语境**：371 对配对句与三级分类体系为中文自建；英文配对语料尚未完成，本研究不对 H2' 在英文条件下的表现作声明，跨语言复现留待后续工作。
- **(a) 自然数据侧**：中文语料无 male/female 目标字段，基于词汇标记的启发式分类在 9 模型上方向分裂、量级近零，本章未将 ZH 自然数据纳入正文图。
- **性别角色／性别表达**维度在 (b) 中配对数量过少（$n=14$），统计功效不足以检验该理论上最敏感的领域；未来配对扩充应优先覆盖此维度。
- **尺度粒度**：7 点 Likert 下较高的"持平"比例（54%–74%）说明偏差虽系统存在但并非每对都被触发；混合效应或有序回归模型可更精细地刻画"何时触发"的预测因素。

```latex
\section{研究范围与分析边界}
\label{sec:scope-notes}

上文报告的研究一(b)配对分析当前仅在\emph{中文}语境下展开：371 对配对句与对应的三级分类体系均为中文自建语料。跨语言泛化所需的英文配对语料目前尚未完成，本文不对 H2' 在英文条件下的表现做出声明；该方向的延伸研究将在讨论部分作为未来工作列出。
```

### 3.4 通往研究二的链接

研究一诊断出的"偏差方向共同、偏差广度异质"模式——即所有模型吸收了相近的性别刻板印象但对齐流程压制程度不同——直接为研究二的动机提供依据：若真人被试（研究 2a）在 371 对上的评分分布不呈现该系统性不对称，则模型的对齐失配是可通过参数高效微调（LoRA，研究 2b）纠正的目标；若真人被试同样呈现一定不对称，则需要更精细的公平性目标设计。研究二的具体方法与结果在论文第五章给出。

---

## 附录：图表清单与文件索引

### 研究一(a) 图表

| 正文引用 | 类型 | 文件路径 |
|---|---|---|
| 图 4.1 | 2×2 散点（F1 vs Brier） | `artifacts/validated_csv_hate_f1_brier_2x2_scatter.pdf` |
| 表 4.1 | 最佳模型汇总 | `artifacts/validated_csv_hate_f1_brier_best_models.csv` |
| 图 4.2 | F1 条形图 | `artifacts/validated_csv_hate_f1_2x2_bars.pdf` |
| 图 4.3 | F1 哑铃图 | `artifacts/validated_csv_hate_f1_2x2_dumbbell.pdf` |
| 图 4.4 | Brier 条形图 | `artifacts/validated_csv_hate_brier_2x2_bars.pdf` |
| 图 4.5 | Brier 哑铃图 | `artifacts/validated_csv_hate_brier_2x2_dumbbell.pdf` |
| 图 4.6 | 自然数据性别目标均值图（EN） | `artifacts/placeholder_a_1a_gender_target_mean_score.pdf` |

### 研究一(b) 图表（7 点 Likert 主分析）

| 正文引用 | 类型 | 文件路径 |
|---|---|---|
| 图 4.7 | 差值分布（4 代表模型） | `outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b/figures/attack_7pt_likert/panels/fig2_difference_distribution_panel_*.pdf` |
| 图 4.8 | 方向性堆叠条形图 | `.../attack_7pt_likert/fig3_directionality.pdf` |
| 表 4.2 | 整体配对差统计（9 模型） | `.../attack_7pt_likert/stats_overall.csv` |
| 图 4.9 | 一级维度森林图（3 诊断模型） | `.../attack_7pt_likert/panels/fig4_level1_forest_panel_*.pdf` |
| 图 4.10 | 三尺度 $d_z$ 散点 | `outputs/.../figures/placeholder_b_three_scale_dz_scatter.pdf` |
| 图 4.11 | 模型×维度 $d_z$ 热图 | `outputs/.../figures/placeholder_c_model_by_level1_dz_heatmap.pdf` |
| 图 4.12 | 配对散点图（4 代表模型） | `.../attack_7pt_likert/panels/fig1_paired_scores_panel_*.pdf` |
| 附录图 | 全部 9 面板完整版（差值分布／森林图／配对散点） | `.../attack_7pt_likert/fig2_*.pdf`、`fig4_*.pdf`、`fig1_*.pdf` |

### 可编译的独立 LaTeX 源文件

完整的、可独立 `xelatex` 编译的源文件为 `slides/results_1a_1b.tex`（17 页，含附录三张 9 面板完整图）。整合进主论文时需删除 `% === STANDALONE-PREAMBLE-BEGIN ===` 至 `% === STANDALONE-DOC-END ===` 之间的 wrapper 块。
