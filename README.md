# 仇恨言论提示运行器

该仓库目前聚焦于计划中研究流程的两个步骤：

1. **数据提取** —— 从 CSV 文件中加载双语仇恨言论刺激样本。
2. **结构化 OpenRouter 提交** —— 将每条样本发送至指定模型，并强制返回 JSON Schema（分数 + 可选理由）。

代码保持扁平结构（单个 `pipeline.py`），便于审计与扩展。

## 快速开始

安装 [uv](https://docs.astral.sh/uv/)（高速 Python 依赖管理器），并用它创建本地虚拟环境及开发依赖：

```bash
uv sync
source .venv/bin/activate
```

在仓库根目录创建 `.env`（或使用现有文件）写入 OpenRouter 凭据：

```bash
echo 'OPENROUTER_API_KEY=sk-or-...' >> .env
```

### 准备数据

使用包含至少两列的 CSV 文件：

| text | language |
| --- | --- |
| `"I hate all dogs"` | `en` |
| `"我喜欢每个人"` | `zh` |

示例 CSV 已放在 `data/sample.csv`，可直接用它做冒烟测试或复制一份改写文本。

### 批量运行

```bash
uv run python run_experiments.py
```

`run_experiments.py` 会按照 `DATASET_PATTERNS` 字典里维护的 `glob` 模式去发现 CSV（默认只启用 `test: data/sample.csv`，需要评估更多语料时直接把路径写进字典即可）。脚本会针对每个匹配到的文件、`prompts.py` 暴露的语言/提示范式，以及 `MODELS` 中列出的模型组合进行迭代，并以 `BATCH_SIZE=10` 切片调用 `pipeline.score_samples_with_status`；`discover_existing_batches` 会用同名 `glob` 路径检查是否已经产出过同一批次，从而支持断点续跑。

每个批次会以 Excel 形式落盘：`outputs/<dataset_type>/<dataset_label>/<language>/<paradigm>/<model>_batch_<index>.xlsx`（模型名中的 `/` 会替换成 `_`）。`save_batch_results` 会为每行补充模型、提示范式、数据集标签与批次编号，便于后续用 polars/numpy 继续聚合。`limit` 参数会直接传给 `pipeline.load_samples`，因此可以按语言裁剪样本数量，在真实跑批前快速 smoke test。

如需在交互式环境里手动取回结果，可在 Python REPL 中导入 `pipeline.run_batch(csv_path, model, language="en", prompt_paradigm="few_shot")`，它会返回包含 `text`、`language`、`score`、`reason` 字段的列表，可自行写出更轻量的分析脚本。

### 单次 CSV 评估

当你只需验证单个 CSV 或模型配置是否工作正常时，最轻量的做法是直接在虚拟环境里调用 `pipeline.run_batch`：

```bash
uv run python - <<'PY'
import pprint
import pipeline

scored_rows = pipeline.run_batch(
    csv_path="data/sample.csv",
    model="openai/gpt-5.1",
    limit=2,
    prompt_paradigm="zero_shot",
)
pprint.pp(scored_rows)
PY
```

上述命令借助 uv 提供的隔离解释器加载少量样本并打印结构化结果；需要测试其他语言或提示范式时，调整 `language` 与 `prompt_paradigm` 形参即可，若想更换模型则修改 `model` 参数。

### 运行测试

```bash
uv run pytest
```

测试套件会检查 CSV 读取、提示构造，并确保请求使用约定的结构化模式。

## 已完成内容

- ✅ 具备语言感知模板的 CSV 读取流程。
- ✅ 针对中英文句子的确定性提示词。
- ✅ 请求结构化输出（0–5 分 + 理由）的 OpenRouter 客户端。
- ✅ 可临时调用的精简 CLI，并对关键辅助函数提供 pytest 覆盖。

## TODO / 后续规划

- ⏳ 扩展 schema，使其携带模型置信度、二元标记及其他元数据。
- ⏳ 实现批处理与退避工具，以支持高吞吐评估。
- ⏳ 增加下游公平性指标（ΔHATE-rate、Δs、flip rate 等）。
- ⏳ 将结构化响应持久化（如 parquet），便于后续统计分析。
- ⏳ 添加少样本与 CoT 场景的提示变体。
