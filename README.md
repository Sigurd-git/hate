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

### 准备数据

使用包含至少两列的 CSV 文件：

| text | language |
| --- | --- |
| `"I hate all dogs"` | `en` |
| `"我喜欢每个人"` | `zh` |

示例文件位于 `sample_data.csv`。

### 批量运行

```bash
export OPENROUTER_API_KEY="sk-or-..."
python pipeline.py sample_data.csv openrouter/auto --limit 2
```

脚本会打印包含结构化响应的 JSON。所有请求均使用 OpenRouter 的结构化输出通道（`response_format=json_schema`），从而优先选择支持确定性 JSON 的模型。

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
