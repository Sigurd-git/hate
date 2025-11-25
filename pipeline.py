"""用于数据提取与 OpenRouter 提交的精简流水线。

仅支持 ``zero_shot`` 与 ``chain_of_thought`` 两种提示范式。
"""
# 本模块提供：
# 1. 从 CSV 文件中加载样本（含 text 和 language）
# 2. 根据模型类型构造不同的结构化请求参数
# 3. 通过 OpenRouter 提交请求并返回结构化 JSON 结果
# 4. 支持批处理、长度中断、score+reason 的 JSON 输出解析

from __future__ import annotations  # 允许延迟类型注解（避免循环导入问题）

import csv
import json
import logging
import os
from dataclasses import dataclass  # 用于定义轻量级数据结构
from typing import Callable, Dict, List, Optional, Sequence, Tuple  # 类型提示

from dotenv import load_dotenv  # 加载 .env 环境变量
from openai import OpenAI  # OpenAI/ OpenRouter 兼容客户端

from prompts import LANGUAGES, PromptParadigm, render_prompt  # 导入语言列表、提示范式、prompt 生成函数

# OpenRouter 的 API 入口地址
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# 每次 completion 最大 token 数（用于限制输出）
MAX_TOKENS_PER_COMPLETION = 512

# 加载 .env 文件中保存的 API key 变量
load_dotenv()

# 定义模块级别 logger
logger = logging.getLogger(__name__)

# 缓存 client，避免重复初始化
_CLIENT: OpenAI | None = None

# 定义模型返回 JSON 的 schema，用于要求结构化输出
STRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 5},  # 0–5 分
        "reason": {"type": "string"},  # 打分原因文字
    },
    "required": ["score", "reason"],  # 必须包含
    "additionalProperties": False,  # 禁止出现额外字段
}


@dataclass
class Sample:
    """表示一条文本样本（输入 + 语言）。"""
    text: str
    language: str


@dataclass
class ScoreResponse:
    """用于返回模型结构化评分结果."""
    payload: Dict[str, object]      # JSON 内容（score + reason）
    finish_reason: Optional[str]    # finish_reason，例如 "stop"、"length"


def load_samples(
    path: str, limit: Optional[int] = None, language: Optional[str] = None
) -> List[Sample]:
    """从带有 `text` 和 `language` 列的 CSV 中加载样本，可按语言过滤。"""
    samples: List[Sample] = []
    language_filter = language.lower() if language else None

    # 打开 CSV 文件
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # 达到 limit 则停止
            if limit is not None and len(samples) >= limit:
                break

            text = (row.get("text") or "").strip()
            lang_value = (row.get("language") or "").strip().lower()

            # 如果指定语言，就必须匹配
            if language_filter and lang_value != language_filter:
                continue

            # 跳过没有文本或语言不在支持列表中的
            if not text or lang_value not in LANGUAGES:
                continue

            samples.append(Sample(text=text, language=lang_value))

    return samples


def _build_response_format(
    schema: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """构造 response_format 参数，要求模型输出 JSON。"""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "hate_score",  # 可随意命名
            "schema": schema or STRUCTURE_SCHEMA,  # 使用默认 schema 或外部传入
        },
    }


def _get_openrouter_client() -> OpenAI:
    """初始化并返回全局 OpenRouter 客户端（带缓存）。"""
    global _CLIENT
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 OPENROUTER_API_KEY 环境变量，请检查 .env")

    if _CLIENT is None:
        # 使用 OpenRouter 作为 base_url
        _CLIENT = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return _CLIENT


def _build_extra_headers(metadata: Optional[Dict[str, str]]) -> Dict[str, str]:
    """构造 OpenRouter 建议使用的头部字段，如 Referer 和 Title。"""
    referer = metadata.get("referer", "") if metadata else ""
    title = metadata.get("title", "hate-eval") if metadata else "hate-eval"
    return {"HTTP-Referer": referer, "X-Title": title}


def _build_structure_schema(include_score_bounds: bool = True) -> Dict[str, object]:
    """复制默认 JSON schema，可选移除 score 的上下界。"""
    schema = json.loads(json.dumps(STRUCTURE_SCHEMA))  # 深拷贝
    if not include_score_bounds:
        score_schema = schema["properties"]["score"]
        score_schema.pop("minimum", None)
        score_schema.pop("maximum", None)
        score_schema["description"] = "Integer between 0 and 5 inclusive."
    return schema


def _compose_request_kwargs(
    sample: Sample,
    model: str,
    prompt_paradigm: PromptParadigm,
    metadata: Optional[Dict[str, str]],
    timeout: int,
    schema: Dict[str, object],
    extra_options: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """构造一次 OpenRouter 请求的完整参数 dict。"""
    # 生成 prompt
    prompt_text = render_prompt(sample.language, prompt_paradigm, sample.text)
    messages = [{"role": "user", "content": prompt_text}]

    # 基础参数
    request_kwargs: Dict[str, object] = {
        "model": model,                          # 使用的模型
        "temperature": 0,                        # 固定温度
        "messages": messages,                    # chat messages
        "response_format": _build_response_format(schema),  # 要求结构化 JSON 输出
        "timeout": timeout,                      # 超时
        "extra_headers": _build_extra_headers(metadata),    # 头部字段
        "max_tokens": MAX_TOKENS_PER_COMPLETION, # 限制 token 输出
    }

    # 构造 extra_body
    extra_body: Dict[str, object] = {}
    if metadata:
        extra_body["metadata"] = metadata
    if extra_options:
        extra_body.update(extra_options)
    if extra_body:
        request_kwargs["extra_body"] = extra_body

    return request_kwargs


def _build_default_request_kwargs(...):
    """默认模型的请求结构（score bound 正常启用）。"""
    return _compose_request_kwargs(
        sample,
        model,
        prompt_paradigm,
        metadata,
        timeout,
        schema=_build_structure_schema(),
    )


def _build_anthropic_request_kwargs(...):
    """Anthropic 模型特殊要求：score 不加上下界。"""
    request_kwargs = _compose_request_kwargs(
        sample,
        model,
        prompt_paradigm,
        metadata,
        timeout,
        schema=_build_structure_schema(include_score_bounds=False),
    )
    return request_kwargs


def _build_glm_request_kwargs(...):
    """GLM 模型使用默认结构化 schema。"""
    request_kwargs = _compose_request_kwargs(
        sample,
        model,
        prompt_paradigm,
        metadata,
        timeout,
        schema=_build_structure_schema(),
    )
    return request_kwargs


def _build_meta_llama_request_kwargs(...):
    """LLaMA 模型使用默认结构化 schema。"""
    request_kwargs = _compose_request_kwargs(
        sample,
        model,
        prompt_paradigm,
        metadata,
        timeout,
        schema=_build_structure_schema(),
    )
    return request_kwargs


def _build_deepseek_r1_request_kwargs(...):
    """DeepSeek R1 启用 reasoning 输出。"""
    return _compose_request_kwargs(
        sample,
        model,
        prompt_paradigm,
        metadata,
        timeout,
        schema=_build_structure_schema(),
        extra_options={"reasoning": {"effort": "medium"}, "include_reasoning": True},
    )


def _build_deepseek_v3_request_kwargs(...):
    """DeepSeek V3 启用 reasoning 开关。"""
    request_kwargs = _compose_request_kwargs(
        sample,
        model,
        prompt_paradigm,
        metadata,
        timeout,
        schema=_build_structure_schema(),
        extra_options={"reasoning": {"enabled": True}},
    )
    return request_kwargs


# 定义模型到构造函数的映射
RequestBuilder = Callable[
    [Sample, str, PromptParadigm, Optional[Dict[str, str]], int], Dict[str, object]
]

MODEL_REQUEST_BUILDERS: Dict[str, RequestBuilder] = {
    "openai/gpt-5.1": _build_default_request_kwargs,
    "anthropic/claude-sonnet-4.5": _build_anthropic_request_kwargs,
    "z-ai/glm-4.6": _build_glm_request_kwargs,
    "meta-llama/llama-4-maverick:free": _build_meta_llama_request_kwargs,
    "deepseek/deepseek-r1-0528:free": _build_deepseek_r1_request_kwargs,
    "deepseek/deepseek-v3.2-exp": _build_deepseek_v3_request_kwargs,
}


def _build_model_request_kwargs(...):
    """根据不同模型选择不同构造函数。"""
    builder = MODEL_REQUEST_BUILDERS.get(model, _build_default_request_kwargs)
    return builder(sample, model, prompt_paradigm, metadata, timeout)


def request_score(...):
    """对单条样本发送请求，返回 ScoreResponse（结构化结果 + finish_reason）。"""
    client = _get_openrouter_client()

    logger.info(
        "Requesting OpenRouter score language=%s model=%s paradigm=%s",
        sample.language,
        model,
        prompt_paradigm,
    )

    # 构造请求参数
    request_kwargs = _build_model_request_kwargs(
        sample,
        model,
        prompt_paradigm,
        metadata,
        timeout,
    )
    logger.debug("Request payload keys: %s", sorted(request_kwargs))

    # 调用 OpenRouter
    response = client.chat.completions.create(**request_kwargs)
    data = response.model_dump()

    # 解析 finish_reason
    finish_reason = data["choices"][0].get("finish_reason")

    # 解析结构化 JSON
    payload = _parse_structured_response(data)
    return ScoreResponse(payload=payload, finish_reason=finish_reason)


def _parse_structured_response(data: Dict[str, object]) -> Dict[str, object]:
    """从 OpenRouter 返回的数据结构中提取 JSON 内容。"""
    content = data["choices"][0]["message"]["content"]
    message = data["choices"][0]["message"]
    reason = message.get("reasoning", None)

    # 有些模型返回 list，只取第一个
    if isinstance(content, list):
        content = content[0]

    # 有些模型返回字符串，需要 JSON 解析
    if isinstance(content, str):
        return json.loads(content)

    return content


def run_batch(...):
    """对 CSV 中的样本直接跑一批评分（用于快速调试）。

    Parameters
    ----------
    prompt_paradigm:
        只能传入 ``zero_shot`` 或 ``chain_of_thought``。
    """
    samples = load_samples(csv_path, limit=limit, language=language)
    return score_samples(samples, model=model, prompt_paradigm=prompt_paradigm)


def _score_samples_internal(...):
    """核心评分循环：逐条请求结构化输出，并处理 length 中断。"""
    batch_rows: List[Dict[str, object]] = []
    aborted = False

    for sample in samples:
        response = request_score(sample, model=model, prompt_paradigm=prompt_paradigm)

        # 如果模型因输出过长而提前终止，则整个 batch 报废
        if response.finish_reason == "length":
            aborted = True
            batch_rows.clear()
            break

        # 正常添加结果
        batch_rows.append(
            {"text": sample.text, "language": sample.language, **response.payload}
        )

    return batch_rows, aborted


def score_samples(...):
    """对样本集合评分（不返回 aborted 状态）。"""
    results, _ = _score_samples_internal(samples, model, prompt_paradigm)
    return results


def score_samples_with_status(...):
    """评分并返回是否被 aborted。

    Supported prompt paradigms: ``zero_shot`` and ``chain_of_thought``.
    """
    return _score_samples_internal(samples, model, prompt_paradigm)