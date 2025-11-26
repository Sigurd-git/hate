"""用于数据提取与 OpenRouter 提交的精简流水线。"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from prompts import LANGUAGES, PromptParadigm, render_prompt

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS_PER_COMPLETION = 512

load_dotenv()

logger = logging.getLogger(__name__)

_CLIENT: OpenAI | None = None

STRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 5},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}


@dataclass
class Sample:
    text: str
    language: str


@dataclass
class ScoreResponse:
    payload: Dict[str, object]
    finish_reason: Optional[str]


@dataclass(frozen=True)
class DatasetLanguageRule:
    default_language: str
    allowed_languages: Tuple[str, ...]


DATASET_LANGUAGE_CONFIG: Dict[str, DatasetLanguageRule] = {
    "5_new_chinesehatedata_2400_balanced": DatasetLanguageRule(
        default_language="zh",
        allowed_languages=("zh",),
    ),
    "5_new_englishhatedata_2400_balanced": DatasetLanguageRule(
        default_language="en",
        allowed_languages=("en",),
    ),
}


# def load_samples(
#     path: str, limit: Optional[int] = None, language: Optional[str] = None
# ) -> List[Sample]:
#     """从含有 `text` 与 `language` 列的 CSV 中载入样本，可选按语言筛选。"""
#     samples: List[Sample] = []
#     language_filter = language.lower() if language else None
#     with open(path, newline="", encoding="utf-8") as handle:
#         reader = csv.DictReader(handle)
#         for row in reader:
#             if limit is not None and len(samples) >= limit:
#                 break
#             text = (row.get("text") or "").strip()
#             lang_value = (row.get("language") or "").strip().lower()
#             if language_filter and lang_value != language_filter:
#                 continue
#             if not text or lang_value not in LANGUAGES:
#                 continue
#             samples.append(Sample(text=text, language=lang_value))
#     return samples

def load_samples(
    path: str,
    limit: Optional[int] = None,
    language: Optional[str] = None,
) -> List[Sample]:
    """
    从新的数据格式文件中加载样本，并根据配置约束语言。
    文件必须包含列：text。自动识别 CSV 或 Excel。
    """
    dataset_identifier = Path(path).stem
    rule = DATASET_LANGUAGE_CONFIG.get(dataset_identifier)
    allowed_languages = (
        tuple(lang.lower() for lang in rule.allowed_languages)
        if rule
        else tuple(lang.lower() for lang in LANGUAGES)
    )
    if not allowed_languages:
        raise ValueError(f"No allowed languages configured for dataset={dataset_identifier}")

    default_language = (
        rule.default_language.lower()
        if rule
        else allowed_languages[0]
    )
    if default_language not in allowed_languages:
        raise ValueError(
            f"Default language must be included in allowed languages for dataset={dataset_identifier}"
        )
    requested_language = language.lower() if language else None
    if requested_language and requested_language not in allowed_languages:
        logger.info(
            "Skip dataset=%s for requested_language=%s allowed=%s",
            dataset_identifier,
            requested_language,
            allowed_languages,
        )
        return []

    resolved_language = requested_language or default_language

    if path.endswith(".csv"):
        dataset_frame = pd.read_csv(path)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        dataset_frame = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file type. Use CSV or Excel.")

    if "text" not in dataset_frame.columns:
        raise ValueError("Input file must contain a 'text' column.")

    dataset_frame["text"] = dataset_frame["text"].astype(str).str.strip()
    has_language_column = "language" in dataset_frame.columns
    if has_language_column:
        dataset_frame["language"] = (
            dataset_frame["language"].astype(str).str.strip().str.lower()
        )

    samples: List[Sample] = []

    for _, row in dataset_frame.iterrows():
        if limit is not None and len(samples) >= limit:
            break

        text = row["text"]
        if not text:
            continue

        if has_language_column:
            row_language = row["language"]
            if not row_language:
                continue
            if row_language not in allowed_languages:
                continue
            if row_language != resolved_language:
                continue
            samples.append(Sample(text=text, language=row_language))
            continue

        samples.append(Sample(text=text, language=resolved_language))

    logger.info(
        "Loaded samples=%s dataset=%s language=%s limit=%s",
        len(samples),
        dataset_identifier,
        resolved_language,
        limit,
    )

    return samples




def _build_response_format(
    schema: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "hate_score",
            "schema": schema or STRUCTURE_SCHEMA,
        },
    }


def _get_openrouter_client() -> OpenAI:
    """Return a cached OpenRouter-aware OpenAI client."""
    global _CLIENT
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OpenRouter API key")
    if _CLIENT is None:
        _CLIENT = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return _CLIENT


def _build_extra_headers(metadata: Optional[Dict[str, str]]) -> Dict[str, str]:
    referer = metadata.get("referer", "") if metadata else ""
    title = metadata.get("title", "hate-eval") if metadata else "hate-eval"
    return {"HTTP-Referer": referer, "X-Title": title}


def _build_structure_schema(include_score_bounds: bool = True) -> Dict[str, object]:
    schema = json.loads(json.dumps(STRUCTURE_SCHEMA))
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
    prompt_text = render_prompt(sample.language, prompt_paradigm, sample.text)
    messages = [{"role": "user", "content": prompt_text}]
    request_kwargs: Dict[str, object] = {
        "model": model,
        "temperature": 0,
        "messages": messages,
        "response_format": _build_response_format(schema),
        "timeout": timeout,
        "extra_headers": _build_extra_headers(metadata),
        "max_tokens": MAX_TOKENS_PER_COMPLETION,
    }
    extra_body: Dict[str, object] = {}
    if metadata:
        extra_body["metadata"] = metadata
    if extra_options:
        extra_body.update(extra_options)
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    return request_kwargs


def _build_default_request_kwargs(
    sample: Sample,
    model: str,
    prompt_paradigm: PromptParadigm,
    metadata: Optional[Dict[str, str]],
    timeout: int,
) -> Dict[str, object]:
    return _compose_request_kwargs(
        sample,
        model,
        prompt_paradigm,
        metadata,
        timeout,
        schema=_build_structure_schema(),
    )


def _build_anthropic_request_kwargs(
    sample: Sample,
    model: str,
    prompt_paradigm: PromptParadigm,
    metadata: Optional[Dict[str, str]],
    timeout: int,
) -> Dict[str, object]:
    request_kwargs = _compose_request_kwargs(
        sample,
        model,
        prompt_paradigm,
        metadata,
        timeout,
        schema=_build_structure_schema(include_score_bounds=False),
    )
    return request_kwargs


def _build_glm_request_kwargs(
    sample: Sample,
    model: str,
    prompt_paradigm: PromptParadigm,
    metadata: Optional[Dict[str, str]],
    timeout: int,
) -> Dict[str, object]:
    request_kwargs = _compose_request_kwargs(
        sample,
        model,
        prompt_paradigm,
        metadata,
        timeout,
        schema=_build_structure_schema(),
    )
    return request_kwargs


def _build_meta_llama_request_kwargs(
    sample: Sample,
    model: str,
    prompt_paradigm: PromptParadigm,
    metadata: Optional[Dict[str, str]],
    timeout: int,
) -> Dict[str, object]:
    request_kwargs = _compose_request_kwargs(
        sample,
        model,
        prompt_paradigm,
        metadata,
        timeout,
        schema=_build_structure_schema(),
    )
    return request_kwargs


def _build_deepseek_r1_request_kwargs(
    sample: Sample,
    model: str,
    prompt_paradigm: PromptParadigm,
    metadata: Optional[Dict[str, str]],
    timeout: int,
) -> Dict[str, object]:
    return _compose_request_kwargs(
        sample,
        model,
        prompt_paradigm,
        metadata,
        timeout,
        schema=_build_structure_schema(),
        extra_options={"reasoning": {"effort": "medium"}, "include_reasoning": True},
    )


def _build_deepseek_v3_request_kwargs(
    sample: Sample,
    model: str,
    prompt_paradigm: PromptParadigm,
    metadata: Optional[Dict[str, str]],
    timeout: int,
) -> Dict[str, object]:
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


def _build_model_request_kwargs(
    sample: Sample,
    model: str,
    prompt_paradigm: PromptParadigm,
    metadata: Optional[Dict[str, str]],
    timeout: int,
) -> Dict[str, object]:
    builder = MODEL_REQUEST_BUILDERS.get(model, _build_default_request_kwargs)
    return builder(sample, model, prompt_paradigm, metadata, timeout)


def request_score(
    sample: Sample,
    model: str,
    prompt_paradigm: PromptParadigm = "zero_shot",
    metadata: Optional[Dict[str, str]] = None,
    timeout: int = 60,
) -> ScoreResponse:
    """向 OpenRouter 发送单条提示并返回结构化结果。"""
    client = _get_openrouter_client()
    logger.info(
        "Requesting OpenRouter score language=%s model=%s paradigm=%s",
        sample.language,
        model,
        prompt_paradigm,
    )

    request_kwargs = _build_model_request_kwargs(
        sample,
        model,
        prompt_paradigm,
        metadata,
        timeout,
    )
    logger.debug("Request payload keys: %s", sorted(request_kwargs))

    response = client.chat.completions.create(**request_kwargs)
    data = response.model_dump()
    finish_reason = data["choices"][0].get("finish_reason")
    payload = _parse_structured_response(data)
    return ScoreResponse(payload=payload, finish_reason=finish_reason)


def _parse_structured_response(data: Dict[str, object]) -> Dict[str, object]:
    content = data["choices"][0]["message"]["content"]
    message = data["choices"][0]["message"]
    reason = message.get("reasoning", None)
    if isinstance(content, list):
        # 结构化输出可能以只包含一个 JSON 字典的列表形式返回
        content = content[0]
    if isinstance(content, str):
        return json.loads(content)
    return content


def run_batch(
    csv_path: str,
    model: str,
    limit: Optional[int] = None,
    language: Optional[str] = None,
    prompt_paradigm: PromptParadigm = "zero_shot",
) -> List[Dict[str, object]]:
    # samples = load_samples(csv_path, limit=limit, language=language)
    samples = load_samples(
    "/Users/baoxuan/Desktop/研究生研究/llm毕业论文/5_new_chinesehatedata_2400_balanced.xlsx",
    language="zh",   # 固定为中文
    limit=2400)
    return score_samples(samples, model=model, prompt_paradigm=prompt_paradigm)


def _score_samples_internal(
    samples: Sequence[Sample],
    model: str,
    prompt_paradigm: PromptParadigm,
) -> Tuple[List[Dict[str, object]], bool]:
    ordered_samples = list(samples)
    batch_rows: List[Dict[str, object]] = []
    aborted = False
    total_samples = len(ordered_samples)
    if total_samples == 0:
        logger.info("No samples detected for scoring; returning empty batch.")
        return batch_rows, aborted

    logger.info(
        "Scoring %d samples using ThreadPoolExecutor with %d workers",
        total_samples,
        total_samples,
    )

    with ThreadPoolExecutor(max_workers=total_samples) as executor:
        scheduled_futures = [
            (
                sample,
                executor.submit(request_score, sample, model, prompt_paradigm),
            )
            for sample in ordered_samples
        ]

        for sample, future in scheduled_futures:
            response = future.result()
            if response.finish_reason == "length":
                aborted = True
                batch_rows.clear()
                break
            batch_rows.append(
                {"text": sample.text, "language": sample.language, **response.payload}
            )

        if aborted:
            for _, future in scheduled_futures:
                if not future.done():
                    future.cancel()

    return batch_rows, aborted


def score_samples(
    samples: Sequence[Sample],
    model: str,
    prompt_paradigm: PromptParadigm = "zero_shot",
) -> List[Dict[str, object]]:
    """Request structured scores for a provided sample collection."""
    results, _ = _score_samples_internal(samples, model, prompt_paradigm)
    return results


def score_samples_with_status(
    samples: Sequence[Sample],
    model: str,
    prompt_paradigm: PromptParadigm = "zero_shot",
) -> Tuple[List[Dict[str, object]], bool]:
    """Request structured scores and return whether the batch was aborted."""
    return _score_samples_internal(samples, model, prompt_paradigm)
