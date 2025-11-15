"""用于数据提取与 OpenRouter 提交的精简流水线。"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib import request as urlrequest
from urllib.error import HTTPError

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

CHINESE_PROMPT = (
    "请根据以下定义评估该句子是否属于仇恨言论，并为其攻击性/仇恨强度在 0–5 分范围内评分。"
    "请严格只输出一个 0–5 的数字。句子：\n\"{text}\""
)

ENGLISH_PROMPT = (
    "Evaluate whether the following sentence constitutes hate speech per the provided definition."
    "Only output a single number from 0 to 5. Sentence:\n\"{text}\""
)

PROMPTS = {
    "zh": CHINESE_PROMPT,
    "en": ENGLISH_PROMPT,
}

STRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 5},
        "reason": {"type": "string"},
    },
    "required": ["score"],
    "additionalProperties": False,
}


@dataclass
class Sample:
    text: str
    language: str


def load_samples(path: str, limit: Optional[int] = None) -> List[Sample]:
    """从含有 `text` 与 `language` 列的 CSV 中载入样本。"""
    samples: List[Sample] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if limit is not None and len(samples) >= limit:
                break
            text = (row.get("text") or "").strip()
            language = (row.get("language") or "").strip().lower()
            if not text or language not in PROMPTS:
                continue
            samples.append(Sample(text=text, language=language))
    return samples


def build_prompt(sample: Sample) -> str:
    try:
        template = PROMPTS[sample.language]
    except KeyError as exc:  # pragma: no cover - 防御性分支
        raise ValueError(f"Unsupported language: {sample.language}") from exc
    return template.format(text=sample.text)


def _build_response_format() -> Dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "hate_score",
            "schema": STRUCTURE_SCHEMA,
        },
    }


def request_score(
    sample: Sample,
    model: str,
    api_key: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    timeout: int = 60,
) -> Dict[str, object]:
    """向 OpenRouter 发送单条提示并返回结构化结果。"""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OpenRouter API key")

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": build_prompt(sample)}],
        "response_format": _build_response_format(),
    }
    if metadata:
        payload["metadata"] = metadata

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": metadata.get("referer", "") if metadata else "",
        "X-Title": metadata.get("title", "hate-eval") if metadata else "hate-eval",
        "Content-Type": "application/json",
    }

    data = _post_json(OPENROUTER_URL, headers=headers, payload=payload, timeout=timeout)
    return _parse_structured_response(data)


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, object], timeout: int) -> Dict[str, object]:
    encoded = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=encoded, headers=headers, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except HTTPError as exc:  # pragma: no cover - 网络保护
        raise RuntimeError(f"OpenRouter request failed: {exc.read().decode('utf-8', 'ignore')}") from exc


def _parse_structured_response(data: Dict[str, object]) -> Dict[str, object]:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:  # pragma: no cover - 防御性分支
        raise ValueError("Malformed OpenRouter response") from exc

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
    api_key: Optional[str] = None,
) -> List[Dict[str, object]]:
    samples = load_samples(csv_path, limit=limit)
    results = []
    for sample in samples:
        result = request_score(sample, model=model, api_key=api_key)
        results.append({"text": sample.text, "language": sample.language, **result})
    return results


def _cli():  # pragma: no cover - 仅供命令行便利使用
    import argparse

    parser = argparse.ArgumentParser(description="将仇恨言论提示提交至 OpenRouter。")
    parser.add_argument("csv", help="包含 text,language 列的 CSV 路径")
    parser.add_argument("model", help="OpenRouter 模型标识符")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 行")
    parser.add_argument("--api-key", dest="api_key", help="显式提供 OpenRouter API Key")

    args = parser.parse_args()
    results = run_batch(args.csv, args.model, limit=args.limit, api_key=args.api_key)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover - 命令行入口
    _cli()
