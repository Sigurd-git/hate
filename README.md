# Hate Speech Prompt Runner

This repository currently focuses on two steps of the planned study pipeline:

1. **Data extraction** – load bilingual hate-speech stimuli from a flat CSV file.
2. **Structured OpenRouter submission** – send each sample to a selected model while
   forcing JSON-schema output (score + optional reason).

The code is intentionally flat (single `pipeline.py`) so it is easy to audit and extend.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install pytest
```

### Prepare data

Use any CSV file with at least two columns:

| text | language |
| --- | --- |
| `"I hate all dogs"` | `en` |
| `"我喜欢每个人"` | `zh` |

A tiny example is provided in `sample_data.csv`.

### Run a batch

```bash
export OPENROUTER_API_KEY="sk-or-..."
python pipeline.py sample_data.csv openrouter/auto --limit 2
```

The script prints JSON with the structured responses. Every request uses
OpenRouter's structured-output channel (`response_format=json_schema`) so models
that support deterministic JSON will always be preferred.

### Run tests

```bash
pytest
```

The suite checks CSV loading, prompt construction, and verifies that requests are
wired to use the structured schema.

## What is implemented

- ✅ CSV ingestion with language-aware prompt templates.
- ✅ Deterministic prompts for both Chinese and English sentences.
- ✅ OpenRouter client that requests structured output (score 0–5 + reason).
- ✅ Minimal CLI for ad-hoc runs plus pytest coverage for all critical helpers.

## TODO / next steps

- ⏳ Expand the schema to carry model confidence, binary flags, and metadata.
- ⏳ Implement batching + backoff utilities for high-throughput evaluation.
- ⏳ Add downstream fairness metrics (ΔHATE-rate, Δs, flip rate, etc.).
- ⏳ Persist structured responses (e.g., parquet) for later statistical analysis.
- ⏳ Add prompt variants for few-shot and CoT paradigms.
