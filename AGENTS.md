# Repository Guidelines

## Project Structure & Module Organization
- `pipeline.py` centralizes the hate-speech scoring pipeline: CSV ingestion, prompt construction, OpenRouter submission, and result normalization.
- `tests/` contains pytest-based unit tests that mirror the structure of `pipeline.py`; add a matching `test_<feature>.py` file whenever you introduce new helpers.
- `sample_data.csv` is the manual smoke-test fixture, while `pyproject.toml` and `uv.lock` lock dependency versions managed by uv.

## Build, Test, and Development Commands
- `uv sync` installs runtime and dev dependencies as pinned in `uv.lock`; re-run after editing `pyproject.toml`.
- `uv run python pipeline.py sample_data.csv openrouter/auto --limit 2` performs a representative batch submission; swap the CSV path, model, or limit as needed.
- `uv run pytest` executes the test suite; use `-k <pattern>` to focus on a subset during development.

## Coding Style & Naming Conventions
- Follow Python 3.11 standards with 4-space indentation, descriptive identifiers (e.g., `structured_response`, not `resp`), and module-level constants in ALL_CAPS.
- Maintain full type annotations and lightweight dataclasses for shared records; prefer pure functions for data preparation to keep the pipeline testable.
- Keep third-party usage minimal; when adding analytics helpers with polars or numpy, wrap them in composable functions and document the dependency updates in `pyproject.toml`.

## Testing Guidelines
- Use pytest with function names beginning with `test_` and fixtures from `tmp_path` or `monkeypatch` to isolate side effects.
- Ensure new features ship with unit tests covering I/O parsing, prompt formatting, and error handling; treat JSON-schema interactions as contract tests.
- Favor table-driven tests when adding languages or schema fields and keep mocks scoped to the OpenRouter transport.

## Commit & Pull Request Guidelines
- Follow the existing history: short imperative subjects (e.g., “Add OpenRouter pipeline”), optionally prefixed with scope keywords if helpful.
- Each PR should describe the motivation, list observable changes, include testing notes (`uv run pytest` output), and reference related issues.
- Provide screenshots or JSON snippets only when UI or schema changes benefit from visual confirmation, and double-check secrets are never embedded.

## Security & Configuration Tips
- Export `OPENROUTER_API_KEY` before running the pipeline locally; rely on `.env` entries only if the shell session never prints logs to shared channels.
- Use dedicated metadata (title, referer) when integrating new clients to keep OpenRouter rate-limits traceable, and avoid persisting raw responses without agreed retention policies.
