"""Sample runner covering languages, prompt paradigms, datasets, and model choices."""

from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import polars as pl

from prompts import LANGUAGES, PARADIGMS, PromptParadigm

import pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

DATASET_PATTERNS: Dict[str, Sequence[str]] = {
    "natural": [
        "data/5_new_chinesehatedata_2400_balanced.xlsx",
        "data/5_new_englishhatedata_2400_balanced.xlsx",
    ],
    # "group_swap": [
    #     "data/1b_groupswap_demensionsentence.xlsx",
    #     "data/group_swap/*.csv",
    # ],
    # "test": ["data/sample.csv"],
}

GROUP_SWAP_COLUMNS = ("具体内容-男人版", "具体内容-女人版")
GROUP_SWAP_LANGUAGE = "zh"
GROUP_SWAP_MALE_ZEROSHOT_LABEL = "男人版-zeroshot攻击性评分"
GROUP_SWAP_MALE_COT_LABEL = "男人版-cot攻击性评分"
GROUP_SWAP_MALE_COT_REASON_LABEL = "男人版-cot攻击性reason"
GROUP_SWAP_FEMALE_ZEROSHOT_LABEL = "女人版-zeroshot攻击性评分"
GROUP_SWAP_FEMALE_COT_LABEL = "女人版-cot攻击性评分"
GROUP_SWAP_FEMALE_COT_REASON_LABEL = "女人版-cot攻击性reason"

MODELS: Sequence[str] = (
    "openai/gpt-5.1",
    "anthropic/claude-opus-4.5",
    "z-ai/glm-4.6",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-v3.2-exp",
    "moonshotai/kimi-k2-thinking",
    "qwen/qwen-2.5-72b-instruct",
)

OUTPUT_ROOT = Path("outputs")
MAX_WORKERS = 8
REQUEST_TIMEOUT_SECONDS = 60
FUTURE_COMPLETION_TIMEOUT_SECONDS = 120
PROGRESS_REPORT_INTERVAL = 20
MAX_FAILURE_LOGS = 20


def discover_datasets(patterns: Iterable[str]) -> List[Path]:
    """Return sorted dataset paths for a set of glob patterns."""
    matches: List[Path] = []
    for pattern in patterns:
        pattern_matches = [Path(candidate) for candidate in glob(pattern)]
        if not pattern_matches:
            logging.warning("No datasets found for pattern: %s", pattern)
        matches.extend(pattern_matches)
    return sorted(matches)


def iter_dataset_map() -> Dict[str, List[Path]]:
    """Map dataset categories to their resolved files."""
    dataset_map: Dict[str, List[Path]] = {}
    for category, patterns in DATASET_PATTERNS.items():
        dataset_map[category] = discover_datasets(patterns)
    return dataset_map


def build_output_directory(
    dataset_type: str,
    dataset_label: str,
    language: str,
    paradigm: PromptParadigm,
) -> Path:
    """Return the directory where sample XLSX files should live."""
    return OUTPUT_ROOT / dataset_type / dataset_label / language / paradigm


def build_sample_prefix(model: str) -> str:
    """Return the filename prefix for a sample file."""
    return model.replace("/", "_")


def build_sample_path(
    dataset_type: str,
    dataset_label: str,
    model: str,
    language: str,
    paradigm: PromptParadigm,
    sample_index: int,
) -> Path:
    """Return path for a single XLSX sample."""
    output_dir = build_output_directory(dataset_type, dataset_label, language, paradigm)
    prefix = build_sample_prefix(model)
    return output_dir / f"{prefix}_sample_{sample_index:06d}.xlsx"


def discover_existing_samples(
    dataset_type: str,
    dataset_label: str,
    model: str,
    language: str,
    paradigm: PromptParadigm,
) -> Dict[int, Path]:
    """Find already generated XLSX samples for a given configuration."""
    output_dir = build_output_directory(dataset_type, dataset_label, language, paradigm)
    prefix = build_sample_prefix(model)
    pattern = str(output_dir / f"{prefix}_sample_*.xlsx")
    matches = sorted(Path(candidate) for candidate in glob(pattern))
    existing: Dict[int, Path] = {}
    for match in matches:
        stem = match.stem
        marker = stem.split("_sample_")[-1]
        if marker.isdigit():
            existing[int(marker)] = match
    if matches:
        logging.info(
            "Found %s existing samples for model=%s language=%s paradigm=%s dataset=%s",
            len(matches),
            model,
            language,
            paradigm,
            dataset_label,
        )
    return existing


def save_sample_results(
    target: Path,
    row: Dict[str, object],
    metadata: Dict[str, object],
    sample_index: int,
) -> None:
    """Write sample results to XLSX."""
    target.parent.mkdir(parents=True, exist_ok=True)
    enriched_row = {
        **row,
        "model": metadata["model"],
        "prompt_paradigm": metadata["paradigm"],
        "dataset_type": metadata["dataset_type"],
        "dataset_label": metadata["dataset_label"],
        "sample_index": sample_index,
    }
    frame = pl.DataFrame([enriched_row])
    frame.write_excel(target, worksheet="results")
    logging.info(
        "Saved sample_index=%s target=%s",
        sample_index,
        target,
    )


def score_and_save_sample(
    sample: pipeline.Sample,
    dataset_type: str,
    dataset_label: str,
    model: str,
    language: str,
    paradigm: PromptParadigm,
    sample_index: int,
    target: Path,
    request_timeout: int,
) -> str | None:
    """Score a single sample and persist its result."""
    logging.info(
        "Scoring sample model=%s language=%s paradigm=%s dataset=%s sample_index=%s",
        model,
        language,
        paradigm,
        dataset_label,
        sample_index,
    )
    try:
        response = pipeline.request_score(
            sample,
            model=model,
            prompt_paradigm=paradigm,
            timeout=request_timeout,
        )
        if response.finish_reason == "length":
            logging.warning(
                "Sample finish_reason=length model=%s language=%s paradigm=%s dataset=%s sample_index=%s",
                model,
                language,
                paradigm,
                dataset_label,
                sample_index,
            )
        row = {"text": sample.text, "language": sample.language, **response.payload}
        metadata = {
            "model": model,
            "paradigm": paradigm,
            "dataset_type": dataset_type,
            "dataset_label": dataset_label,
        }
        save_sample_results(target, row, metadata, sample_index)
        return None
    except Exception:
        logging.exception(
            "Sample error model=%s language=%s paradigm=%s dataset=%s sample_index=%s",
            model,
            language,
            paradigm,
            dataset_label,
            sample_index,
        )
        return "request_score_error"


def load_group_swap_records(dataset_path: Path) -> List[Dict[str, str]]:
    """Read group-swap rows while preserving row order."""
    def _normalize_cell(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and value != value:
            return ""
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return ""
        return text

    if dataset_path.suffix.lower() == ".csv":
        frame = pl.read_csv(str(dataset_path))
        row_iterable = frame.iter_rows(named=True)
        columns = frame.columns
    elif dataset_path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            frame = pl.read_excel(str(dataset_path))
        except ModuleNotFoundError as exc:
            if "fastexcel" not in str(exc).lower():
                raise
            import pandas as pd  # type: ignore

            pandas_frame = pd.read_excel(str(dataset_path))
            columns = list(pandas_frame.columns)
            row_iterable = pandas_frame.to_dict(orient="records")
        else:
            row_iterable = frame.iter_rows(named=True)
            columns = frame.columns
    else:
        raise ValueError(f"Unsupported group_swap file type: {dataset_path}")

    for required_column in GROUP_SWAP_COLUMNS:
        if required_column not in columns:
            raise ValueError(
                f"Missing required column in group_swap dataset: {required_column}"
            )

    records: List[Dict[str, str]] = []
    for row in row_iterable:
        male_sentence = _normalize_cell(row.get(GROUP_SWAP_COLUMNS[0]))
        female_sentence = _normalize_cell(row.get(GROUP_SWAP_COLUMNS[1]))
        if not male_sentence and not female_sentence:
            continue
        records.append(
            {
                GROUP_SWAP_COLUMNS[0]: male_sentence,
                GROUP_SWAP_COLUMNS[1]: female_sentence,
            }
        )

    logging.info(
        "Loaded %s rows from group_swap dataset=%s",
        len(records),
        dataset_path.name,
    )
    return records


def score_group_swap_dataset(
    dataset_path: Path,
    model: str,
    dataset_label: str,
    request_timeout: int,
    limit: int | None = None,
) -> Tuple[int, int]:
    """Score each group-swap row separately for male/female versions and two paradigms."""
    records = load_group_swap_records(dataset_path)
    if limit is not None:
        records = records[:limit]
    if not records:
        return 0, 0

    total_records = len(records)
    logging.info(
        "Scoring group_swap model=%s total_records=%s",
        model,
        total_records,
    )

    success_count = 0
    skipped_count = 0
    for row_index, row in enumerate(records):
        if row_index % 20 == 0:
            logging.info(
                "Group_swap progress model=%s row=%s/%s",
                model,
                row_index + 1,
                total_records,
            )

        row_output_target = build_group_swap_row_output_path(
            dataset_label=dataset_label,
            model=model,
            row_index=row_index,
        )
        if row_output_target.exists():
            skipped_count += 1
            logging.info(
                "Skip existing group_swap row model=%s dataset=%s row=%s target=%s",
                model,
                dataset_label,
                row_index + 1,
                row_output_target,
            )
            continue

        male_sentence = row[GROUP_SWAP_COLUMNS[0]]
        female_sentence = row[GROUP_SWAP_COLUMNS[1]]

        male_zeroshot_response = pipeline.request_score(
            pipeline.Sample(text=male_sentence, language=GROUP_SWAP_LANGUAGE),
            model=model,
            prompt_paradigm="zero_shot",
            timeout=request_timeout,
            prompt_mode="attack_7pt_likert",
            score_max=6,
        )
        male_cot_response = pipeline.request_score(
            pipeline.Sample(text=male_sentence, language=GROUP_SWAP_LANGUAGE),
            model=model,
            prompt_paradigm="chain_of_thought",
            timeout=request_timeout,
            prompt_mode="attack_7pt_likert",
            score_max=6,
        )
        female_zeroshot_response = pipeline.request_score(
            pipeline.Sample(text=female_sentence, language=GROUP_SWAP_LANGUAGE),
            model=model,
            prompt_paradigm="zero_shot",
            timeout=request_timeout,
            prompt_mode="attack_7pt_likert",
            score_max=6,
        )
        female_cot_response = pipeline.request_score(
            pipeline.Sample(text=female_sentence, language=GROUP_SWAP_LANGUAGE),
            model=model,
            prompt_paradigm="chain_of_thought",
            timeout=request_timeout,
            prompt_mode="attack_7pt_likert",
            score_max=6,
        )

        scored_row = {
            GROUP_SWAP_COLUMNS[0]: male_sentence,
            GROUP_SWAP_COLUMNS[1]: female_sentence,
            GROUP_SWAP_MALE_ZEROSHOT_LABEL: male_zeroshot_response.payload["score"],
            GROUP_SWAP_MALE_COT_LABEL: male_cot_response.payload["score"],
            GROUP_SWAP_MALE_COT_REASON_LABEL: male_cot_response.payload.get("reason", ""),
            GROUP_SWAP_FEMALE_ZEROSHOT_LABEL: female_zeroshot_response.payload["score"],
            GROUP_SWAP_FEMALE_COT_LABEL: female_cot_response.payload["score"],
            GROUP_SWAP_FEMALE_COT_REASON_LABEL: female_cot_response.payload.get("reason", ""),
        }
        save_group_swap_row_results(row_output_target, scored_row)
        success_count += 1

    return success_count, skipped_count


def save_group_swap_row_results(target: Path, row: Dict[str, object]) -> None:
    """Persist one group-swap row in one file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        GROUP_SWAP_COLUMNS[0],
        GROUP_SWAP_COLUMNS[1],
        GROUP_SWAP_MALE_ZEROSHOT_LABEL,
        GROUP_SWAP_MALE_COT_LABEL,
        GROUP_SWAP_MALE_COT_REASON_LABEL,
        GROUP_SWAP_FEMALE_ZEROSHOT_LABEL,
        GROUP_SWAP_FEMALE_COT_LABEL,
        GROUP_SWAP_FEMALE_COT_REASON_LABEL,
    ]
    ordered_row = {column: row.get(column, "") for column in columns}
    frame = pl.DataFrame([ordered_row])
    frame.write_excel(target)
    logging.info("Saved group_swap row target=%s", target)


def build_group_swap_row_output_path(
    dataset_label: str, model: str, row_index: int
) -> Path:
    """Return output path for an individual group_swap row."""
    prefix = build_sample_prefix(model)
    return (
        OUTPUT_ROOT
        / "group_swap"
        / dataset_label
        / f"{prefix}_group_swap_row_{row_index:06d}.xlsx"
    )


def run_experiments(limit: int | None = None) -> None:
    """Iterate every combination of language, paradigm, dataset, and model."""
    dataset_map = iter_dataset_map()
    for dataset_type, files in dataset_map.items():
        for dataset_path in files:
            dataset_label = dataset_path.stem
            logging.info(
                "Processing dataset type=%s file=%s", dataset_type, dataset_path
            )

            if dataset_type == "group_swap":
                for model in MODELS:
                    try:
                        completed_count, skipped_count = score_group_swap_dataset(
                            dataset_path,
                            model=model,
                            dataset_label=dataset_label,
                            request_timeout=REQUEST_TIMEOUT_SECONDS,
                            limit=limit,
                        )
                    except Exception:
                        logging.exception(
                            "Group_swap scoring failed model=%s dataset=%s file=%s",
                            model,
                            dataset_label,
                            dataset_path.name,
                        )
                        continue
                    logging.info(
                        "Group_swap summary model=%s dataset=%s completed=%s skipped=%s",
                        model,
                        dataset_label,
                        completed_count,
                        skipped_count,
                    )
                continue

            for model in MODELS:
                for language in LANGUAGES:
                    samples = pipeline.load_samples(
                        str(dataset_path), limit=limit, language=language
                    )
                    if not samples:
                        logging.info(
                            "No samples remain for dataset=%s language=%s",
                            dataset_label,
                            language,
                        )
                        continue
                    for paradigm in PARADIGMS:
                        existing_map = discover_existing_samples(
                            dataset_type,
                            dataset_label,
                            model,
                            language,
                            paradigm,
                        )
                        if existing_map:
                            logging.info(
                                "Using %s cached sample outputs model=%s language=%s paradigm=%s dataset=%s",
                                len(existing_map),
                                model,
                                language,
                                paradigm,
                                dataset_label,
                            )
                        executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
                        scheduled_futures: Dict[object, int] = {}
                        skipped_count = 0
                        success_count = 0
                        failed_samples: List[Dict[str, object]] = []
                        for sample_index, sample in enumerate(samples):
                            if sample_index in existing_map:
                                skipped_count += 1
                                logging.info(
                                    "Skip existing sample model=%s language=%s paradigm=%s dataset=%s sample_index=%s",
                                    model,
                                    language,
                                    paradigm,
                                    dataset_label,
                                    sample_index,
                                )
                                continue
                            target = build_sample_path(
                                dataset_type,
                                dataset_label,
                                model,
                                language,
                                paradigm,
                                sample_index,
                            )
                            future = executor.submit(
                                score_and_save_sample,
                                sample,
                                dataset_type,
                                dataset_label,
                                model,
                                language,
                                paradigm,
                                sample_index,
                                target,
                                REQUEST_TIMEOUT_SECONDS,
                            )
                            scheduled_futures[future] = sample_index
                        if not scheduled_futures:
                            logging.info(
                                "No new samples for model=%s language=%s paradigm=%s dataset=%s",
                                model,
                                language,
                                paradigm,
                                dataset_label,
                            )
                            logging.info(
                                "Completed samples model=%s language=%s paradigm=%s dataset=%s total=%s",
                                model,
                                language,
                                paradigm,
                                dataset_label,
                                len(samples),
                            )
                            executor.shutdown(wait=True)
                            continue
                        timed_out_batch = False
                        pending_futures = set(scheduled_futures.keys())
                        total_to_process = len(pending_futures)
                        completed_count = 0
                        while pending_futures:
                            done_futures, pending_futures = wait(
                                pending_futures,
                                timeout=FUTURE_COMPLETION_TIMEOUT_SECONDS,
                                return_when=FIRST_COMPLETED,
                            )
                            if not done_futures:
                                timed_out_batch = True
                                for future in pending_futures:
                                    failed_samples.append(
                                        {
                                            "sample_index": scheduled_futures[future],
                                            "error": "batch_wait_timeout",
                                        }
                                    )
                                    if not future.cancel():
                                        logging.debug(
                                            "Could not cancel sample_index=%s (already running)",
                                            scheduled_futures[future],
                                        )
                                completed_count += len(pending_futures)
                                logging.warning(
                                    "Timeout waiting for sample tasks model=%s language=%s paradigm=%s dataset=%s timed_out=%s pending=%s",
                                    model,
                                    language,
                                    paradigm,
                                    dataset_label,
                                    len(failed_samples),
                                    len(pending_futures),
                                )
                                break
                            for future in done_futures:
                                completed_count += 1
                                sample_index = scheduled_futures[future]
                                try:
                                    error_message = future.result()
                                except Exception as error:
                                    error_message = f"{type(error).__name__}: {error}"
                                if error_message is None:
                                    success_count += 1
                                else:
                                    failed_samples.append(
                                        {
                                            "sample_index": sample_index,
                                            "error": error_message,
                                        }
                                    )
                                if (
                                    completed_count % PROGRESS_REPORT_INTERVAL == 0
                                    or completed_count == total_to_process
                                ):
                                    logging.info(
                                        "Progress model=%s language=%s paradigm=%s dataset=%s completed=%s/%s success=%s failed=%s skipped=%s",
                                        model,
                                        language,
                                        paradigm,
                                        dataset_label,
                                        completed_count,
                                        total_to_process,
                                        success_count,
                                        len(failed_samples),
                                        skipped_count,
                                    )
                        if timed_out_batch:
                            logging.warning(
                                "Shutting down executor without waiting because batch timed out in order to avoid full pipeline stall."
                            )
                            executor.shutdown(wait=False, cancel_futures=True)
                        else:
                            executor.shutdown(wait=True)
                        logging.info(
                            "Worker summary model=%s language=%s paradigm=%s dataset=%s completed=%s/%s success=%s failed=%s skipped=%s",
                            model,
                            language,
                            paradigm,
                            dataset_label,
                            completed_count if scheduled_futures else 0,
                            total_to_process if scheduled_futures else 0,
                            success_count,
                            len(failed_samples),
                            skipped_count,
                        )
                        if failed_samples:
                            for failure in failed_samples[:MAX_FAILURE_LOGS]:
                                logging.warning(
                                    "Failed sample_index=%s reason=%s",
                                    failure["sample_index"],
                                    failure["error"],
                                )
                            if len(failed_samples) > MAX_FAILURE_LOGS:
                                logging.warning(
                                    "More failures truncated: total=%s",
                                    len(failed_samples),
                                )
                        logging.info(
                            "Completed samples model=%s language=%s paradigm=%s dataset=%s total=%s",
                            model,
                            language,
                            paradigm,
                            dataset_label,
                            len(samples),
                        )
    logging.info("All samples completed successfully.")


if __name__ == "__main__":
    run_experiments(limit=None)
