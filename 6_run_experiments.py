"""Sample runner covering languages, prompt paradigms, datasets, and model choices."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    # "group_swap": ["data/group_swap/*.csv"],
    # "test": ["data/sample.csv"],
}

MODELS: Sequence[str] = (
    # "openai/gpt-5.1",
    # "anthropic/claude-opus-4.5",
    # "z-ai/glm-4.6",
    # "meta-llama/llama-4-maverick",
    # "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-v3.2-exp",
    # "baidu/ernie-x1",
    # "bigscience/bloomz",
)

OUTPUT_ROOT = Path("outputs")
MAX_WORKERS = 16


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
) -> None:
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
    except Exception:
        logging.exception(
            "Sample error model=%s language=%s paradigm=%s dataset=%s sample_index=%s",
            model,
            language,
            paradigm,
            dataset_label,
            sample_index,
        )
        return


def run_experiments(limit: int | None = None) -> None:
    """Iterate every combination of language, paradigm, dataset, and model."""
    dataset_map = iter_dataset_map()
    for dataset_type, files in dataset_map.items():
        for dataset_path in files:
            dataset_label = dataset_path.stem
            logging.info(
                "Processing dataset type=%s file=%s", dataset_type, dataset_path
            )
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
                        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                            scheduled_futures = []
                            for sample_index, sample in enumerate(samples):
                                if sample_index in existing_map:
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
                                scheduled_futures.append(
                                    executor.submit(
                                        score_and_save_sample,
                                        sample,
                                        dataset_type,
                                        dataset_label,
                                        model,
                                        language,
                                        paradigm,
                                        sample_index,
                                        target,
                                    )
                                )
                            for future in as_completed(scheduled_futures):
                                future.result()
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
