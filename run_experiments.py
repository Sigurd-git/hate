"""Batch runner covering languages, prompt paradigms, datasets, and model choices."""

from __future__ import annotations

import logging
from glob import glob
from pathlib import Path
from typing import Dict, List, Sequence

import polars as pl

from prompts import LANGUAGES, PARADIGMS, PromptParadigm

import pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

DATASET_PATTERNS: Dict[str, str] = {
    # "natural": "data/natural/*.csv",
    # "group_swap": "data/group_swap/*.csv",
    "test": "data/sample.csv",
}

MODELS: Sequence[str] = (
    "openai/gpt-5.1",
    "anthropic/claude-sonnet-4.5",
    # "z-ai/glm-4.6",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-v3.2-exp",
    # "baidu/ernie-x1",
    # "bigscience/bloomz",
)

OUTPUT_ROOT = Path("outputs")
BATCH_SIZE = 10


def discover_datasets(pattern: str) -> List[Path]:
    """Return sorted dataset paths for a given glob pattern."""
    matches = sorted(Path(candidate) for candidate in glob(pattern))
    if not matches:
        logging.warning("No datasets found for pattern: %s", pattern)
    return matches


def iter_dataset_map() -> Dict[str, List[Path]]:
    """Map dataset categories to their resolved files."""
    dataset_map: Dict[str, List[Path]] = {}
    for category, pattern in DATASET_PATTERNS.items():
        dataset_map[category] = discover_datasets(pattern)
    return dataset_map


def build_output_directory(
    dataset_type: str,
    dataset_label: str,
    language: str,
    paradigm: PromptParadigm,
) -> Path:
    """Return the directory where batch XLSX files should live."""
    return OUTPUT_ROOT / dataset_type / dataset_label / language / paradigm


def build_batch_prefix(model: str) -> str:
    """Return the filename prefix for a batch file."""
    return model.replace("/", "_")


def build_batch_path(
    dataset_type: str,
    dataset_label: str,
    model: str,
    language: str,
    paradigm: PromptParadigm,
    batch_index: int,
) -> Path:
    """Return path for a single XLSX batch."""
    output_dir = build_output_directory(dataset_type, dataset_label, language, paradigm)
    prefix = build_batch_prefix(model)
    return output_dir / f"{prefix}_batch_{batch_index:03d}.xlsx"


def discover_existing_batches(
    dataset_type: str,
    dataset_label: str,
    model: str,
    language: str,
    paradigm: PromptParadigm,
) -> Dict[int, Path]:
    """Find already generated XLSX batches for a given configuration."""
    output_dir = build_output_directory(dataset_type, dataset_label, language, paradigm)
    prefix = build_batch_prefix(model)
    pattern = str(output_dir / f"{prefix}_batch_*.xlsx")
    matches = sorted(Path(candidate) for candidate in glob(pattern))
    existing: Dict[int, Path] = {}
    for match in matches:
        stem = match.stem
        marker = stem.split("_batch_")[-1]
        if marker.isdigit():
            existing[int(marker)] = match
    if matches:
        logging.info(
            "Found %s existing batches for model=%s language=%s paradigm=%s dataset=%s",
            len(matches),
            model,
            language,
            paradigm,
            dataset_label,
        )
    return existing


def save_batch_results(
    target: Path,
    rows: List[Dict[str, object]],
    metadata: Dict[str, object],
) -> None:
    """Write batch results to XLSX."""
    target.parent.mkdir(parents=True, exist_ok=True)
    enriched_rows = [
        {
            **row,
            "model": metadata["model"],
            "prompt_paradigm": metadata["paradigm"],
            "dataset_type": metadata["dataset_type"],
            "dataset_label": metadata["dataset_label"],
            "batch_index": metadata["batch_index"],
        }
        for row in rows
    ]
    frame = pl.DataFrame(enriched_rows)
    frame.write_excel(target, worksheet="results")
    logging.info(
        "Saved batch_index=%s samples=%s target=%s",
        metadata["batch_index"],
        len(rows),
        target,
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
                        existing_map = discover_existing_batches(
                            dataset_type,
                            dataset_label,
                            model,
                            language,
                            paradigm,
                        )
                        total_batches = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE
                        for batch_index in range(total_batches):
                            if batch_index in existing_map:
                                logging.info(
                                    "Skip existing batch model=%s language=%s paradigm=%s dataset=%s batch=%s",
                                    model,
                                    language,
                                    paradigm,
                                    dataset_label,
                                    batch_index,
                                )
                                continue
                            start = batch_index * BATCH_SIZE
                            end = min(start + BATCH_SIZE, len(samples))
                            batch_samples = samples[start:end]
                            if not batch_samples:
                                continue
                            logging.info(
                                "Scoring batch model=%s language=%s paradigm=%s dataset=%s batch=%s",
                                model,
                                language,
                                paradigm,
                                dataset_label,
                                batch_index,
                            )
                            results, batch_aborted = pipeline.score_samples_with_status(
                                batch_samples,
                                model=model,
                                prompt_paradigm=paradigm,
                            )
                            if batch_aborted:
                                logging.warning(
                                    "Skip saving batch due to length finish_reason model=%s language=%s paradigm=%s dataset=%s batch=%s",
                                    model,
                                    language,
                                    paradigm,
                                    dataset_label,
                                    batch_index,
                                )
                                continue
                            metadata = {
                                "model": model,
                                "paradigm": paradigm,
                                "dataset_type": dataset_type,
                                "dataset_label": dataset_label,
                                "batch_index": batch_index,
                            }
                            target = build_batch_path(
                                dataset_type,
                                dataset_label,
                                model,
                                language,
                                paradigm,
                                batch_index,
                            )
                            save_batch_results(target, results, metadata)


if __name__ == "__main__":
    run_experiments(limit=None)
