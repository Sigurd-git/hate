#!/usr/bin/env python3
import argparse
import glob
import logging
import os
import re
from pathlib import Path

import pandas as pd

CHINESE_PERIOD = "。"
DEFAULT_MERGED_FILE = (
    "outputs/natural/5_new_chinesehatedata_2400_balanced/zh/merged_zh_models.xlsx"
)
DEFAULT_COT_DIR = (
    "outputs/natural/5_new_chinesehatedata_2400_balanced/zh/chain_of_thought"
)
CLAUDE_FILENAME_PATTERN = "anthropic_claude-opus-4.5_sample_*.xlsx"
SAMPLE_INDEX_PATTERN = re.compile(r"sample_(\d{6})\.xlsx$")


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for removing Claude samples with missing periods."""
    argument_parser = argparse.ArgumentParser(
        description=(
            "Delete Claude chain-of-thought sample files whose CoT reason does not end with "
            "the Chinese period character."
        )
    )
    argument_parser.add_argument(
        "--merged-file",
        default=DEFAULT_MERGED_FILE,
        help="Path to merged_zh_models.xlsx used to detect missing-period samples.",
    )
    argument_parser.add_argument(
        "--cot-dir",
        default=DEFAULT_COT_DIR,
        help="Directory containing chain_of_thought Excel files.",
    )
    argument_parser.add_argument(
        "--cot-column",
        default="claude4.5_cot_reason",
        help="Column name that stores Claude CoT reason text.",
    )
    argument_parser.add_argument(
        "--index-column",
        default="sample_index",
        help="Column name that stores sample index values.",
    )
    argument_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be deleted without removing them.",
    )
    return argument_parser.parse_args()


def configure_logging() -> None:
    """Configure logging for clear, stepwise processing details."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_missing_period_indices(
    merged_file_path: str, cot_column_name: str, index_column_name: str
) -> set[int]:
    """Load sample indices whose Claude CoT reason does not end with a Chinese period."""
    merged_dataframe = pd.read_excel(merged_file_path)
    if cot_column_name not in merged_dataframe.columns:
        raise KeyError(f"Missing CoT column: {cot_column_name}")
    if index_column_name not in merged_dataframe.columns:
        raise KeyError(f"Missing index column: {index_column_name}")

    normalized_cot_text = (
        merged_dataframe[cot_column_name].astype("string").fillna("").str.strip()
    )
    missing_period_mask = ~normalized_cot_text.str.endswith(CHINESE_PERIOD)
    missing_period_indices = merged_dataframe.loc[missing_period_mask, index_column_name]
    return set(missing_period_indices.astype(int).tolist())


def collect_claude_files(chain_of_thought_dir: str) -> list[str]:
    """Collect Claude sample files using glob and return sorted paths."""
    search_pattern = os.path.join(chain_of_thought_dir, CLAUDE_FILENAME_PATTERN)
    matched_paths = glob.glob(search_pattern)
    return sorted(matched_paths)


def extract_sample_index(file_path: str) -> int | None:
    """Extract the numeric sample index from a Claude filename."""
    match = SAMPLE_INDEX_PATTERN.search(os.path.basename(file_path))
    if not match:
        return None
    return int(match.group(1))


def remove_missing_period_files(
    file_paths: list[str], missing_period_indices: set[int], dry_run: bool
) -> tuple[int, int]:
    """Delete files that match missing-period sample indices, returning deleted/total counts."""
    deleted_count = 0
    for file_path in file_paths:
        sample_index = extract_sample_index(file_path)
        if sample_index is None:
            logging.warning("Skipping file with unexpected name: %s", file_path)
            continue
        if sample_index not in missing_period_indices:
            continue

        if dry_run:
            logging.info("Dry run: would delete %s", file_path)
            continue

        Path(file_path).unlink()
        deleted_count += 1
        logging.info("Deleted %s", file_path)
    return deleted_count, len(file_paths)


def main() -> None:
    """Orchestrate loading missing-period indices and deleting Claude sample files."""
    configure_logging()
    arguments = parse_arguments()

    logging.info("Loading missing-period indices from %s", arguments.merged_file)
    missing_period_indices = load_missing_period_indices(
        arguments.merged_file,
        arguments.cot_column,
        arguments.index_column,
    )
    logging.info("Missing-period sample count: %d", len(missing_period_indices))

    file_paths = collect_claude_files(arguments.cot_dir)
    logging.info("Claude files discovered: %d", len(file_paths))

    deleted_count, total_files = remove_missing_period_files(
        file_paths, missing_period_indices, arguments.dry_run
    )
    logging.info(
        "Processed %d files, deleted %d files (dry_run=%s).",
        total_files,
        deleted_count,
        arguments.dry_run,
    )


if __name__ == "__main__":
    main()
