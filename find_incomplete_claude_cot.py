#!/usr/bin/env python3
import argparse
import glob
import logging

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for locating incomplete Claude CoT entries."""
    argument_parser = argparse.ArgumentParser(
        description=(
            "Locate Claude CoT reasons that do not end with the Chinese full stop '。' "
            "and report their sample_index values."
        )
    )
    argument_parser.add_argument(
        "--input",
        default="outputs/natural/5_new_chinesehatedata_2400_balanced/zh/merged_zh_models.xlsx",
        help="Glob pattern or file path to the Excel file(s) to scan.",
    )
    argument_parser.add_argument(
        "--cot-column",
        default="claude4.5_cot_reason",
        help="Column name that contains the Claude CoT reason text.",
    )
    argument_parser.add_argument(
        "--index-column",
        default="sample_index",
        help="Column name that contains the sample index identifiers.",
    )
    argument_parser.add_argument(
        "--sheet-name",
        default=0,
        help="Sheet name or index to read from the Excel workbook.",
    )
    argument_parser.add_argument(
        "--incomplete-output",
        default="outputs/claude_cot_missing_period.xlsx",
        help="Path for the Excel file containing Claude CoT entries missing the Chinese period.",
    )
    argument_parser.add_argument(
        "--complete-output",
        default="outputs/claude_cot_with_period.xlsx",
        help="Path for the Excel file containing Claude CoT entries that end with the Chinese period.",
    )
    return argument_parser.parse_args()


def configure_logging() -> None:
    """Configure structured logging for clear step-by-step progress updates."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_excel_files(file_pattern: str, sheet_name: str | int) -> list[tuple[str, pd.DataFrame]]:
    """Load Excel files matched by the glob pattern and return path/DataFrame pairs."""
    matched_paths = sorted(glob.glob(file_pattern))
    if not matched_paths:
        raise FileNotFoundError(f"No files matched pattern: {file_pattern}")

    workbook_entries: list[tuple[str, pd.DataFrame]] = []
    for file_path in matched_paths:
        logging.info("Reading Excel file: %s", file_path)
        workbook_entries.append((file_path, pd.read_excel(file_path, sheet_name=sheet_name)))
    return workbook_entries


def find_incomplete_cot_indices(
    dataframe: pd.DataFrame,
    cot_column_name: str,
    index_column_name: str,
) -> tuple[pd.Series, pd.Series]:
    """Return the sample indices missing the Chinese full stop and a mask of empty CoT entries."""
    if cot_column_name not in dataframe.columns:
        raise KeyError(f"Missing CoT column: {cot_column_name}")
    if index_column_name not in dataframe.columns:
        raise KeyError(f"Missing index column: {index_column_name}")

    cot_text = dataframe[cot_column_name]
    normalized_cot_text = cot_text.astype("string").fillna("").str.strip()
    empty_cot_mask = normalized_cot_text.eq("")
    missing_period_mask = ~normalized_cot_text.str.endswith("。")
    missing_period_indices = dataframe.loc[missing_period_mask, index_column_name]
    return missing_period_indices, empty_cot_mask


def split_cot_by_period(
    dataframe: pd.DataFrame, cot_column_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into rows with and without the Chinese period ending."""
    cot_text = dataframe[cot_column_name]
    normalized_cot_text = cot_text.astype("string").fillna("").str.strip()
    missing_period_mask = ~normalized_cot_text.str.endswith("。")
    incomplete_dataframe = dataframe.loc[missing_period_mask].copy()
    complete_dataframe = dataframe.loc[~missing_period_mask].copy()
    return incomplete_dataframe, complete_dataframe


def main() -> None:
    """Run the end-to-end scan and print sample_index values for incomplete Claude CoT entries."""
    configure_logging()
    arguments = parse_arguments()

    workbook_entries = load_excel_files(arguments.input, arguments.sheet_name)
    incomplete_entries: list[pd.DataFrame] = []
    complete_entries: list[pd.DataFrame] = []
    for file_path, dataframe in workbook_entries:
        logging.info("Scanning %s for incomplete Claude CoT entries.", file_path)
        missing_period_indices, empty_cot_mask = find_incomplete_cot_indices(
            dataframe,
            arguments.cot_column,
            arguments.index_column,
        )
        incomplete_dataframe, complete_dataframe = split_cot_by_period(
            dataframe, arguments.cot_column
        )
        incomplete_dataframe["source_file"] = file_path
        complete_dataframe["source_file"] = file_path
        incomplete_entries.append(incomplete_dataframe)
        complete_entries.append(complete_dataframe)
        empty_count = int(empty_cot_mask.sum())
        logging.info("Empty CoT entries: %d", empty_count)
        logging.info("Entries without trailing '。': %d", len(missing_period_indices))

        print(f"\nFile: {file_path}")
        print("sample_index values with Claude CoT not ending in '。':")
        for sample_index_value in missing_period_indices.tolist():
            print(sample_index_value)

    merged_incomplete = (
        pd.concat(incomplete_entries, ignore_index=True) if incomplete_entries else pd.DataFrame()
    )
    merged_complete = (
        pd.concat(complete_entries, ignore_index=True) if complete_entries else pd.DataFrame()
    )
    logging.info(
        "Writing %d incomplete entries to %s",
        len(merged_incomplete),
        arguments.incomplete_output,
    )
    merged_incomplete.to_excel(arguments.incomplete_output, index=False)
    logging.info(
        "Writing %d complete entries to %s",
        len(merged_complete),
        arguments.complete_output,
    )
    merged_complete.to_excel(arguments.complete_output, index=False)


if __name__ == "__main__":
    main()
