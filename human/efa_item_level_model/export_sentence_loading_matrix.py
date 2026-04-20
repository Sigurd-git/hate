from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
VERSION_DIRECTORY_MAP = {
    "男人版": "male_version_kaiser_efa",
    "女人版": "female_version_kaiser_efa",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one-row-per-sentence loading matrices for male and female sentence-level EFA results."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory containing the male_version_10factor and female_version_10factor folders.",
    )
    return parser.parse_args()


def build_sentence_loading_table(version_output_dir: Path) -> pd.DataFrame:
    loading_path = version_output_dir / "sentence_factor_loadings.csv"
    loading_frame = pd.read_csv(loading_path)
    factor_columns = sorted(
        [column_name for column_name in loading_frame.columns if column_name.startswith("Factor_")],
        key=lambda column_name: int(column_name.split("_")[1]),
    )

    selected_columns = [
        "template_id",
        "shown_version",
        "shown_text",
        "dimension_1",
        *factor_columns,
    ]
    missing_columns = [column_name for column_name in selected_columns if column_name not in loading_frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {loading_path}: {missing_columns}")

    sentence_loading_table = loading_frame.loc[:, selected_columns].copy()
    sentence_loading_table = sentence_loading_table.rename(
        columns={
            "template_id": "template_id",
            "shown_version": "shown_version",
            "shown_text": "sentence_text",
            "dimension_1": "original_dimension",
        }
    )
    sentence_loading_table = sentence_loading_table.sort_values(["template_id"]).reset_index(drop=True)
    return sentence_loading_table


def save_sentence_loading_table(version_output_dir: Path, sentence_loading_table: pd.DataFrame) -> None:
    csv_path = version_output_dir / "sentence_loading_matrix_all_explored_factors.csv"
    xlsx_path = version_output_dir / "sentence_loading_matrix_all_explored_factors.xlsx"
    sentence_loading_table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    sentence_loading_table.to_excel(xlsx_path, index=False)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved XLSX: {xlsx_path}")
    print(f"Rows: {len(sentence_loading_table)}")


def main() -> None:
    args = parse_arguments()
    for shown_version, directory_name in VERSION_DIRECTORY_MAP.items():
        version_output_dir = args.output_root / directory_name
        sentence_loading_table = build_sentence_loading_table(version_output_dir)
        save_sentence_loading_table(version_output_dir, sentence_loading_table)
        print(f"Finished exporting merged loading table for {shown_version}.")


if __name__ == "__main__":
    main()
