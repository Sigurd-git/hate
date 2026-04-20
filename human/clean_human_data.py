from __future__ import annotations

import argparse
from pathlib import Path

from human_cleaning import CleaningConfig, run_cleaning_pipeline
from human_cleaning.config import (
    DEFAULT_LONG_PATH,
    DEFAULT_MANUAL_FINAL_EXCLUSION_PATH,
    DEFAULT_MANUAL_FRAUD_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WIDE_PATH,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run participant-level and trial-level cleaning for the human aggression-rating dataset."
    )
    parser.add_argument("--wide-path", type=Path, default=DEFAULT_WIDE_PATH, help="Path to the wide participant master CSV.")
    parser.add_argument("--long-path", type=Path, default=DEFAULT_LONG_PATH, help="Path to the long trial-level CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for cleaned outputs and QC reports.")
    parser.add_argument(
        "--manual-fraud-path",
        type=Path,
        default=DEFAULT_MANUAL_FRAUD_PATH,
        help="Optional CSV containing session_id values to hard-exclude as manual fraud clusters.",
    )
    parser.add_argument(
        "--manual-final-exclusion-path",
        type=Path,
        default=DEFAULT_MANUAL_FINAL_EXCLUSION_PATH,
        help="Optional CSV containing session_id values to exclude only from final_clean exports.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    manual_fraud_path = args.manual_fraud_path if args.manual_fraud_path.exists() else None
    manual_final_exclusion_path = args.manual_final_exclusion_path if args.manual_final_exclusion_path.exists() else None
    outputs = run_cleaning_pipeline(
        wide_path=args.wide_path,
        long_path=args.long_path,
        output_dir=args.output_dir,
        manual_fraud_path=manual_fraud_path,
        manual_final_exclusion_path=manual_final_exclusion_path,
        config=CleaningConfig(),
    )

    participant_qc = outputs["participant_qc"]
    print("Cleaning completed.")
    print(f"Participant QC master: {outputs['participant_qc_path']}")
    print(f"Hard-cleaned participants: {outputs['hard_cleaned_path']}")
    print(f"Strict-cleaned participants: {outputs['strict_cleaned_path']}")
    print(f"Enriched trials: {outputs['long_enriched_path']}")
    print(f"Suspicious review CSV: {outputs['suspicious_review_path']}")
    print(f"Final clean wide CSV: {outputs['final_clean_wide_csv_path']}")
    print(f"Final clean wide XLSX: {outputs['final_clean_wide_xlsx_path']}")
    print(f"Final clean long CSV: {outputs['final_clean_long_csv_path']}")
    print(f"Final clean long XLSX: {outputs['final_clean_long_xlsx_path']}")
    print(f"Ultra-clean participants CSV: {outputs['ultra_cleaned_path']}")
    print(f"Ultra-clean wide CSV: {outputs['ultra_clean_wide_csv_path']}")
    print(f"Ultra-clean wide XLSX: {outputs['ultra_clean_wide_xlsx_path']}")
    print(f"Ultra-clean long CSV: {outputs['ultra_clean_long_csv_path']}")
    print(f"Ultra-clean long XLSX: {outputs['ultra_clean_long_xlsx_path']}")
    print(f"Summary report: {outputs['summary_report_path']}")
    print(f"Total participants: {len(participant_qc)}")
    print(f"Retained after hard cleaning: {(~participant_qc['hard_exclusion_flag']).sum()}")
    print(f"Retained after strict cleaning: {(~participant_qc['strict_exclusion_flag']).sum()}")
    print(f"Flagged for suspicious-behavior review: {participant_qc['suspicious_behavior_review_flag'].sum()}")
    print(f"Retained after ultra-clean screen: {len(outputs['ultra_clean_participants'])}")


if __name__ == "__main__":
    main()
