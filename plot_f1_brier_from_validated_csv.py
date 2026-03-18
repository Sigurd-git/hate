from __future__ import annotations

from pathlib import Path

import pandas as pd

from plot_f1_brier import (
    MODEL_LABELS,
    build_color_mapping,
    compute_best_models,
    create_dumbbell_figure,
    create_scatter_figure,
    create_single_metric_bar_figure,
)


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_ROOT / "artifacts" / "validated_model_f1_brier_skip_missing.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "validated_csv_hate_f1_brier_best_models.csv"
OUTPUT_SCATTER_PNG = OUTPUT_DIR / "validated_csv_hate_f1_brier_2x2_scatter.png"
OUTPUT_SCATTER_PDF = OUTPUT_DIR / "validated_csv_hate_f1_brier_2x2_scatter.pdf"
OUTPUT_F1_BAR_PNG = OUTPUT_DIR / "validated_csv_hate_f1_2x2_bars.png"
OUTPUT_F1_BAR_PDF = OUTPUT_DIR / "validated_csv_hate_f1_2x2_bars.pdf"
OUTPUT_BRIER_BAR_PNG = OUTPUT_DIR / "validated_csv_hate_brier_2x2_bars.png"
OUTPUT_BRIER_BAR_PDF = OUTPUT_DIR / "validated_csv_hate_brier_2x2_bars.pdf"
OUTPUT_F1_DUMBBELL_PNG = OUTPUT_DIR / "validated_csv_hate_f1_2x2_dumbbell.png"
OUTPUT_F1_DUMBBELL_PDF = OUTPUT_DIR / "validated_csv_hate_f1_2x2_dumbbell.pdf"
OUTPUT_BRIER_DUMBBELL_PNG = OUTPUT_DIR / "validated_csv_hate_brier_2x2_dumbbell.png"
OUTPUT_BRIER_DUMBBELL_PDF = OUTPUT_DIR / "validated_csv_hate_brier_2x2_dumbbell.pdf"


def load_metrics_table(csv_path: Path) -> pd.DataFrame:
    """Load a tidy metrics CSV and attach the plotting display labels."""
    dataframe = pd.read_csv(csv_path)
    required_columns = {"language", "model", "setting", "n_scored", "f1", "brier"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input CSV is missing required columns: {missing_text}")

    dataframe = dataframe.copy()
    dataframe["model_label"] = dataframe["model"].map(MODEL_LABELS).fillna(dataframe["model"])
    dataframe["language"] = dataframe["language"].astype(str)
    dataframe["setting"] = dataframe["setting"].astype(str)
    return dataframe


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dataframe = load_metrics_table(INPUT_CSV)
    color_mapping = build_color_mapping(metrics_dataframe["model_label"].unique())
    summary_dataframe = compute_best_models(metrics_dataframe)
    summary_dataframe.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    scatter_path, scatter_panel_counts = create_scatter_figure(
        dataframe=metrics_dataframe,
        png_path=OUTPUT_SCATTER_PNG,
        pdf_path=OUTPUT_SCATTER_PDF,
        color_mapping=color_mapping,
    )
    f1_bar_path, f1_bar_panel_counts = create_single_metric_bar_figure(
        dataframe=metrics_dataframe,
        png_path=OUTPUT_F1_BAR_PNG,
        pdf_path=OUTPUT_F1_BAR_PDF,
        color_mapping=color_mapping,
        metric_column="f1",
        metric_title="F1 across language and prompting settings (validated CSV)",
        direction_label="F1 ↑",
        larger_is_better=True,
    )
    brier_bar_path, brier_bar_panel_counts = create_single_metric_bar_figure(
        dataframe=metrics_dataframe,
        png_path=OUTPUT_BRIER_BAR_PNG,
        pdf_path=OUTPUT_BRIER_BAR_PDF,
        color_mapping=color_mapping,
        metric_column="brier",
        metric_title="Brier score across language and prompting settings (validated CSV)",
        direction_label="Brier score ↓",
        larger_is_better=False,
    )
    f1_dumbbell_path, f1_dumbbell_panel_counts = create_dumbbell_figure(
        dataframe=metrics_dataframe,
        png_path=OUTPUT_F1_DUMBBELL_PNG,
        pdf_path=OUTPUT_F1_DUMBBELL_PDF,
        color_mapping=color_mapping,
        metric_column="f1",
        metric_title="Within-model F1 comparisons across prompting and language conditions (validated CSV)",
        larger_is_better=True,
    )
    brier_dumbbell_path, brier_dumbbell_panel_counts = create_dumbbell_figure(
        dataframe=metrics_dataframe,
        png_path=OUTPUT_BRIER_DUMBBELL_PNG,
        pdf_path=OUTPUT_BRIER_DUMBBELL_PDF,
        color_mapping=color_mapping,
        metric_column="brier",
        metric_title="Within-model Brier comparisons across prompting and language conditions (validated CSV)",
        larger_is_better=False,
    )

    print(f"Input CSV: {INPUT_CSV}")
    print(f"Saved summary CSV: {OUTPUT_SUMMARY_CSV}")
    print(f"Saved scatter figure: {scatter_path}")
    print(f"Saved F1 bar figure: {f1_bar_path}")
    print(f"Saved Brier bar figure: {brier_bar_path}")
    print(f"Saved F1 dumbbell figure: {f1_dumbbell_path}")
    print(f"Saved Brier dumbbell figure: {brier_dumbbell_path}")

    print("Scatter panel counts:")
    for panel_name, row_count in scatter_panel_counts.items():
        print(f"  {panel_name}: {row_count}")

    print("F1 bar panel counts:")
    for panel_name, row_count in f1_bar_panel_counts.items():
        print(f"  {panel_name}: {row_count}")

    print("Brier bar panel counts:")
    for panel_name, row_count in brier_bar_panel_counts.items():
        print(f"  {panel_name}: {row_count}")

    print("F1 dumbbell panel counts:")
    for panel_name, row_count in f1_dumbbell_panel_counts.items():
        print(f"  {panel_name}: {row_count}")

    print("Brier dumbbell panel counts:")
    for panel_name, row_count in brier_dumbbell_panel_counts.items():
        print(f"  {panel_name}: {row_count}")

    print("Best-model summary:")
    print(summary_dataframe.to_string(index=False))


if __name__ == "__main__":
    main()
