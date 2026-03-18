from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from plot_f1_brier import (
    DEFAULT_OUTPUT_DIR,
    build_color_mapping,
    compute_best_models,
    create_dumbbell_figure,
    create_scatter_figure,
    create_single_metric_bar_figure,
)


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = DEFAULT_OUTPUT_DIR
VALIDATED_WORKBOOKS = {
    "zh": PROJECT_ROOT / "artifacts" / "merged_zh_models_validated.xlsx",
    "en": PROJECT_ROOT / "artifacts" / "merged_en_models_validated.xlsx",
}
OUTPUT_METRICS_CSV = ARTIFACTS_DIR / "validated_model_f1_brier_skip_missing.csv"
OUTPUT_SUMMARY_CSV = ARTIFACTS_DIR / "validated_hate_f1_brier_best_models.csv"
OUTPUT_SCATTER_PNG = ARTIFACTS_DIR / "validated_hate_f1_brier_2x2_scatter.png"
OUTPUT_SCATTER_PDF = ARTIFACTS_DIR / "validated_hate_f1_brier_2x2_scatter.pdf"
OUTPUT_F1_BAR_PNG = ARTIFACTS_DIR / "validated_hate_f1_2x2_bars.png"
OUTPUT_F1_BAR_PDF = ARTIFACTS_DIR / "validated_hate_f1_2x2_bars.pdf"
OUTPUT_BRIER_BAR_PNG = ARTIFACTS_DIR / "validated_hate_brier_2x2_bars.png"
OUTPUT_BRIER_BAR_PDF = ARTIFACTS_DIR / "validated_hate_brier_2x2_bars.pdf"
OUTPUT_F1_DUMBBELL_PNG = ARTIFACTS_DIR / "validated_hate_f1_2x2_dumbbell.png"
OUTPUT_F1_DUMBBELL_PDF = ARTIFACTS_DIR / "validated_hate_f1_2x2_dumbbell.pdf"
OUTPUT_BRIER_DUMBBELL_PNG = ARTIFACTS_DIR / "validated_hate_brier_2x2_dumbbell.png"
OUTPUT_BRIER_DUMBBELL_PDF = ARTIFACTS_DIR / "validated_hate_brier_2x2_dumbbell.pdf"


def scores_to_binary(score_series: pd.Series) -> pd.Series:
    """Convert 0-5 scores into binary hate predictions using the project's existing threshold."""
    return (score_series.astype(float) >= 1.0).astype(int)


def scores_to_probability(score_series: pd.Series) -> pd.Series:
    """Convert 0-5 scores into probabilities using the project's existing normalization rule."""
    return score_series.astype(float).clip(lower=0.0, upper=5.0) / 5.0


def compute_metrics_for_language(language: str, workbook_path: Path) -> pd.DataFrame:
    """Compute per-model per-setting F1 and Brier from one validated merged workbook."""
    merged_dataframe = pd.read_excel(workbook_path, sheet_name="merged")
    label_series = merged_dataframe["label/2classes"].astype(int)

    score_columns = sorted(column for column in merged_dataframe.columns if column.endswith("_score"))
    rows: list[dict[str, object]] = []

    for score_column in score_columns:
        model_name, setting_name, _ = score_column.rsplit("_", 2)
        score_series = pd.to_numeric(merged_dataframe[score_column], errors="coerce")
        valid_mask = score_series.notna()
        valid_scores = score_series.loc[valid_mask]
        valid_labels = label_series.loc[valid_mask]

        if valid_scores.empty:
            continue

        binary_predictions = scores_to_binary(valid_scores)
        probability_predictions = scores_to_probability(valid_scores)
        brier_value = float(((probability_predictions - valid_labels.astype(float)) ** 2).mean())

        rows.append(
            {
                "language": language,
                "model": model_name,
                "setting": setting_name,
                "n_scored": int(valid_mask.sum()),
                "f1": float(f1_score(valid_labels, binary_predictions, zero_division=0)),
                "brier": brier_value,
            }
        )

    return pd.DataFrame(rows).sort_values(["language", "model", "setting"]).reset_index(drop=True)


def build_metrics_table() -> pd.DataFrame:
    """Combine the Chinese and English validated workbook metrics into one tidy table."""
    metric_frames = [
        compute_metrics_for_language(language=language, workbook_path=workbook_path)
        for language, workbook_path in VALIDATED_WORKBOOKS.items()
    ]
    metrics_dataframe = pd.concat(metric_frames, ignore_index=True)
    metrics_dataframe = metrics_dataframe.sort_values(["language", "setting", "model"]).reset_index(drop=True)
    return metrics_dataframe


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dataframe = build_metrics_table()
    metrics_dataframe.to_csv(OUTPUT_METRICS_CSV, index=False)

    color_mapping = build_color_mapping(metrics_dataframe["model"].map(lambda value: value).replace({}, regex=False))
    # Rebuild color mapping using the plotting module's display labels.
    metrics_dataframe = metrics_dataframe.copy()
    from plot_f1_brier import MODEL_LABELS

    metrics_dataframe["model_label"] = metrics_dataframe["model"].map(MODEL_LABELS).fillna(metrics_dataframe["model"])
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
        metric_title="F1 across language and prompting settings (validated outputs)",
        direction_label="F1 ↑",
        larger_is_better=True,
    )
    brier_bar_path, brier_bar_panel_counts = create_single_metric_bar_figure(
        dataframe=metrics_dataframe,
        png_path=OUTPUT_BRIER_BAR_PNG,
        pdf_path=OUTPUT_BRIER_BAR_PDF,
        color_mapping=color_mapping,
        metric_column="brier",
        metric_title="Brier score across language and prompting settings (validated outputs)",
        direction_label="Brier score ↓",
        larger_is_better=False,
    )
    f1_dumbbell_path, f1_dumbbell_panel_counts = create_dumbbell_figure(
        dataframe=metrics_dataframe,
        png_path=OUTPUT_F1_DUMBBELL_PNG,
        pdf_path=OUTPUT_F1_DUMBBELL_PDF,
        color_mapping=color_mapping,
        metric_column="f1",
        metric_title="Within-model F1 comparisons across prompting and language conditions (validated outputs)",
        larger_is_better=True,
    )
    brier_dumbbell_path, brier_dumbbell_panel_counts = create_dumbbell_figure(
        dataframe=metrics_dataframe,
        png_path=OUTPUT_BRIER_DUMBBELL_PNG,
        pdf_path=OUTPUT_BRIER_DUMBBELL_PDF,
        color_mapping=color_mapping,
        metric_column="brier",
        metric_title="Within-model Brier comparisons across prompting and language conditions (validated outputs)",
        larger_is_better=False,
    )

    print(f"Saved metrics CSV: {OUTPUT_METRICS_CSV}")
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
