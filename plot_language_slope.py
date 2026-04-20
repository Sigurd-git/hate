"""Language-slope figure for Study 1a.

Shows each of the nine LLMs as a point in the Chinese panel connected by a
slope line to the same model in the English panel, for F1 and Brier, faceted
by prompt setting (zero-shot, CoT). Overlays panel-level mean +/- SD whiskers
so that the variance-inflation pattern on the Chinese side is immediately
visible alongside the mean shift.

Run:
    uv run python plot_language_slope.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from plot_f1_brier import (
    BACKGROUND_COLOR,
    BASE_FONT_FAMILY,
    CONNECTOR_COLOR,
    EXCLUDED_MODEL_NAMES,
    GRID_COLOR,
    MODEL_LABELS,
    MUTED_MODEL_COLORS,
    SPINE_COLOR,
    SUBTLE_TEXT_COLOR,
    TEXT_COLOR,
    build_color_mapping,
    configure_plot_theme,
)

# Paths -----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_ROOT / "artifacts" / "validated_model_f1_brier_skip_missing.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_PDF = OUTPUT_DIR / "validated_csv_hate_f1_brier_language_slope.pdf"
OUTPUT_PNG = OUTPUT_DIR / "validated_csv_hate_f1_brier_language_slope.png"

# Panel layout ---------------------------------------------------------------
# Two columns = two metrics (F1, Brier). Two rows = two prompt settings.
METRIC_SPECS = [
    {
        "column": "f1",
        "title": "F1",
        "direction_label": "F1 (higher is better)",
        "larger_is_better": True,
    },
    {
        "column": "brier",
        "title": "Brier",
        "direction_label": "Brier (lower is better)",
        "larger_is_better": False,
    },
]

SETTING_SPECS = [
    {"column_value": "zeroshot", "title": "Zero-shot"},
    {"column_value": "cot", "title": "CoT"},
]

LANGUAGE_ORDER = ["zh", "en"]
LANGUAGE_LABELS = {"zh": "Chinese", "en": "English"}

# Visual constants -----------------------------------------------------------
MEAN_MARKER_COLOR = "#2F3E4D"
MEAN_WHISKER_COLOR = "#2F3E4D"
MODEL_MARKER_SIZE = 7.6
MODEL_LINE_ALPHA = 0.75
MEAN_MARKER_SIZE = 10.0


def load_table() -> pd.DataFrame:
    """Load and normalise the validated metrics CSV."""
    dataframe = pd.read_csv(INPUT_CSV)
    dataframe = dataframe.loc[
        ~dataframe["model"].astype(str).isin(EXCLUDED_MODEL_NAMES)
    ].copy()
    dataframe["model_label"] = (
        dataframe["model"].map(MODEL_LABELS).fillna(dataframe["model"])
    )
    dataframe["language"] = dataframe["language"].str.lower()
    dataframe["setting"] = dataframe["setting"].str.lower()
    return dataframe


def draw_panel(
    axis: plt.Axes,
    subset: pd.DataFrame,
    metric_column: str,
    color_mapping: dict[str, str],
) -> None:
    """Draw one Chinese -> English slope panel for a given metric."""
    x_positions = {"zh": 0.0, "en": 1.0}

    # One slope line per model connecting its zh and en scores.
    for model_label, model_sub in subset.groupby("model_label"):
        if set(model_sub["language"]) != set(LANGUAGE_ORDER):
            continue
        paired = (
            model_sub.set_index("language")[metric_column].reindex(LANGUAGE_ORDER).to_numpy()
        )
        xs = np.array([x_positions["zh"], x_positions["en"]])
        color = color_mapping.get(model_label, CONNECTOR_COLOR)
        axis.plot(
            xs,
            paired,
            color=color,
            linewidth=1.35,
            alpha=MODEL_LINE_ALPHA,
            zorder=2,
        )
        axis.plot(
            xs,
            paired,
            linestyle="none",
            marker="o",
            markersize=MODEL_MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )

    # Overlay panel mean +/- SD whiskers on each language.
    for language in LANGUAGE_ORDER:
        language_values = subset.loc[subset["language"] == language, metric_column].to_numpy()
        mean_value = float(np.mean(language_values))
        std_value = float(np.std(language_values, ddof=1))
        axis.errorbar(
            x_positions[language],
            mean_value,
            yerr=std_value,
            fmt="D",
            color=MEAN_WHISKER_COLOR,
            ecolor=MEAN_WHISKER_COLOR,
            elinewidth=1.4,
            capsize=5.0,
            markerfacecolor=MEAN_MARKER_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=MEAN_MARKER_SIZE,
            zorder=4,
        )

    axis.set_xticks(list(x_positions.values()))
    axis.set_xticklabels([LANGUAGE_LABELS[lang] for lang in LANGUAGE_ORDER])
    axis.set_xlim(-0.35, 1.35)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.tick_params(axis="both", length=0)


def build_figure(dataframe: pd.DataFrame, color_mapping: dict[str, str]) -> plt.Figure:
    """Assemble the 2x2 slope figure (rows = settings, columns = metrics)."""
    number_of_rows = len(SETTING_SPECS)
    number_of_columns = len(METRIC_SPECS)
    figure, axes_grid = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(9.4, 7.2),
        sharex=False,
    )
    axes_grid = np.atleast_2d(axes_grid)

    for row_index, setting_spec in enumerate(SETTING_SPECS):
        for column_index, metric_spec in enumerate(METRIC_SPECS):
            axis = axes_grid[row_index][column_index]
            panel_subset = dataframe.loc[
                dataframe["setting"] == setting_spec["column_value"]
            ]
            draw_panel(axis, panel_subset, metric_spec["column"], color_mapping)
            axis.set_title(
                f"{setting_spec['title']} · {metric_spec['title']}",
                fontsize=12.0,
                color=TEXT_COLOR,
            )
            axis.set_ylabel(metric_spec["direction_label"])

    # Shared legend describing per-model line color and the mean marker.
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color_mapping[model_label],
            markeredgecolor="white",
            markeredgewidth=0.75,
            markersize=7.2,
            label=model_label,
        )
        for model_label in sorted(color_mapping)
    ]
    mean_handle = Line2D(
        [0],
        [0],
        marker="D",
        color="none",
        markerfacecolor=MEAN_MARKER_COLOR,
        markeredgecolor="white",
        markersize=8.2,
        label="Panel mean ± SD",
    )
    figure.legend(
        handles=[*model_handles, mean_handle],
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9.0,
    )

    figure.suptitle(
        "Cross-language slopes — per-model F1/Brier with panel mean ± SD",
        fontsize=13.5,
        color=TEXT_COLOR,
        y=0.995,
    )
    figure.tight_layout(rect=(0.02, 0.06, 0.98, 0.96))
    return figure


def main() -> None:
    """Entry point: build the figure and write both PDF and PNG outputs."""
    configure_plot_theme()
    dataframe = load_table()
    color_mapping = build_color_mapping(dataframe["model_label"].unique())
    figure = build_figure(dataframe, color_mapping)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote: {OUTPUT_PDF}")
    print(f"Wrote: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
