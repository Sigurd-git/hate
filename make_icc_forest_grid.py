"""Item-level ICC(2,1) forest grid: 9 LLMs vs. overall human aggregate, per scale.

Creates a 1x3 forest grid figure summarising item-level absolute-agreement
ICC(2,1) between each LLM and the overall-human aggregate's per-item
delta_F-M, separately for each of the three rating scales. Each panel is one
scale; rows within each panel are the nine LLMs ordered by ICC ascending,
with horizontal 95% CIs from the source CSV.

Input
-----
- ``artifacts/human_model_delta_similarity.csv``
  Required columns: scale, scale_label, model, icc_2_1, icc_2_1_ci_low,
  icc_2_1_ci_high.

Outputs
-------
- ``artifacts/paper_followups/human_model_icc_forest_grid.{pdf,png}``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "artifacts" / "human_model_delta_similarity.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "paper_followups"

SCALES: list[tuple[str, str]] = [
    ("attack_3pt", "3-point"),
    ("attack_7pt_likert", "7-point Likert"),
    ("attack_slider_0_100", "0-100 slider"),
]

POOR_THRESHOLD = 0.50  # conventional cutoff between poor and moderate
MODERATE_THRESHOLD = 0.75  # conventional cutoff between moderate and good
POINT_COLOR = "#2D5C7F"
LINE_COLOR = "#5B7FA0"
ZERO_LINE_COLOR = "#9AA8B7"
POOR_BAND_COLOR = "#F4E2E2"
SPINE_COLOR = "#CCD4DD"
TEXT_COLOR = "#22303C"
SUBTLE_TEXT_COLOR = "#607080"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--input-csv", type=Path, default=DEFAULT_INPUT,
        help="Source per-(model, scale) similarity CSV.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the forest grid figure.",
    )
    return parser.parse_args()


def configure_matplotlib_for_chinese_text() -> None:
    """Inline CJK-friendly matplotlib defaults consistent with sibling scripts."""
    mpl.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 3


def render_icc_forest_grid(
    similarity_frame: pd.DataFrame, output_png: Path, output_pdf: Path
) -> None:
    configure_matplotlib_for_chinese_text()

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), sharey=False)

    x_min = float(np.nanmin(similarity_frame["icc_2_1_ci_low"]))
    x_max = float(np.nanmax(similarity_frame["icc_2_1_ci_high"]))
    x_pad = max(0.04, 0.08 * (x_max - x_min))
    x_lim = (x_min - x_pad, x_max + x_pad)

    for panel_index, (scale_key, scale_label) in enumerate(SCALES):
        axis = axes[panel_index]
        panel_frame = (
            similarity_frame.loc[similarity_frame["scale"] == scale_key]
            .copy()
            .sort_values("icc_2_1", ascending=True)
            .reset_index(drop=True)
        )

        # Shade the "poor" region (ICC < 0.50, conventional Cicchetti cutoff)
        # so the reader sees that all model-scale cells fall inside it.
        axis.axvspan(
            x_lim[0],
            min(POOR_THRESHOLD, x_lim[1]),
            color=POOR_BAND_COLOR,
            alpha=0.55,
            zorder=0,
        )

        y_positions = np.arange(len(panel_frame))
        for row_index, data_row in panel_frame.iterrows():
            axis.hlines(
                y=row_index,
                xmin=data_row["icc_2_1_ci_low"],
                xmax=data_row["icc_2_1_ci_high"],
                color=LINE_COLOR,
                linewidth=1.6,
                alpha=0.95,
            )
            axis.scatter(
                [data_row["icc_2_1"]],
                [row_index],
                s=46,
                color=POINT_COLOR,
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )
            axis.text(
                data_row["icc_2_1_ci_high"] + 0.012,
                row_index,
                f"{data_row['icc_2_1']:.2f}",
                va="center",
                ha="left",
                fontsize=8.6,
                color=SUBTLE_TEXT_COLOR,
            )

        axis.axvline(0.0, color=ZERO_LINE_COLOR, linewidth=0.9, linestyle="--")
        axis.set_yticks(y_positions)
        axis.set_yticklabels(panel_frame["model"], fontsize=9.6)
        axis.set_xlim(*x_lim)
        axis.set_title(scale_label, fontsize=12.0, color=TEXT_COLOR, pad=4.0)
        axis.set_xlabel("ICC(2,1) vs. overall human", fontsize=10.4, color=TEXT_COLOR)
        axis.tick_params(axis="x", labelsize=9.0, colors=SUBTLE_TEXT_COLOR)
        axis.tick_params(axis="y", labelsize=9.6, colors=TEXT_COLOR)
        for spine in axis.spines.values():
            spine.set_color(SPINE_COLOR)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="x", color="#EEF1F5", linewidth=0.7)
        axis.set_axisbelow(True)

    figure.suptitle(
        "Item-level ICC(2,1) of $\\Delta_{F-M}$: each LLM vs. the overall "
        "human aggregate, with 95% CI",
        fontsize=12.6,
        color=TEXT_COLOR,
        y=1.02,
    )
    figure.text(
        0.01,
        -0.02,
        "Pink band: ICC < 0.50 (conventionally 'poor' agreement). "
        "All 27 model-scale cells fall inside it.",
        fontsize=9.0,
        color=SUBTLE_TEXT_COLOR,
    )

    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    similarity_frame = pd.read_csv(arguments.input_csv)

    output_png = arguments.output_dir / "human_model_icc_forest_grid.png"
    output_pdf = output_png.with_suffix(".pdf")
    render_icc_forest_grid(similarity_frame, output_png, output_pdf)
    print(f"  wrote {output_pdf.name}")


if __name__ == "__main__":
    main()
