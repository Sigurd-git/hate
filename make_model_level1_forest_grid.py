"""9-model x 10-domain x 3-scale level-1 forest grid for the paper.

Recreates the figure used in earlier presentation slides ("图4 一级维度效应及其
95% 置信区间，按绝对效应反向排序"). For each (model, scale) cell we plot a
10-row forest of mean female-minus-male deltas with their 95% CIs, ordered
by absolute Cohen's d_z descending. Significant cells (Wilcoxon FDR q<.05)
are coloured; non-significant cells are dimmed.

Inputs
------
- ``outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b/figures/
   {attack_3pt,attack_7pt_likert,attack_slider_0_100}/stats_level1.csv``
  Required columns: model_label, 一级-攻击领域, n, mean_diff_female_minus_male,
  ci_low, ci_high, cohens_dz, wilcoxon_p_fdr.

Outputs
-------
- ``artifacts/paper_followups/model_level1_forest_grid_3pt.{pdf,png}``
- ``artifacts/paper_followups/model_level1_forest_grid_7pt.{pdf,png}``
- ``artifacts/paper_followups/model_level1_forest_grid_slider.{pdf,png}``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
STATS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "group_swap_1b"
    / "1b_groupswap_demensionsentence"
    / "analysis_1b"
    / "figures"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "paper_followups"

# Match the 3x3 ordering preferred by sibling figures so the reader can
# scan the panels in a stable layout.
MODEL_GRID_ORDER: list[str] = [
    "GPT-5.1",
    "Claude Opus 4.5",
    "GLM 4.6",
    "Llama Maverick",
    "DeepSeek R1",
    "DeepSeek V3.2",
    "Kimi K2",
    "Qwen 2.5 72B",
    "Gemma 4 31B",
]

SCALES: list[tuple[str, str, str]] = [
    ("attack_3pt", "3pt", "3-point"),
    ("attack_7pt_likert", "7pt", "7-point Likert"),
    ("attack_slider_0_100", "slider", "0-100 slider"),
]

SIG_COLOR = "#B23A48"
NONSIG_COLOR = "#9AA8B7"
ZERO_LINE_COLOR = "#7F8B97"
SPINE_COLOR = "#CCD4DD"
TEXT_COLOR = "#22303C"
SUBTLE_TEXT_COLOR = "#607080"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the forest grid figures.",
    )
    return parser.parse_args()


def configure_matplotlib_for_chinese_text() -> None:
    """Inline the CJK-friendly matplotlib config used by sibling scripts."""
    mpl.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 3


def load_scale_frame(scale_key: str) -> pd.DataFrame:
    """Load and lightly clean the per-domain stats CSV for a scale."""
    csv_path = STATS_DIR / scale_key / "stats_level1.csv"
    frame = pd.read_csv(csv_path)
    needed_columns = [
        "model_label",
        "一级-攻击领域",
        "n",
        "mean_diff_female_minus_male",
        "ci_low",
        "ci_high",
        "cohens_dz",
        "wilcoxon_p_fdr",
    ]
    return frame[needed_columns].copy()


def render_single_scale_grid(
    scale_frame: pd.DataFrame,
    scale_label: str,
    output_png: Path,
    output_pdf: Path,
) -> None:
    """Render one 3x3 grid of per-model forests for a given rating scale."""
    configure_matplotlib_for_chinese_text()

    # Shared x-axis range so panels are visually comparable.
    x_min = float(np.nanmin(scale_frame["ci_low"]))
    x_max = float(np.nanmax(scale_frame["ci_high"]))
    x_pad = max(0.05, 0.06 * (x_max - x_min))
    x_lim = (x_min - x_pad, x_max + x_pad)

    figure, axes_grid = plt.subplots(3, 3, figsize=(13.0, 12.5), sharex=True)
    panel_letters = list("ABCDEFGHI")

    for panel_index, model_name in enumerate(MODEL_GRID_ORDER):
        row_index, col_index = divmod(panel_index, 3)
        axis = axes_grid[row_index, col_index]

        model_frame = scale_frame.loc[
            scale_frame["model_label"] == model_name
        ].copy()
        # Order rows by absolute dz descending so the strongest domain
        # sits on top (matches the legacy presentation figure).
        model_frame["abs_dz"] = model_frame["cohens_dz"].abs()
        model_frame = model_frame.sort_values(
            "abs_dz", ascending=True
        ).reset_index(drop=True)

        y_positions = np.arange(len(model_frame))
        for row_position, data_row in model_frame.iterrows():
            is_significant = (
                pd.notna(data_row["wilcoxon_p_fdr"])
                and data_row["wilcoxon_p_fdr"] < 0.05
            )
            point_color = SIG_COLOR if is_significant else NONSIG_COLOR
            point_alpha = 1.0 if is_significant else 0.55

            axis.hlines(
                y=row_position,
                xmin=data_row["ci_low"],
                xmax=data_row["ci_high"],
                color=point_color,
                linewidth=1.6,
                alpha=point_alpha,
            )
            axis.scatter(
                [data_row["mean_diff_female_minus_male"]],
                [row_position],
                s=44,
                color=point_color,
                edgecolors="white",
                linewidths=0.7,
                alpha=point_alpha,
                zorder=3,
            )

        axis.axvline(0.0, color=ZERO_LINE_COLOR, linewidth=0.9, linestyle="--")
        axis.set_yticks(y_positions)
        axis.set_yticklabels(
            [
                f"{name} ({int(n_value)})"
                for name, n_value in zip(
                    model_frame["一级-攻击领域"], model_frame["n"]
                )
            ],
            fontsize=8.6,
        )
        axis.set_xlim(*x_lim)
        axis.set_title(
            f"{panel_letters[panel_index]}. {model_name} · {scale_label}",
            fontsize=11.2,
            color=TEXT_COLOR,
            loc="left",
            pad=4.0,
        )
        axis.tick_params(axis="x", labelsize=9.0, colors=SUBTLE_TEXT_COLOR)
        axis.tick_params(axis="y", labelsize=8.4, colors=TEXT_COLOR)
        for spine in axis.spines.values():
            spine.set_color(SPINE_COLOR)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="x", color="#EEF1F5", linewidth=0.7)
        axis.set_axisbelow(True)

    # Shared x-axis label on the bottom row.
    for col_index in range(3):
        axes_grid[2, col_index].set_xlabel(
            "Mean Δ (female − male)", fontsize=10.4, color=TEXT_COLOR
        )

    figure.suptitle(
        f"Per-domain mean Δ$_{{F-M}}$ with 95% CI · {scale_label} · "
        f"sorted by |$d_z$| descending",
        fontsize=13.4,
        color=TEXT_COLOR,
        y=0.995,
    )

    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker="o", color=SIG_COLOR, linewidth=1.6,
            markersize=7, label="Wilcoxon FDR q < .05",
        ),
        plt.Line2D(
            [0], [0],
            marker="o", color=NONSIG_COLOR, linewidth=1.6,
            markersize=7, alpha=0.55, label="n.s. (q ≥ .05)",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        fontsize=9.6,
    )

    figure.tight_layout(rect=[0.0, 0.02, 1.0, 0.98])
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    for scale_key, scale_short_tag, scale_label in SCALES:
        scale_frame = load_scale_frame(scale_key)
        output_png = (
            arguments.output_dir
            / f"model_level1_forest_grid_{scale_short_tag}.png"
        )
        output_pdf = output_png.with_suffix(".pdf")
        render_single_scale_grid(scale_frame, scale_label, output_png, output_pdf)
        print(f"  wrote {output_pdf.name}")


if __name__ == "__main__":
    main()
