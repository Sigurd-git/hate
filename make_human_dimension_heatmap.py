"""Human-side per-dimension dz heatmap (Task A of the paper-followups bundle).

This is the human counterpart to the existing model-side figure
``artifacts/model_by_level1_by_scale_dz_heatmap.png``. Together with that
figure it supports paper Section R4 ("the gap concentrates on sexualization
and appearance dimensions for both LLMs and humans").

For each rating scale, we compute Cohen's dz of the female-minus-male
attack rating gap within each first-level attack domain, separately for
the three human rater groups (overall, female participants, male
participants). dz is computed across items inside the domain
(unnormalized scale; consistent with the model-side heatmap).

Inputs
------
- ``artifacts/human_model_dz_descriptive_comparison/all_rater_item_deltas.csv``
  Required columns: rater_kind, rater_id, rater_label, condition,
  item_id, dimension_1, delta_female_minus_male.

Outputs
-------
- ``artifacts/paper_followups/human_by_level1_by_scale_dz_heatmap.png``
- ``artifacts/paper_followups/human_by_level1_by_scale_dz_heatmap.pdf``
- ``artifacts/paper_followups/human_by_level1_by_scale_dz.csv``
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "human_model_dz_descriptive_comparison"
    / "all_rater_item_deltas.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "paper_followups"


# Scale presentation order matches the existing model-side figure.
SCALE_ORDER: list[tuple[str, str]] = [
    ("attack_3pt", "3-point"),
    ("attack_7pt_likert", "7-point Likert"),
    ("attack_slider_0_100", "0-100 slider"),
]

# Human rater column order: keep "全部人类" leftmost so the reader's eye
# anchors on the overall result, then move outward to female/male.
HUMAN_RATER_ORDER: list[tuple[str, str]] = [
    ("human_all", "全部人类"),
    ("human_female", "女性被试"),
    ("human_male", "男性被试"),
]

# Same domain ordering as the existing model heatmap so the two figures
# are visually comparable side by side.
LEVEL1_ORDER: list[str] = [
    "性化攻击（性羞辱）",
    "外貌形象攻击",
    "性别角色/性别表达攻击",
    "人际关系攻击",
    "道德品行攻击",
    "经济资源攻击",
    "社会地位攻击",
    "情绪稳定攻击",
    "能力才干攻击",
    "智力理性攻击",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Long-form item-level deltas across all 12 raters and 3 scales.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the figure and CSV.",
    )
    return parser.parse_args()


def configure_matplotlib_for_chinese_text() -> None:
    """Inline the CJK-friendly matplotlib config used by sibling scripts.

    Noto Sans CJK ships as a TrueType Collection. Matplotlib mislabels TTC
    as CFF when writing pdf.fonttype=42, which produces tofu boxes in some
    PDF readers. fonttype=3 (Type 3, paths) avoids the issue cleanly.
    """
    mpl.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 3


def cohens_dz(values: np.ndarray) -> float:
    """Cohen's dz on a paired-difference vector (one entry per item)."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size < 3:
        return float("nan")
    standard_deviation = float(np.std(finite_values, ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation == 0.0:
        return float("nan")
    return float(np.mean(finite_values) / standard_deviation)


def compute_dimension_dz_table(item_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item-level deltas into per-(rater, scale, domain) dz."""
    human_frame = item_frame.loc[item_frame["rater_kind"] == "human"].copy()
    human_frame["delta_female_minus_male"] = pd.to_numeric(
        human_frame["delta_female_minus_male"], errors="coerce"
    )

    rows: list[dict[str, object]] = []
    for (rater_id, condition_name, dimension_label), group in human_frame.groupby(
        ["rater_id", "condition", "dimension_1"], sort=False
    ):
        delta_values = group["delta_female_minus_male"].to_numpy(dtype=float)
        rows.append(
            {
                "rater_id": rater_id,
                "rater_label": group["rater_label"].iloc[0],
                "condition": condition_name,
                "scale_label": dict(SCALE_ORDER).get(condition_name, condition_name),
                "dimension_1": dimension_label,
                "n_items": int(np.sum(np.isfinite(delta_values))),
                "mean_delta": float(np.nanmean(delta_values)),
                "sd_delta": float(np.nanstd(delta_values, ddof=1))
                if np.sum(np.isfinite(delta_values)) >= 2
                else float("nan"),
                "dz": cohens_dz(delta_values),
            }
        )
    return pd.DataFrame(rows)


def build_panel_matrix(
    dimension_table: pd.DataFrame, condition_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dz_matrix, n_matrix) shaped (n_domains, n_human_groups)."""
    panel_frame = dimension_table.loc[dimension_table["condition"] == condition_name]
    dz_matrix = np.full((len(LEVEL1_ORDER), len(HUMAN_RATER_ORDER)), np.nan)
    n_matrix = np.zeros_like(dz_matrix)
    for _, row in panel_frame.iterrows():
        dimension_name = row["dimension_1"]
        rater_id = row["rater_id"]
        if dimension_name not in LEVEL1_ORDER:
            continue
        rater_index = next(
            (
                idx
                for idx, (candidate_id, _) in enumerate(HUMAN_RATER_ORDER)
                if candidate_id == rater_id
            ),
            None,
        )
        if rater_index is None:
            continue
        row_index = LEVEL1_ORDER.index(dimension_name)
        dz_matrix[row_index, rater_index] = float(row["dz"])
        n_matrix[row_index, rater_index] = float(row["n_items"])
    return dz_matrix, n_matrix


def render_heatmap(
    dimension_table: pd.DataFrame, output_png_path: Path, output_pdf_path: Path
) -> None:
    """Draw the 1x3 heatmap (one panel per scale)."""
    configure_matplotlib_for_chinese_text()

    panel_matrices: list[tuple[str, np.ndarray, np.ndarray]] = []
    for condition_name, scale_label in SCALE_ORDER:
        dz_matrix, n_matrix = build_panel_matrix(dimension_table, condition_name)
        panel_matrices.append((scale_label, dz_matrix, n_matrix))

    finite_dz_values = np.concatenate(
        [matrix[np.isfinite(matrix)] for _, matrix, _ in panel_matrices]
    )
    max_abs = float(np.nanmax(np.abs(finite_dz_values))) if finite_dz_values.size else 1.0
    vmax = max(0.4, max_abs * 1.05)
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    color_map = plt.get_cmap("RdBu_r")

    figure, axes = plt.subplots(1, 3, figsize=(12.8, 6.2), sharey=False)
    last_image = None

    for panel_index, (scale_label, dz_matrix, n_matrix) in enumerate(panel_matrices):
        axis = axes[panel_index]
        last_image = axis.imshow(dz_matrix, cmap=color_map, norm=norm, aspect="auto")

        for row_index in range(dz_matrix.shape[0]):
            for col_index in range(dz_matrix.shape[1]):
                value = dz_matrix[row_index, col_index]
                if not np.isfinite(value):
                    continue
                # White text inside dark cells, dark text on faint cells.
                text_color = "white" if abs(value) > vmax * 0.55 else "#222222"
                axis.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9.4,
                    color=text_color,
                )

        axis.set_xticks(np.arange(len(HUMAN_RATER_ORDER)))
        axis.set_xticklabels(
            [label for _, label in HUMAN_RATER_ORDER],
            rotation=20,
            ha="right",
            fontsize=9.6,
        )
        if panel_index == 0:
            axis.set_yticks(np.arange(len(LEVEL1_ORDER)))
            axis.set_yticklabels(LEVEL1_ORDER, fontsize=9.4)
        else:
            axis.set_yticks(np.arange(len(LEVEL1_ORDER)))
            axis.set_yticklabels([])
        axis.set_title(scale_label, fontsize=12.5)
        axis.tick_params(axis="both", length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)

    color_bar = figure.colorbar(
        last_image, ax=axes, fraction=0.022, pad=0.02, shrink=0.85
    )
    color_bar.set_label(r"Cohen's $d_z$ (female $-$ male)", fontsize=10.5)

    figure.suptitle(
        "Human side per-dimension effect sizes: 3 rater groups x 10 attack "
        "domains x 3 rating scales",
        fontsize=12.6,
        y=1.00,
    )

    figure.savefig(output_png_path, dpi=300, bbox_inches="tight")
    figure.savefig(output_pdf_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    item_frame = pd.read_csv(arguments.input_path)
    dimension_table = compute_dimension_dz_table(item_frame)
    dimension_table = dimension_table.sort_values(
        ["condition", "rater_id", "dimension_1"], kind="mergesort"
    ).reset_index(drop=True)

    output_csv_path = arguments.output_dir / "human_by_level1_by_scale_dz.csv"
    dimension_table.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    output_png_path = arguments.output_dir / "human_by_level1_by_scale_dz_heatmap.png"
    output_pdf_path = output_png_path.with_suffix(".pdf")
    render_heatmap(dimension_table, output_png_path, output_pdf_path)

    # Console summary so the caller can verify quickly.
    pivot_3pt = (
        dimension_table.loc[dimension_table["condition"] == "attack_3pt"]
        .pivot_table(index="dimension_1", columns="rater_label", values="dz")
        .reindex(LEVEL1_ORDER)
    )
    print("Task A finished. Top-level summary (3-point dz by domain x rater):")
    print(pivot_3pt.round(3).to_string())
    print(f"\nFigure: {output_png_path}")
    print(f"Table:  {output_csv_path}")


if __name__ == "__main__":
    main()
