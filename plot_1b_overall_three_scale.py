"""Three-scale overall effect-size forest (F1) and directionality grid (F2).

Inputs:
    outputs/.../figures/{attack_3pt,attack_7pt_likert,attack_slider_0_100}/
        stats_overall.csv, stats_directionality.csv

Outputs (under artifacts/):
    overall_dz_three_scale_forest.pdf/.png     (F1)
    directionality_three_scale_grid.pdf/.png   (F2)

F1: one-row x three-panel forest plot; each panel is one rating scale; nine
models plotted as horizontal 95% CI lines on Cohen's d_z with a coloured dot
at the point estimate; panels share the same x-axis range so the reader can
visually compare the spread of d_z across scales. CI for d_z is derived from
the mean-diff 95% CI via the identity d_z = mean_diff / sd_paired (see
approximate_dz_ci()).

F2: one-row x three-panel stacked horizontal bars; each panel is one rating
scale; each bar (one per model) shows the proportion of 371 pairs where the
female version scored higher, tied, or male-higher. Illustrates how the
tie-rate contracts as scale resolution grows (3pt -> 7pt -> slider).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_f1_brier import (
    SPINE_COLOR,
    SUBTLE_TEXT_COLOR,
    TEXT_COLOR,
    build_color_mapping,
    configure_plot_theme,
)


PROJECT_ROOT = Path(__file__).resolve().parent
STATS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "group_swap_1b"
    / "1b_groupswap_demensionsentence"
    / "analysis_1b"
    / "figures"
)
ARTIFACTS = PROJECT_ROOT / "artifacts"

SCALE_ORDER: list[tuple[str, str]] = [
    ("attack_3pt", "3-point"),
    ("attack_7pt_likert", "7-point Likert"),
    ("attack_slider_0_100", "0\u2013100 slider"),
]

# Colour-blind-safe diverging triad for the directionality stacked bars.
FEMALE_HIGHER_COLOR = "#C8524D"
TIES_COLOR = "#B7BDC4"
MALE_HIGHER_COLOR = "#3E6FAA"


def load_overall_per_scale() -> dict[str, pd.DataFrame]:
    """Read the three stats_overall.csv files keyed by scale identifier."""
    tables: dict[str, pd.DataFrame] = {}
    for scale_key, _ in SCALE_ORDER:
        csv_path = STATS_DIR / scale_key / "stats_overall.csv"
        tables[scale_key] = pd.read_csv(csv_path)
    return tables


def load_directionality_per_scale() -> dict[str, pd.DataFrame]:
    """Read the three stats_directionality.csv files keyed by scale."""
    tables: dict[str, pd.DataFrame] = {}
    for scale_key, _ in SCALE_ORDER:
        csv_path = STATS_DIR / scale_key / "stats_directionality.csv"
        tables[scale_key] = pd.read_csv(csv_path)
    return tables


def approximate_dz_ci(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the mean-diff 95% CI into an approximate d_z 95% CI.

    The stats_overall.csv file contains ``ci_low`` / ``ci_high`` for the raw
    mean-diff (female minus male). Cohen's d_z is defined as
    ``d_z = mean_diff / sd_paired``, so the implicit paired-sample SD is
    ``sd_paired = mean_diff / cohens_dz`` (per model). Scaling the mean-diff
    CI by ``cohens_dz / mean_diff`` therefore yields a first-order CI on
    Cohen's d_z. Guarded with a 1e-6 floor: if either component is too close
    to zero the CI collapses to the point estimate and a warning is printed.
    """
    frame = frame.copy()
    dz = frame["cohens_dz"].astype(float)
    md = frame["mean_diff_female_minus_male"].astype(float)
    safe_mask = (dz.abs() > 1e-6) & (md.abs() > 1e-6)
    unsafe = frame.loc[~safe_mask]
    if len(unsafe):
        for _, row in unsafe.iterrows():
            print(
                f"[warn] approximate_dz_ci: degenerate dz or mean_diff for "
                f"{row.get('model_label', '?')} in {row.get('setting', '?')}"
            )
    scale_factor = np.where(safe_mask, dz / md, 0.0)
    frame["dz_ci_low"] = np.where(
        safe_mask, frame["ci_low"].astype(float) * scale_factor, dz
    )
    frame["dz_ci_high"] = np.where(
        safe_mask, frame["ci_high"].astype(float) * scale_factor, dz
    )
    # Ensure low <= high after the scaling (flip if sign of md flips).
    lo = np.minimum(frame["dz_ci_low"], frame["dz_ci_high"])
    hi = np.maximum(frame["dz_ci_low"], frame["dz_ci_high"])
    frame["dz_ci_low"] = lo
    frame["dz_ci_high"] = hi
    return frame


def build_f1_figure(overall_tables: dict[str, pd.DataFrame]) -> plt.Figure:
    """Three-panel horizontal forest plot of Cohen's d_z per model per scale."""
    configure_plot_theme()
    plt.rcParams["font.family"] = [
        "Noto Sans CJK JP",
        *plt.rcParams.get("font.family", []),
    ]

    dz_frames = {key: approximate_dz_ci(df) for key, df in overall_tables.items()}

    # Shared x range across panels so bar lengths are directly comparable.
    x_lo = min(float(f["dz_ci_low"].min()) for f in dz_frames.values())
    x_hi = max(float(f["dz_ci_high"].max()) for f in dz_frames.values())
    x_pad = 0.05 * (x_hi - x_lo)
    x_lim = (x_lo - x_pad, x_hi + x_pad)

    all_models = set()
    for df in dz_frames.values():
        all_models.update(df["model_label"].astype(str).unique())
    color_map = build_color_mapping(all_models)

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 5.6), sharex=True)
    for panel_idx, (scale_key, scale_label) in enumerate(SCALE_ORDER):
        frame = dz_frames[scale_key].sort_values(
            "cohens_dz", ascending=True
        ).reset_index(drop=True)
        y_positions = np.arange(len(frame))
        axis = axes[panel_idx]
        axis.hlines(
            y=y_positions,
            xmin=frame["dz_ci_low"],
            xmax=frame["dz_ci_high"],
            color="#22303C",
            linewidth=1.4,
            alpha=0.85,
        )
        axis.scatter(
            frame["cohens_dz"],
            y_positions,
            s=70,
            color=[color_map[m] for m in frame["model_label"]],
            edgecolors="#22303C",
            linewidths=0.8,
            zorder=3,
        )
        axis.set_yticks(y_positions)
        axis.set_yticklabels(frame["model_label"], fontsize=9.4)
        axis.axvline(0.0, color=SPINE_COLOR, linewidth=0.8, linestyle="--")
        axis.set_xlim(x_lim)
        axis.set_title(scale_label, fontsize=12, color=TEXT_COLOR)
        axis.set_xlabel(r"Cohen's $d_z$ (female $-$ male)")
        axis.tick_params(axis="both", length=0)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    figure.suptitle(
        "Study 1b overall effect sizes across three rating scales "
        "(9 models, 371 paired sentences)",
        fontsize=13.2,
        color=TEXT_COLOR,
        y=0.995,
    )
    figure.text(
        0.5,
        0.005,
        "Each dot = one LLM; horizontal line = approximate 95% CI on d_z. "
        "Panels share the same x-axis. All 27 cells are BH-FDR q < .001.",
        ha="center",
        fontsize=9,
        color=SUBTLE_TEXT_COLOR,
    )
    figure.tight_layout(rect=(0.01, 0.04, 0.99, 0.96))
    return figure


def build_f2_figure(direction_tables: dict[str, pd.DataFrame]) -> plt.Figure:
    """Three-panel stacked horizontal bars for directionality by scale."""
    configure_plot_theme()
    plt.rcParams["font.family"] = [
        "Noto Sans CJK JP",
        *plt.rcParams.get("font.family", []),
    ]

    model_order = sorted(
        direction_tables[SCALE_ORDER[0][0]]["model_label"].astype(str).unique()
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 5.8), sharey=True)
    for panel_idx, (scale_key, scale_label) in enumerate(SCALE_ORDER):
        axis = axes[panel_idx]
        frame = (
            direction_tables[scale_key]
            .set_index("model_label")
            .reindex(model_order)
            .reset_index()
        )
        y_positions = np.arange(len(frame))
        female_pct = frame["female_higher_pct"].to_numpy() * 100.0
        tie_pct = frame["ties_pct"].to_numpy() * 100.0
        male_pct = frame["male_higher_pct"].to_numpy() * 100.0

        axis.barh(
            y_positions,
            female_pct,
            color=FEMALE_HIGHER_COLOR,
            edgecolor="white",
            linewidth=0.6,
        )
        axis.barh(
            y_positions,
            tie_pct,
            left=female_pct,
            color=TIES_COLOR,
            edgecolor="white",
            linewidth=0.6,
        )
        axis.barh(
            y_positions,
            male_pct,
            left=female_pct + tie_pct,
            color=MALE_HIGHER_COLOR,
            edgecolor="white",
            linewidth=0.6,
        )

        # Label each segment inside the bar if it is wide enough to read.
        for row_idx in range(len(frame)):
            segments = [
                (0.0, female_pct[row_idx]),
                (female_pct[row_idx], tie_pct[row_idx]),
                (female_pct[row_idx] + tie_pct[row_idx], male_pct[row_idx]),
            ]
            for left, width in segments:
                if width < 6.0:
                    continue
                text_color = "white" if width >= 10.0 else TEXT_COLOR
                axis.text(
                    left + width / 2.0,
                    y_positions[row_idx],
                    f"{width:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8.4,
                    color=text_color,
                    fontweight="semibold",
                )

        axis.set_yticks(y_positions)
        axis.set_yticklabels(model_order, fontsize=9.4)
        axis.set_xlim(0, 100)
        axis.set_xlabel("Proportion of 371 paired sentences (%)")
        axis.set_title(scale_label, fontsize=12, color=TEXT_COLOR)
        axis.tick_params(axis="both", length=0)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=FEMALE_HIGHER_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=TIES_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=MALE_HIGHER_COLOR),
    ]
    figure.legend(
        legend_handles,
        ["Female version higher", "Tie", "Male version higher"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    figure.suptitle(
        "Study 1b directionality across three rating scales "
        "(% of 371 paired sentences)",
        fontsize=13.2,
        color=TEXT_COLOR,
        y=0.995,
    )
    figure.tight_layout(rect=(0.01, 0.06, 0.99, 0.96))
    return figure


def _print_summary(
    overall: dict[str, pd.DataFrame], direction: dict[str, pd.DataFrame]
) -> None:
    """Print concise diagnostics: d_z pivot, max FDR q, tie rate range."""
    rows: list[dict] = []
    for scale_key, scale_label in SCALE_ORDER:
        for _, row in overall[scale_key].iterrows():
            rows.append(
                {
                    "model": row["model_label"],
                    "scale": scale_label,
                    "dz": row["cohens_dz"],
                }
            )
    long_frame = pd.DataFrame(rows)
    pivot = long_frame.pivot(index="model", columns="scale", values="dz")[
        [s for _, s in SCALE_ORDER]
    ]
    pivot.loc["__median__"] = pivot.median()
    print("=== Cohen's d_z pivot (models x scales) ===")
    print(pivot.round(3).to_string())
    print()
    for scale_key, scale_label in SCALE_ORDER:
        q_max = float(overall[scale_key]["wilcoxon_p_fdr"].max())
        tie_lo = float(direction[scale_key]["ties_pct"].min()) * 100.0
        tie_hi = float(direction[scale_key]["ties_pct"].max()) * 100.0
        print(
            f"  {scale_label}: max FDR q = {q_max:.2e} | tie-rate "
            f"range = {tie_lo:.1f}%–{tie_hi:.1f}%"
        )


def main() -> None:
    """Compute, plot, and save the three-scale overall figures."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    overall = load_overall_per_scale()
    direction = load_directionality_per_scale()

    _print_summary(overall, direction)

    f1_fig = build_f1_figure(overall)
    f1_pdf = ARTIFACTS / "overall_dz_three_scale_forest.pdf"
    f1_png = ARTIFACTS / "overall_dz_three_scale_forest.png"
    f1_fig.savefig(f1_pdf, bbox_inches="tight")
    f1_fig.savefig(f1_png, dpi=220, bbox_inches="tight")
    plt.close(f1_fig)
    print(f"\nWrote: {f1_pdf}")
    print(f"Wrote: {f1_png}")

    f2_fig = build_f2_figure(direction)
    f2_pdf = ARTIFACTS / "directionality_three_scale_grid.pdf"
    f2_png = ARTIFACTS / "directionality_three_scale_grid.png"
    f2_fig.savefig(f2_pdf, bbox_inches="tight")
    f2_fig.savefig(f2_png, dpi=220, bbox_inches="tight")
    plt.close(f2_fig)
    print(f"Wrote: {f2_pdf}")
    print(f"Wrote: {f2_png}")


if __name__ == "__main__":
    main()
