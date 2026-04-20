"""Item-level human-vs-LLM bias-pattern similarity, split by participant gender.

This is a parallel companion to ``human_model_delta_similarity.py`` (the
all-participant baseline). Same metrics, same models, same scales, but the
human-side per-item Delta is computed twice: once using only male participants'
ratings, once using only female participants' ratings. The output answers the
question: which model's bias pattern best matches male raters, and which best
matches female raters, on each of the three response scales.

Outputs:
    artifacts/human_model_delta_similarity_by_participant_gender.csv
    artifacts/human_model_delta_similarity_by_participant_gender.pdf
    artifacts/human_model_delta_similarity_by_participant_gender.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Reuse all metric/data/visual helpers from the baseline 2a script so the
# numbers stay strictly comparable. We do NOT redefine these functions here.
from human_model_delta_similarity import (
    BOOTSTRAP_SEED,
    HUMAN_CSV,
    MODEL_FEMALE_COLS,
    MODEL_MALE_COLS,
    N_BOOTSTRAP,
    SCALE_ORDER,
    bootstrap_ci,
    icc_two_raters,
    lin_ccc,
    load_model_deltas,
    mae,
    quadratic_weighted_kappa_on_signs,
    rmse,
    spearman_rho,
)
from plot_f1_brier import (
    SPINE_COLOR,
    TEXT_COLOR,
    build_color_mapping,
    configure_plot_theme,
)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_CSV = OUTPUT_DIR / "human_model_delta_similarity_by_participant_gender.csv"
OUTPUT_PDF = OUTPUT_DIR / "human_model_delta_similarity_by_participant_gender.pdf"
OUTPUT_PNG = OUTPUT_DIR / "human_model_delta_similarity_by_participant_gender.png"

PARTICIPANT_GENDER_ORDER: list[tuple[str, str]] = [
    ("男", "男性被试"),
    ("女", "女性被试"),
]


def load_human_deltas_by_gender(
    participant_gender: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Per-scale wide-form human Delta (F-M means) for one participant gender.

    Returns a (delta_per_scale, n_participants_per_scale) pair so the caller
    can record sample size next to each metric row.
    """
    human_long = pd.read_csv(HUMAN_CSV, low_memory=False)
    human_long["response_value"] = pd.to_numeric(
        human_long["response_value"], errors="coerce"
    )
    # Filter to participants of the requested gender; this is the only line
    # that differs from the all-participant baseline.
    human_long = human_long.loc[human_long["gender"] == participant_gender].copy()

    deltas_per_scale: dict[str, pd.DataFrame] = {}
    n_per_scale: dict[str, int] = {}
    for scale_key, _ in SCALE_ORDER:
        scale_rows = human_long.loc[human_long["condition"] == scale_key].copy()
        n_per_scale[scale_key] = int(scale_rows["session_id"].nunique())
        version_means = (
            scale_rows.groupby(["item_id", "shown_version"])["response_value"]
            .mean()
            .unstack("shown_version")
        )
        delta = version_means["女人版"] - version_means["男人版"]
        delta.name = "delta_human"
        deltas_per_scale[scale_key] = delta.reset_index()
    return deltas_per_scale, n_per_scale


def compute_metrics_for_one_gender(
    participant_gender: str,
    participant_gender_label: str,
    model_deltas: dict[str, pd.DataFrame],
) -> list[dict]:
    """Run the full metric pipeline for a single participant-gender subgroup."""
    human_deltas, n_human_per_scale = load_human_deltas_by_gender(participant_gender)
    rows: list[dict] = []
    for scale_key, scale_label in SCALE_ORDER:
        human_frame = human_deltas[scale_key]
        for model_label, model_frame in model_deltas.items():
            merged = human_frame.merge(
                model_frame[["item_id", f"delta_{scale_key}"]],
                on="item_id",
                how="inner",
            ).dropna()
            human_vector = merged["delta_human"].to_numpy(dtype=float)
            model_vector = merged[f"delta_{scale_key}"].to_numpy(dtype=float)
            n_items = len(merged)

            row: dict = {
                "participant_gender": participant_gender,
                "participant_gender_label": participant_gender_label,
                "scale": scale_key,
                "scale_label": scale_label,
                "model": model_label,
                "n_items": n_items,
                "n_human_participants": n_human_per_scale[scale_key],
            }

            # Primary: ICC(2,1) absolute agreement.
            icc_21 = icc_two_raters(human_vector, model_vector, "2,1")
            icc_21_low, icc_21_high = bootstrap_ci(
                lambda a, b: icc_two_raters(a, b, "2,1"),
                human_vector, model_vector, N_BOOTSTRAP, BOOTSTRAP_SEED,
            )
            row["icc_2_1"] = icc_21
            row["icc_2_1_ci_low"] = icc_21_low
            row["icc_2_1_ci_high"] = icc_21_high

            # Secondary: ICC(3,1) consistency.
            icc_31 = icc_two_raters(human_vector, model_vector, "3,1")
            icc_31_low, icc_31_high = bootstrap_ci(
                lambda a, b: icc_two_raters(a, b, "3,1"),
                human_vector, model_vector, N_BOOTSTRAP, BOOTSTRAP_SEED + 1,
            )
            row["icc_3_1"] = icc_31
            row["icc_3_1_ci_low"] = icc_31_low
            row["icc_3_1_ci_high"] = icc_31_high

            # Scale-specific robustness checks (kept identical to baseline).
            if scale_key == "attack_3pt":
                kappa = quadratic_weighted_kappa_on_signs(human_vector, model_vector)
                kappa_low, kappa_high = bootstrap_ci(
                    quadratic_weighted_kappa_on_signs,
                    human_vector, model_vector, N_BOOTSTRAP, BOOTSTRAP_SEED + 2,
                )
                row["weighted_kappa"] = kappa
                row["weighted_kappa_ci_low"] = kappa_low
                row["weighted_kappa_ci_high"] = kappa_high

            rho = spearman_rho(human_vector, model_vector)
            rho_low, rho_high = bootstrap_ci(
                spearman_rho, human_vector, model_vector, N_BOOTSTRAP, BOOTSTRAP_SEED + 3,
            )
            row["spearman_rho"] = rho
            row["spearman_rho_ci_low"] = rho_low
            row["spearman_rho_ci_high"] = rho_high

            if scale_key == "attack_slider_0_100":
                ccc = lin_ccc(human_vector, model_vector)
                ccc_low, ccc_high = bootstrap_ci(
                    lin_ccc, human_vector, model_vector, N_BOOTSTRAP, BOOTSTRAP_SEED + 4,
                )
                rmse_value = rmse(human_vector, model_vector)
                rmse_low, rmse_high = bootstrap_ci(
                    rmse, human_vector, model_vector, N_BOOTSTRAP, BOOTSTRAP_SEED + 5,
                )
                mae_value = mae(human_vector, model_vector)
                mae_low, mae_high = bootstrap_ci(
                    mae, human_vector, model_vector, N_BOOTSTRAP, BOOTSTRAP_SEED + 6,
                )
                row["lin_ccc"] = ccc
                row["lin_ccc_ci_low"] = ccc_low
                row["lin_ccc_ci_high"] = ccc_high
                row["rmse"] = rmse_value
                row["rmse_ci_low"] = rmse_low
                row["rmse_ci_high"] = rmse_high
                row["mae"] = mae_value
                row["mae_ci_low"] = mae_low
                row["mae_ci_high"] = mae_high

            rows.append(row)
    return rows


def build_forest_figure(metric_table: pd.DataFrame) -> plt.Figure:
    """2-row x 3-col forest grid: rows = participant gender, cols = scale."""
    configure_plot_theme()
    # Add a CJK fallback to the global font.family list so subplot titles
    # containing Chinese ("男性被试" / "女性被试") render as glyphs, not boxes.
    plt.rcParams["font.family"] = [
        "Noto Sans CJK JP",
        *plt.rcParams.get("font.family", []),
    ]
    color_mapping = build_color_mapping(metric_table["model"].unique())

    # Shared x range across all six panels for direct visual comparison.
    icc_low_min = metric_table[["icc_2_1_ci_low", "icc_3_1_ci_low"]].min().min()
    icc_high_max = metric_table[["icc_2_1_ci_high", "icc_3_1_ci_high"]].max().max()
    x_padding = 0.04 * (icc_high_max - icc_low_min)
    shared_xlim = (icc_low_min - x_padding, icc_high_max + x_padding)

    figure, axes = plt.subplots(2, 3, figsize=(12.4, 8.6), sharex=True)

    for row_index, (gender_key, gender_label) in enumerate(PARTICIPANT_GENDER_ORDER):
        for col_index, (scale_key, scale_label) in enumerate(SCALE_ORDER):
            axis = axes[row_index, col_index]
            panel_frame = metric_table.loc[
                (metric_table["participant_gender"] == gender_key)
                & (metric_table["scale"] == scale_key)
            ].copy()
            panel_frame = panel_frame.sort_values("icc_2_1", ascending=True)
            model_order = panel_frame["model"].tolist()
            y_positions = np.arange(len(model_order))

            # ICC(2,1) primary: filled dots + horizontal CI bars.
            icc21 = panel_frame["icc_2_1"].to_numpy()
            icc21_low = panel_frame["icc_2_1_ci_low"].to_numpy()
            icc21_high = panel_frame["icc_2_1_ci_high"].to_numpy()
            axis.hlines(
                y=y_positions, xmin=icc21_low, xmax=icc21_high,
                color="#22303C", linewidth=1.4, alpha=0.85,
            )
            axis.scatter(
                icc21, y_positions, s=58,
                color=[color_mapping[m] for m in model_order],
                edgecolors="#22303C", linewidths=0.8, zorder=3,
            )

            # ICC(3,1) secondary: hollow diamonds offset slightly.
            icc31 = panel_frame["icc_3_1"].to_numpy()
            icc31_low = panel_frame["icc_3_1_ci_low"].to_numpy()
            icc31_high = panel_frame["icc_3_1_ci_high"].to_numpy()
            axis.hlines(
                y=y_positions - 0.22, xmin=icc31_low, xmax=icc31_high,
                color="#8897A4", linewidth=1.1, alpha=0.85,
            )
            axis.scatter(
                icc31, y_positions - 0.22, s=44,
                facecolors="white", edgecolors="#22303C", linewidths=1.0,
                marker="D", zorder=3,
            )

            n_human = int(panel_frame["n_human_participants"].iloc[0])
            axis.set_yticks(y_positions)
            axis.set_yticklabels(model_order, fontsize=9.0)
            axis.set_title(
                f"{gender_label} / {scale_label}  (n={n_human})",
                fontsize=11, color=TEXT_COLOR,
            )
            axis.axvline(0.0, color=SPINE_COLOR, linewidth=0.8, linestyle="--", zorder=1)
            axis.set_xlim(shared_xlim)
            if row_index == 1:
                axis.set_xlabel(r"ICC (bias-pattern $\Delta_i$)")
            axis.tick_params(axis="both", length=0)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=8,
               markerfacecolor="#2F3E4D", markeredgecolor="#22303C",
               label="ICC(2,1) absolute agreement"),
        Line2D([0], [0], marker="D", linestyle="none", markersize=7,
               markerfacecolor="white", markeredgecolor="#22303C",
               label="ICC(3,1) consistency"),
    ]
    figure.legend(
        handles=legend_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=10,
    )
    figure.suptitle(
        r"Human-vs-model bias-pattern similarity ($\Delta_i = F - M$), "
        "split by participant gender",
        fontsize=13.6, color=TEXT_COLOR, y=0.997,
    )
    figure.tight_layout(rect=(0.01, 0.04, 0.99, 0.965))
    return figure


def main() -> None:
    """Run the full per-gender pipeline and write all artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sanity check: male+female participant counts should sum to 779 across
    # the long file (between-subjects design, each session in one condition).
    raw_long = pd.read_csv(HUMAN_CSV, low_memory=False)
    n_male_total = int(raw_long.loc[raw_long["gender"] == "男", "session_id"].nunique())
    n_female_total = int(raw_long.loc[raw_long["gender"] == "女", "session_id"].nunique())
    print(
        f"[sanity] male participants: {n_male_total}, female participants: "
        f"{n_female_total}, sum: {n_male_total + n_female_total}"
    )

    model_deltas = load_model_deltas()
    all_rows: list[dict] = []
    for gender_key, gender_label in PARTICIPANT_GENDER_ORDER:
        all_rows.extend(
            compute_metrics_for_one_gender(gender_key, gender_label, model_deltas)
        )
    metric_table = pd.DataFrame(all_rows)
    metric_table.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote: {OUTPUT_CSV}")

    # Per-cell top-1 model report.
    print("\n=== Top-1 model per (participant gender × scale) by ICC(2,1) ===")
    for gender_key, gender_label in PARTICIPANT_GENDER_ORDER:
        for scale_key, scale_label in SCALE_ORDER:
            cell = metric_table.loc[
                (metric_table["participant_gender"] == gender_key)
                & (metric_table["scale"] == scale_key)
            ].sort_values("icc_2_1", ascending=False).iloc[0]
            print(
                f"  {gender_label} / {scale_label}: "
                f"{cell['model']}  "
                f"ICC(2,1)={cell['icc_2_1']:.3f} "
                f"[{cell['icc_2_1_ci_low']:.3f}, {cell['icc_2_1_ci_high']:.3f}]  "
                f"ICC(3,1)={cell['icc_3_1']:.3f}  "
                f"n_items={int(cell['n_items'])}  "
                f"n_humans={int(cell['n_human_participants'])}"
            )

    figure = build_forest_figure(metric_table)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"\nWrote: {OUTPUT_PDF}")
    print(f"Wrote: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
