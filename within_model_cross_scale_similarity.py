"""Within-model cross-scale similarity of per-item bias patterns.

For each LLM, the 1b paired-prompt experiment yields a per-item bias score
Delta_i = score(female) - score(male) on each of three rating scales:
3-point ordinal, 7-point Likert, and 0-100 slider. Across 371 items, this
produces three 371-vectors per model. The question this script answers is:
within a single model, how similar are these three bias vectors? If the
biases are "the same idea expressed at different resolutions" we should see
high cross-scale rank correlation; if scale-format induces qualitatively
different bias patterns, the correlations will be low or inconsistent.

Metric: Spearman rho (rank-based, scale-invariant). Kendall tau is reported
as a robustness check. 95% bootstrap CIs (n_boot=2000) by item resampling.

Outputs:
    artifacts/within_model_cross_scale_similarity.csv
    artifacts/within_model_cross_scale_similarity.pdf
    artifacts/within_model_cross_scale_similarity.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from human_model_delta_similarity import (
    BOOTSTRAP_SEED,
    MODEL_FEMALE_COLS,
    MODEL_MALE_COLS,
    N_BOOTSTRAP,
    SCALE_ORDER,
    bootstrap_ci,
    load_model_deltas,
)
from plot_f1_brier import (
    SPINE_COLOR,
    TEXT_COLOR,
    build_color_mapping,
    configure_plot_theme,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_CSV = OUTPUT_DIR / "within_model_cross_scale_similarity.csv"
OUTPUT_PDF = OUTPUT_DIR / "within_model_cross_scale_similarity.pdf"
OUTPUT_PNG = OUTPUT_DIR / "within_model_cross_scale_similarity.png"

# Three unordered pairs of scales to correlate within each model.
SCALE_PAIRS: list[tuple[str, str, str]] = [
    ("attack_3pt", "attack_7pt_likert", "3pt vs 7pt"),
    ("attack_7pt_likert", "attack_slider_0_100", "7pt vs slider"),
    ("attack_3pt", "attack_slider_0_100", "3pt vs slider"),
]


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; pandas wrapper handles ties cleanly."""
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau-b correlation as a robustness check."""
    if len(a) < 2:
        return float("nan")
    return float(stats.kendalltau(a, b).correlation)


def compute_within_model_table() -> pd.DataFrame:
    """Build a long table: (model, scale_pair) -> Spearman + Kendall + CIs."""
    model_deltas = load_model_deltas()
    rows: list[dict] = []
    for model_label, model_frame in model_deltas.items():
        for scale_a, scale_b, pair_label in SCALE_PAIRS:
            paired = model_frame[
                ["item_id", f"delta_{scale_a}", f"delta_{scale_b}"]
            ].dropna()
            vector_a = paired[f"delta_{scale_a}"].to_numpy(dtype=float)
            vector_b = paired[f"delta_{scale_b}"].to_numpy(dtype=float)
            n_items = len(paired)

            rho = spearman_rho(vector_a, vector_b)
            rho_low, rho_high = bootstrap_ci(
                spearman_rho, vector_a, vector_b, N_BOOTSTRAP, BOOTSTRAP_SEED,
            )
            tau = kendall_tau(vector_a, vector_b)
            tau_low, tau_high = bootstrap_ci(
                kendall_tau, vector_a, vector_b, N_BOOTSTRAP, BOOTSTRAP_SEED + 1,
            )
            rows.append({
                "model": model_label,
                "scale_a": scale_a,
                "scale_b": scale_b,
                "scale_pair_label": pair_label,
                "n_items": n_items,
                "spearman_rho": rho,
                "spearman_rho_ci_low": rho_low,
                "spearman_rho_ci_high": rho_high,
                "kendall_tau": tau,
                "kendall_tau_ci_low": tau_low,
                "kendall_tau_ci_high": tau_high,
            })
    return pd.DataFrame(rows)


def build_figure(metric_table: pd.DataFrame) -> plt.Figure:
    """Three-panel forest: each panel is one scale-pair; rows = models.

    Models are sorted by Spearman rho on the first scale pair (3pt vs 7pt)
    so the eye can scan vertically across the three panels and check whether
    the within-model ranking of cross-scale similarity is itself consistent.
    """
    configure_plot_theme()
    plt.rcParams["font.family"] = [
        "Noto Sans CJK JP", *plt.rcParams.get("font.family", []),
    ]
    color_mapping = build_color_mapping(metric_table["model"].unique())

    sort_pair = SCALE_PAIRS[0][2]
    sort_frame = metric_table.loc[
        metric_table["scale_pair_label"] == sort_pair
    ].sort_values("spearman_rho", ascending=True)
    model_order = sort_frame["model"].tolist()
    y_positions = np.arange(len(model_order))

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 5.4), sharex=True, sharey=True)

    rho_low_min = metric_table["spearman_rho_ci_low"].min()
    rho_high_max = metric_table["spearman_rho_ci_high"].max()
    x_padding = 0.04 * (rho_high_max - rho_low_min)
    shared_xlim = (rho_low_min - x_padding, rho_high_max + x_padding)

    for panel_index, (_, _, pair_label) in enumerate(SCALE_PAIRS):
        axis = axes[panel_index]
        panel_frame = metric_table.loc[
            metric_table["scale_pair_label"] == pair_label
        ].set_index("model").reindex(model_order).reset_index()

        rho = panel_frame["spearman_rho"].to_numpy()
        rho_low = panel_frame["spearman_rho_ci_low"].to_numpy()
        rho_high = panel_frame["spearman_rho_ci_high"].to_numpy()

        axis.hlines(
            y=y_positions, xmin=rho_low, xmax=rho_high,
            color="#22303C", linewidth=1.4, alpha=0.85,
        )
        axis.scatter(
            rho, y_positions, s=66,
            color=[color_mapping[m] for m in model_order],
            edgecolors="#22303C", linewidths=0.8, zorder=3,
        )

        # Kendall tau as hollow diamonds, slightly offset, for robustness.
        tau = panel_frame["kendall_tau"].to_numpy()
        tau_low = panel_frame["kendall_tau_ci_low"].to_numpy()
        tau_high = panel_frame["kendall_tau_ci_high"].to_numpy()
        axis.hlines(
            y=y_positions - 0.22, xmin=tau_low, xmax=tau_high,
            color="#8897A4", linewidth=1.0, alpha=0.85,
        )
        axis.scatter(
            tau, y_positions - 0.22, s=44,
            facecolors="white", edgecolors="#22303C", linewidths=1.0,
            marker="D", zorder=3,
        )

        axis.set_yticks(y_positions)
        axis.set_yticklabels(model_order, fontsize=9.4)
        axis.set_title(pair_label, fontsize=12, color=TEXT_COLOR)
        axis.axvline(0.0, color=SPINE_COLOR, linewidth=0.8, linestyle="--", zorder=1)
        axis.set_xlim(shared_xlim)
        axis.set_xlabel(r"correlation of per-item $\Delta_i$ across scales")
        axis.tick_params(axis="both", length=0)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=8,
               markerfacecolor="#2F3E4D", markeredgecolor="#22303C",
               label=r"Spearman $\rho$"),
        Line2D([0], [0], marker="D", linestyle="none", markersize=7,
               markerfacecolor="white", markeredgecolor="#22303C",
               label=r"Kendall $\tau$"),
    ]
    figure.legend(
        handles=legend_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=10,
    )
    figure.suptitle(
        r"Within-model cross-scale similarity of per-item bias $\Delta_i$",
        fontsize=13.6, color=TEXT_COLOR, y=0.995,
    )
    figure.tight_layout(rect=(0.01, 0.06, 0.99, 0.96))
    return figure


def main() -> None:
    """Compute, save, and report the cross-scale similarity table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_table = compute_within_model_table()
    metric_table.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote: {OUTPUT_CSV}\n")

    print("=== Within-model cross-scale similarity (Spearman rho [95% CI]) ===")
    pivot = metric_table.pivot(
        index="model", columns="scale_pair_label", values="spearman_rho"
    )[[label for _, _, label in SCALE_PAIRS]]
    print(pivot.round(3).to_string())
    print()
    print("Per-pair median rho across the 9 models:")
    print(pivot.median().round(3).to_string())

    figure = build_figure(metric_table)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"\nWrote: {OUTPUT_PDF}")
    print(f"Wrote: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
