"""Within-human cross-scale similarity of per-item bias patterns.

Companion to ``within_model_cross_scale_similarity.py``. For each human
subgroup (all participants / male participants / female participants) and
for each pair of response scales (3pt vs 7pt, 7pt vs slider, 3pt vs slider),
we compute:

    rho = Spearman correlation of Delta_i^(scale A) and Delta_i^(scale B)

across the 371 paired items. Delta_i^(scale) is the per-item mean-score
difference (female-referent minus male-referent) aggregated across the
subgroup's participants at that scale. 95% CIs from bootstrap (B=2000,
item resample).

We also produce a human-vs-LLM joint figure that places each of the three
human subgroups alongside the nine LLMs on one axis per scale-pair, to
make the two populations directly comparable.

Outputs:
    artifacts/within_human_cross_scale_similarity.csv
    artifacts/within_human_cross_scale_similarity.pdf
    artifacts/within_human_cross_scale_similarity.png
    artifacts/within_rater_cross_scale_similarity_human_vs_llm.pdf
    artifacts/within_rater_cross_scale_similarity_human_vs_llm.png
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
    HUMAN_CSV,
    N_BOOTSTRAP,
    SCALE_ORDER,
    bootstrap_ci,
)
from plot_f1_brier import (
    SPINE_COLOR,
    SUBTLE_TEXT_COLOR,
    TEXT_COLOR,
    build_color_mapping,
    configure_plot_theme,
)
from within_model_cross_scale_similarity import (
    SCALE_PAIRS,
    compute_within_model_table,
)

# Paths -----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_CSV = OUTPUT_DIR / "within_human_cross_scale_similarity.csv"
OUTPUT_PDF = OUTPUT_DIR / "within_human_cross_scale_similarity.pdf"
OUTPUT_PNG = OUTPUT_DIR / "within_human_cross_scale_similarity.png"
JOINT_PDF = OUTPUT_DIR / "within_rater_cross_scale_similarity_human_vs_llm.pdf"
JOINT_PNG = OUTPUT_DIR / "within_rater_cross_scale_similarity_human_vs_llm.png"

# Subgroup registry: (filter predicate, Chinese display label, code key).
SUBGROUPS: list[tuple[str, str, str]] = [
    ("all", "全体被试", "ALL"),
    ("男", "男性被试", "M"),
    ("女", "女性被试", "F"),
]

HUMAN_COLOR_MAP = {
    "全体被试": "#2F3E4D",
    "男性被试": "#4C7BB5",
    "女性被试": "#C9736B",
}

# English display labels used on matplotlib axes to avoid CJK font warnings
# (the Chinese-language body text in the .tex refers to these by the Chinese
# subgroup name, so only the figure axes need to be English).
HUMAN_LABEL_EN = {
    "全体被试": "Humans (all)",
    "男性被试": "Humans (male)",
    "女性被试": "Humans (female)",
}


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation via pandas (handles ties cleanly)."""
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau-b correlation."""
    if len(a) < 2:
        return float("nan")
    return float(stats.kendalltau(a, b).correlation)


def load_human_deltas_by_subgroup(subgroup_key: str) -> dict[str, pd.DataFrame]:
    """Per-scale wide-form per-item Delta computed from the requested subgroup.

    subgroup_key = "all" -> keep all participants; "男"/"女" -> filter gender.
    """
    human_long = pd.read_csv(HUMAN_CSV, low_memory=False)
    if subgroup_key != "all":
        human_long = human_long.loc[human_long["gender"] == subgroup_key].copy()
    human_long["response_value"] = pd.to_numeric(
        human_long["response_value"], errors="coerce"
    )

    result_per_scale: dict[str, pd.DataFrame] = {}
    for scale_key, _ in SCALE_ORDER:
        scale_rows = human_long.loc[human_long["condition"] == scale_key].copy()
        version_means = (
            scale_rows.groupby(["item_id", "shown_version"])["response_value"]
            .mean()
            .unstack("shown_version")
        )
        delta = version_means["女人版"] - version_means["男人版"]
        delta.name = "delta"
        result_per_scale[scale_key] = delta.reset_index()
    return result_per_scale


def compute_within_human_table() -> pd.DataFrame:
    """Three subgroups x three scale pairs -> nine (rho, tau) rows with CIs."""
    rows: list[dict] = []
    for subgroup_key, subgroup_label, subgroup_code in SUBGROUPS:
        deltas = load_human_deltas_by_subgroup(subgroup_key)
        for scale_a, scale_b, pair_label in SCALE_PAIRS:
            frame_a = deltas[scale_a].rename(columns={"delta": "delta_a"})
            frame_b = deltas[scale_b].rename(columns={"delta": "delta_b"})
            merged = frame_a.merge(frame_b, on="item_id", how="inner").dropna()
            vector_a = merged["delta_a"].to_numpy(dtype=float)
            vector_b = merged["delta_b"].to_numpy(dtype=float)

            rho = spearman_rho(vector_a, vector_b)
            rho_low, rho_high = bootstrap_ci(
                spearman_rho, vector_a, vector_b, N_BOOTSTRAP, BOOTSTRAP_SEED,
            )
            tau = kendall_tau(vector_a, vector_b)
            tau_low, tau_high = bootstrap_ci(
                kendall_tau, vector_a, vector_b, N_BOOTSTRAP, BOOTSTRAP_SEED + 1,
            )
            rows.append({
                "subgroup_key": subgroup_key,
                "subgroup_label": subgroup_label,
                "subgroup_code": subgroup_code,
                "scale_a": scale_a,
                "scale_b": scale_b,
                "scale_pair_label": pair_label,
                "n_items": len(merged),
                "spearman_rho": rho,
                "spearman_rho_ci_low": rho_low,
                "spearman_rho_ci_high": rho_high,
                "kendall_tau": tau,
                "kendall_tau_ci_low": tau_low,
                "kendall_tau_ci_high": tau_high,
            })
    return pd.DataFrame(rows)


def build_human_only_figure(human_table: pd.DataFrame) -> plt.Figure:
    """Three-panel forest: one panel per scale pair; rows = 3 subgroups."""
    configure_plot_theme()
    figure, axes = plt.subplots(1, 3, figsize=(10.6, 3.4), sharex=True, sharey=True)

    rho_low_min = human_table["spearman_rho_ci_low"].min()
    rho_high_max = human_table["spearman_rho_ci_high"].max()
    x_pad = 0.04 * (rho_high_max - rho_low_min)
    xlim = (rho_low_min - x_pad, rho_high_max + x_pad)

    subgroup_order = ["全体被试", "男性被试", "女性被试"]
    y_positions = np.arange(len(subgroup_order))

    for panel_index, (_, _, pair_label) in enumerate(SCALE_PAIRS):
        axis = axes[panel_index]
        panel = human_table.loc[
            human_table["scale_pair_label"] == pair_label
        ].set_index("subgroup_label").reindex(subgroup_order).reset_index()

        rho = panel["spearman_rho"].to_numpy()
        rho_low = panel["spearman_rho_ci_low"].to_numpy()
        rho_high = panel["spearman_rho_ci_high"].to_numpy()
        colors = [HUMAN_COLOR_MAP[name] for name in subgroup_order]

        axis.hlines(
            y=y_positions, xmin=rho_low, xmax=rho_high,
            color="#22303C", linewidth=1.4, alpha=0.85,
        )
        axis.scatter(
            rho, y_positions, s=72, color=colors,
            edgecolors="#22303C", linewidths=0.8, zorder=3,
        )
        axis.set_yticks(y_positions)
        axis.set_yticklabels(
            [HUMAN_LABEL_EN[name] for name in subgroup_order], fontsize=10
        )
        axis.set_title(pair_label, fontsize=12, color=TEXT_COLOR)
        axis.axvline(0.0, color=SPINE_COLOR, linewidth=0.8, linestyle="--", zorder=1)
        axis.set_xlim(xlim)
        axis.set_xlabel(r"Spearman $\rho$ of $\Delta_i$")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(axis="both", length=0)

    figure.suptitle(
        r"Within-human cross-scale similarity of per-item bias $\Delta_i$",
        fontsize=13.4, color=TEXT_COLOR, y=1.02,
    )
    figure.tight_layout()
    return figure


def build_joint_figure(
    human_table: pd.DataFrame, model_table: pd.DataFrame
) -> plt.Figure:
    """Place three human subgroups alongside nine LLMs in one figure.

    Produces a 1 x 3 grid (one panel per scale pair). Within each panel,
    y-axis is rater (3 humans grouped at top, 9 LLMs below); x-axis is
    Spearman rho with 95% bootstrap CI. Colors distinguish human vs LLM.
    """
    configure_plot_theme()
    model_color_map = build_color_mapping(model_table["model"].unique())

    figure, axes = plt.subplots(1, 3, figsize=(13.6, 6.2), sharex=True, sharey=True)

    rho_low_min = min(
        human_table["spearman_rho_ci_low"].min(),
        model_table["spearman_rho_ci_low"].min(),
    )
    rho_high_max = max(
        human_table["spearman_rho_ci_high"].max(),
        model_table["spearman_rho_ci_high"].max(),
    )
    x_pad = 0.04 * (rho_high_max - rho_low_min)
    xlim = (rho_low_min - x_pad, rho_high_max + x_pad)

    # Row order: humans at top, separator gap, then LLMs sorted by rho on the
    # first scale-pair so that ordering matches the existing LLM figure.
    human_order = ["女性被试", "男性被试", "全体被试"]
    first_pair_label = SCALE_PAIRS[0][2]
    llm_sort = model_table.loc[
        model_table["scale_pair_label"] == first_pair_label
    ].sort_values("spearman_rho", ascending=True)
    llm_order = llm_sort["model"].tolist()

    row_labels = (
        llm_order
        + [""]  # blank gap row between LLM block and human block
        + [HUMAN_LABEL_EN[name] for name in human_order]
    )
    y_positions = np.arange(len(row_labels))

    for panel_index, (_, _, pair_label) in enumerate(SCALE_PAIRS):
        axis = axes[panel_index]

        llm_panel = model_table.loc[
            model_table["scale_pair_label"] == pair_label
        ].set_index("model").reindex(llm_order).reset_index()
        human_panel = human_table.loc[
            human_table["scale_pair_label"] == pair_label
        ].set_index("subgroup_label").reindex(human_order).reset_index()

        # LLM rows.
        for row_index, (_, llm_row) in enumerate(llm_panel.iterrows()):
            y = y_positions[row_index]
            axis.hlines(
                y=y,
                xmin=llm_row["spearman_rho_ci_low"],
                xmax=llm_row["spearman_rho_ci_high"],
                color="#22303C", linewidth=1.3, alpha=0.85,
            )
            axis.scatter(
                llm_row["spearman_rho"], y, s=66,
                color=model_color_map[llm_row["model"]],
                edgecolors="#22303C", linewidths=0.8, zorder=3,
            )

        # Human rows (plotted above the gap).
        human_start = len(llm_order) + 1
        for row_offset, (_, human_row) in enumerate(human_panel.iterrows()):
            y = y_positions[human_start + row_offset]
            axis.hlines(
                y=y,
                xmin=human_row["spearman_rho_ci_low"],
                xmax=human_row["spearman_rho_ci_high"],
                color="#22303C", linewidth=1.6, alpha=0.9,
            )
            axis.scatter(
                human_row["spearman_rho"], y, s=100,
                color=HUMAN_COLOR_MAP[human_row["subgroup_label"]],
                edgecolors="#22303C", linewidths=1.0, marker="s", zorder=3,
            )

        axis.axhline(y=len(llm_order) + 0.5, color=SUBTLE_TEXT_COLOR,
                     linewidth=0.6, linestyle=":", zorder=1)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(row_labels, fontsize=9.6)
        axis.set_title(pair_label, fontsize=12.4, color=TEXT_COLOR)
        axis.axvline(0.0, color=SPINE_COLOR, linewidth=0.8, linestyle="--", zorder=1)
        axis.set_xlim(xlim)
        axis.set_xlabel(r"Spearman $\rho$ of $\Delta_i$ across scales")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(axis="both", length=0)

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=10,
               markerfacecolor="#2F3E4D", markeredgecolor="#22303C",
               label="human subgroup"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=8,
               markerfacecolor="#8897A4", markeredgecolor="#22303C",
               label="LLM"),
    ]
    figure.legend(
        handles=legend_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=11,
    )
    figure.suptitle(
        r"Cross-scale $\Delta_i$ similarity: human subgroups vs. nine LLMs",
        fontsize=13.6, color=TEXT_COLOR, y=1.00,
    )
    figure.tight_layout(rect=(0.01, 0.04, 0.99, 0.97))
    return figure


def main() -> None:
    """Compute, write, and render human + joint cross-scale tables/figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    human_table = compute_within_human_table()
    human_table.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote: {OUTPUT_CSV}")

    print("\n=== Within-human cross-scale Spearman rho (with 95% CI) ===")
    pivot = human_table.pivot(
        index="subgroup_label", columns="scale_pair_label", values="spearman_rho"
    )[[label for _, _, label in SCALE_PAIRS]]
    print(pivot.round(3).to_string())

    figure = build_human_only_figure(human_table)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote: {OUTPUT_PDF}")
    print(f"Wrote: {OUTPUT_PNG}")

    # Re-use the LLM table we already ship; recompute instead of reading CSV
    # so the two sides are guaranteed to use the same pipeline settings.
    model_table = compute_within_model_table()
    joint = build_joint_figure(human_table, model_table)
    joint.savefig(JOINT_PDF, bbox_inches="tight")
    joint.savefig(JOINT_PNG, dpi=220, bbox_inches="tight")
    plt.close(joint)
    print(f"Wrote: {JOINT_PDF}")
    print(f"Wrote: {JOINT_PNG}")

    print("\n=== Side-by-side median rho ===")
    print("Humans (median of 3 subgroups):")
    print(
        human_table.groupby("scale_pair_label")["spearman_rho"]
        .median().reindex([label for _, _, label in SCALE_PAIRS]).round(3).to_string()
    )
    print("LLMs (median of 9 models):")
    print(
        model_table.groupby("scale_pair_label")["spearman_rho"]
        .median().reindex([label for _, _, label in SCALE_PAIRS]).round(3).to_string()
    )


if __name__ == "__main__":
    main()
