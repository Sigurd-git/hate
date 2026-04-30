"""Generate main-text figures for the revised paper narrative.

The figures are deliberately tied to the paper's four central empirical
questions:

1. Does the model-human magnitude comparison change with the human reference?
2. Do humans and LLMs locate the effect in the same attack domains?
3. Does rating granularity change the apparent model-human gap?
4. Does aggregate magnitude alignment imply item-level alignment?

Outputs are written to ``artifacts/paper_revision/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "artifacts" / "human_model_dz_descriptive_comparison"
FOLLOWUP_DIR = PROJECT_ROOT / "artifacts" / "paper_followups"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "paper_revision"

ITEM_DELTA_PATH = INPUT_DIR / "all_rater_item_deltas.csv"
DZ_STATS_PATH = INPUT_DIR / "dz_descriptive_statistics.csv"
HUMAN_DOMAIN_PATH = FOLLOWUP_DIR / "human_by_level1_by_scale_dz.csv"
MODEL_DOMAIN_PATH = PROJECT_ROOT / "artifacts" / "model_by_level1_by_scale.csv"
ICC_PATH = PROJECT_ROOT / "artifacts" / "human_model_delta_similarity_by_participant_gender.csv"
OVERALL_ICC_PATH = PROJECT_ROOT / "artifacts" / "human_model_delta_similarity.csv"

BOOTSTRAP_SEED = 20260429
BOOTSTRAP_ITERATIONS = 10_000

SCALE_ORDER = ["attack_3pt", "attack_7pt_likert", "attack_slider_0_100"]
SCALE_LABELS = {
    "attack_3pt": "3-point",
    "attack_7pt_likert": "7-point",
    "attack_slider_0_100": "Slider",
}
HUMAN_REFERENCE_ORDER = ["human_male", "human_all", "human_female"]
HUMAN_REFERENCE_LABELS = {
    "human_male": "Male participants",
    "human_all": "Overall humans",
    "human_female": "Female participants",
}
MODEL_ORDER = [
    "DeepSeek-R1",
    "DeepSeek-V3.2",
    "GLM-4.6",
    "Llama-4-Maverick",
    "Gemma-4-31B",
    "Claude-4.5",
    "Qwen-2.5-72B",
    "Kimi-K2",
    "GPT-5.1",
]
DISPLAY_MODEL_LABELS = {
    "Claude-4.5": "Claude-Opus-4.5",
    "Kimi-K2": "Kimi-K2-Thinking",
}
DOMAIN_ORDER = [
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
DOMAIN_SHORT_LABELS = {
    "性化攻击（性羞辱）": "Sexualization",
    "外貌形象攻击": "Appearance",
    "性别角色/性别表达攻击": "Gender role/expression",
    "人际关系攻击": "Interpersonal relations",
    "道德品行攻击": "Moral character",
    "经济资源攻击": "Economic resources",
    "社会地位攻击": "Social status",
    "情绪稳定攻击": "Emotional stability",
    "能力才干攻击": "Capability/competence",
    "智力理性攻击": "Intellectual rationality",
}


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 3,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def cohens_dz(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size < 2:
        return float("nan")
    standard_deviation = float(np.std(finite_values, ddof=1))
    if np.isclose(standard_deviation, 0.0):
        return 0.0
    return float(np.mean(finite_values) / standard_deviation)


def display_model_label(label: str) -> str:
    return DISPLAY_MODEL_LABELS.get(label, label)


def save_figure(figure: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(figure)


def make_reference_group_bootstrap_forest(item_frame: pd.DataFrame) -> pd.DataFrame:
    three_point_frame = item_frame.loc[item_frame["condition"] == "attack_3pt"].copy()
    pivot_frame = three_point_frame.pivot_table(
        index="item_id",
        columns="rater_id",
        values="delta_female_minus_male",
        aggfunc="mean",
    )
    label_map = (
        three_point_frame[["rater_id", "rater_label"]]
        .drop_duplicates()
        .set_index("rater_id")["rater_label"]
        .to_dict()
    )
    model_ids = [
        rater_id
        for rater_id, label in label_map.items()
        if rater_id not in HUMAN_REFERENCE_ORDER and label in MODEL_ORDER
    ]
    model_ids = sorted(model_ids, key=lambda value: MODEL_ORDER.index(label_map[value]))

    random_generator = np.random.default_rng(BOOTSTRAP_SEED)
    n_items = len(pivot_frame)
    rows: list[dict[str, object]] = []
    for reference_id in HUMAN_REFERENCE_ORDER:
        reference_values = pivot_frame[reference_id].to_numpy(dtype=float)
        reference_dz = cohens_dz(reference_values)
        for model_id in model_ids:
            model_values = pivot_frame[model_id].to_numpy(dtype=float)
            model_dz = cohens_dz(model_values)
            observed_gap = model_dz - reference_dz
            paired_values = np.column_stack([model_values, reference_values])
            bootstrap_values = np.empty(BOOTSTRAP_ITERATIONS)
            for iteration_index in range(BOOTSTRAP_ITERATIONS):
                sample_indices = random_generator.integers(0, n_items, n_items)
                bootstrap_sample = paired_values[sample_indices]
                bootstrap_values[iteration_index] = cohens_dz(
                    bootstrap_sample[:, 0]
                ) - cohens_dz(bootstrap_sample[:, 1])
            ci_low, ci_high = np.percentile(bootstrap_values, [2.5, 97.5])
            rows.append(
                {
                    "reference_id": reference_id,
                    "reference_label": HUMAN_REFERENCE_LABELS[reference_id],
                    "model_id": model_id,
                    "model_label": display_model_label(label_map[model_id]),
                    "model_order": MODEL_ORDER.index(label_map[model_id]),
                    "reference_dz": reference_dz,
                    "model_dz": model_dz,
                    "delta_dz": observed_gap,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "interval_relation": "above"
                    if ci_low > 0
                    else "below"
                    if ci_high < 0
                    else "overlaps",
                }
            )

    bootstrap_frame = pd.DataFrame(rows)
    bootstrap_frame.to_csv(
        OUTPUT_DIR / "reference_group_delta_dz_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    palette = {"above": "#B23A48", "below": "#2F6F9F", "overlaps": "#6F7F8F"}
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 5.8), sharey=False, sharex=True)
    for axis, reference_id in zip(axes, HUMAN_REFERENCE_ORDER, strict=True):
        panel_frame = bootstrap_frame.loc[
            bootstrap_frame["reference_id"] == reference_id
        ].sort_values("model_order", ascending=False)
        y_positions = np.arange(len(panel_frame))
        for row_index, (_, row) in enumerate(panel_frame.iterrows()):
            color = palette[row["interval_relation"]]
            axis.hlines(row_index, row["ci_low"], row["ci_high"], color=color, linewidth=2.0)
            axis.scatter(row["delta_dz"], row_index, s=48, color=color, edgecolor="white", zorder=3)
        axis.axvline(0, color="#2B2B2B", linewidth=1.0, linestyle="--")
        axis.set_title(HUMAN_REFERENCE_LABELS[reference_id], fontsize=11.5)
        axis.set_xlabel(r"Model $d_z$ $-$ human-reference $d_z$")
        axis.grid(axis="x", color="#E6EAF0")
        axis.grid(axis="y", visible=False)
        axis.set_yticks(y_positions)
        if axis is axes[0]:
            axis.set_yticklabels(panel_frame["model_label"], fontsize=9.0)
            axis.tick_params(axis="y", length=0, labelleft=True)
        else:
            axis.set_yticklabels([])
            axis.tick_params(axis="y", length=0, labelleft=False)
        axis.set_xlim(-0.58, 0.50)
    figure.suptitle(
        r"The magnitude verdict changes with the human reference group (3-point scale)",
        fontsize=13.0,
        y=1.02,
    )
    save_figure(figure, "fig_r3_reference_group_delta_dz_forest")
    return bootstrap_frame


def make_domain_alignment_forest() -> None:
    # Project convention: ten-domain comparisons must be shown as forest plots,
    # not heatmaps, so readers can inspect domain order and effect size directly.
    human_domain = pd.read_csv(HUMAN_DOMAIN_PATH)
    model_domain = pd.read_csv(MODEL_DOMAIN_PATH)

    model_summary = (
        model_domain.groupby(["scale_key", "一级-攻击领域"], as_index=False)
        .agg(dz=("cohens_dz", "median"))
        .assign(rater_group="LLM median", rater_order=3)
        .rename(columns={"scale_key": "condition", "一级-攻击领域": "dimension_1"})
    )
    human_summary = human_domain.loc[
        human_domain["rater_id"].isin(["human_all", "human_female", "human_male"])
    ].copy()
    human_summary["rater_group"] = human_summary["rater_id"].map(
        {
            "human_all": "Overall humans",
            "human_female": "Female participants",
            "human_male": "Male participants",
        }
    )
    human_summary["rater_order"] = human_summary["rater_id"].map(
        {"human_all": 0, "human_female": 1, "human_male": 2}
    )
    combined_frame = pd.concat(
        [
            human_summary[
                ["condition", "dimension_1", "dz", "rater_group", "rater_order"]
            ],
            model_summary[
                ["condition", "dimension_1", "dz", "rater_group", "rater_order"]
            ],
        ],
        ignore_index=True,
    )
    combined_frame["scale_label"] = combined_frame["condition"].map(SCALE_LABELS)
    combined_frame["domain_label"] = combined_frame["dimension_1"].map(DOMAIN_SHORT_LABELS)
    combined_frame["domain_order"] = combined_frame["dimension_1"].map(
        {domain: index for index, domain in enumerate(DOMAIN_ORDER)}
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.6, 7.0), sharey=True)
    palette = {
        "Overall humans": "#1f77b4",
        "Female participants": "#d62728",
        "Male participants": "#2ca02c",
        "LLM median": "#4c4c4c",
    }
    marker_map = {
        "Overall humans": "o",
        "Female participants": "D",
        "Male participants": "s",
        "LLM median": "^",
    }
    y_positions = {domain: len(DOMAIN_ORDER) - 1 - index for index, domain in enumerate(DOMAIN_ORDER)}
    vertical_offsets = {
        "Overall humans": 0.24,
        "Female participants": 0.08,
        "Male participants": -0.08,
        "LLM median": -0.24,
    }
    x_limit = max(1.6, float(combined_frame["dz"].abs().max()) * 1.08)
    for axis, scale_key in zip(axes, SCALE_ORDER, strict=True):
        panel_frame = combined_frame.loc[combined_frame["condition"] == scale_key]
        for domain in DOMAIN_ORDER:
            axis.axhline(y_positions[domain], color="#e5e7eb", linewidth=0.7, zorder=0)
        axis.axvline(0, color="#6b7280", linestyle="--", linewidth=0.9, zorder=1)
        for rater_group in ["Overall humans", "Female participants", "Male participants", "LLM median"]:
            rater_frame = panel_frame.loc[panel_frame["rater_group"] == rater_group].copy()
            rater_frame["y"] = (
                rater_frame["dimension_1"].map(y_positions) + vertical_offsets[rater_group]
            )
            axis.scatter(
                rater_frame["dz"],
                rater_frame["y"],
                s=38,
                marker=marker_map[rater_group],
                color=palette[rater_group],
                edgecolor="white",
                linewidth=0.5,
                label=rater_group,
                zorder=3,
            )
        axis.set_title(SCALE_LABELS[scale_key], fontsize=11.5)
        axis.set_xlabel(r"Cohen's $d_z$")
        axis.set_xlim(-0.35, x_limit)
        axis.set_ylim(-0.8, len(DOMAIN_ORDER) - 0.2)
        axis.set_yticks([y_positions[domain] for domain in DOMAIN_ORDER])
        axis.set_yticklabels([DOMAIN_SHORT_LABELS[domain] for domain in DOMAIN_ORDER], fontsize=9.1)
        axis.tick_params(axis="x", labelsize=8.8)
        axis.tick_params(axis="y", labelsize=9.1)
        axis.grid(axis="x", color="#e5e7eb", linewidth=0.7)
        axis.grid(axis="y", visible=False)
    axes[0].set_ylabel("First-level attack domain")
    for tick_label in axes[0].get_yticklabels()[:2]:
        tick_label.set_fontweight("bold")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    figure.suptitle("Domain-level effects concentrate in sexualization and appearance", fontsize=13.0, y=0.99)
    figure.subplots_adjust(bottom=0.13, top=0.91, wspace=0.18)
    save_figure(figure, "fig_r4_domain_alignment_forest")


def make_scale_response_trajectory(dz_stats: pd.DataFrame) -> None:
    plot_frame = dz_stats.copy()
    plot_frame["scale_short"] = plot_frame["condition"].map(SCALE_LABELS)
    plot_frame["scale_order"] = plot_frame["condition"].map(
        {scale_key: index for index, scale_key in enumerate(SCALE_ORDER)}
    )
    plot_frame["display_label"] = plot_frame["rater_label"].map(display_model_label)
    plot_frame.loc[plot_frame["rater_id"] == "human_all", "display_label"] = "Overall humans"
    plot_frame.loc[plot_frame["rater_id"] == "human_male", "display_label"] = "Male participants"
    plot_frame.loc[plot_frame["rater_id"] == "human_female", "display_label"] = "Female participants"

    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    model_frame = plot_frame.loc[plot_frame["rater_kind"] == "model"]
    for _, group in model_frame.groupby("rater_id"):
        group = group.sort_values("scale_order")
        label = display_model_label(group["rater_label"].iloc[0])
        highlight = label in {"Gemma-4-31B", "GPT-5.1", "Claude-Opus-4.5"}
        axis.plot(
            group["scale_short"],
            group["cohens_dz"],
            color="#7F8790" if not highlight else {"Gemma-4-31B": "#B23A48", "GPT-5.1": "#7A4EA3", "Claude-Opus-4.5": "#D08A24"}[label],
            linewidth=1.2 if not highlight else 2.3,
            alpha=0.45 if not highlight else 0.95,
            marker="o",
            markersize=3.8 if not highlight else 5.2,
        )
        if highlight:
            axis.text(
                2.04,
                group.sort_values("scale_order")["cohens_dz"].iloc[-1],
                label,
                va="center",
                fontsize=8.8,
                color={"Gemma-4-31B": "#B23A48", "GPT-5.1": "#7A4EA3", "Claude-Opus-4.5": "#D08A24"}[label],
            )

    human_styles = {
        "Overall humans": ("#111111", "-", 2.8),
        "Female participants": ("#2F6F9F", "-", 2.5),
        "Male participants": ("#3A7D44", "-", 2.5),
    }
    for human_id, group in plot_frame.loc[plot_frame["rater_kind"] == "human"].groupby("rater_id"):
        group = group.sort_values("scale_order")
        label = group["display_label"].iloc[0]
        color, linestyle, linewidth = human_styles[label]
        axis.plot(
            group["scale_short"],
            group["cohens_dz"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker="o",
            markersize=6.0,
            label=label,
        )
    axis.set_ylabel(r"Cohen's $d_z$ of $\Delta_{F-M}$")
    axis.set_xlabel("Rating format")
    axis.set_ylim(0.0, 1.12)
    axis.set_xlim(-0.06, 2.55)
    axis.legend(frameon=False, loc="upper left")
    axis.grid(axis="y", color="#E6EAF0")
    axis.grid(axis="x", visible=False)
    axis.set_title("Finer scales dampen human effects but amplify most LLM effects", fontsize=12.5)
    save_figure(figure, "fig_r5_scale_response_trajectory")


def make_aggregate_item_dissociation(
    participant_gender_icc_frame: pd.DataFrame,
    overall_icc_frame: pd.DataFrame,
    dz_stats: pd.DataFrame,
) -> None:
    overall_icc = overall_icc_frame.copy()
    overall_icc["participant_gender_label"] = "Overall humans"
    overall_icc["participant_gender"] = "overall"
    combined_icc = pd.concat(
        [overall_icc, participant_gender_icc_frame],
        ignore_index=True,
        sort=False,
    )

    model_dz = dz_stats.loc[dz_stats["rater_kind"] == "model", [
        "rater_label", "condition", "cohens_dz"
    ]].rename(columns={"cohens_dz": "model_dz"})
    human_dz = dz_stats.loc[
        (dz_stats["rater_id"] == "human_all"), ["condition", "cohens_dz"]
    ].rename(columns={"cohens_dz": "overall_human_dz"})
    overall_icc["rater_label"] = overall_icc["model"].map(display_model_label)
    merged_frame = overall_icc.merge(
        model_dz,
        left_on=["model", "scale"],
        right_on=["rater_label", "condition"],
        how="left",
        suffixes=("", "_model_stats"),
    )
    if merged_frame["model_dz"].isna().all():
        model_dz_alt = dz_stats.loc[dz_stats["rater_kind"] == "model", [
            "rater_label", "condition", "cohens_dz"
        ]].copy()
        model_dz_alt["display_label"] = model_dz_alt["rater_label"].map(display_model_label)
        merged_frame = overall_icc.merge(
            model_dz_alt,
            left_on=["rater_label", "scale"],
            right_on=["display_label", "condition"],
            how="left",
        ).rename(columns={"cohens_dz": "model_dz"})
    merged_frame = merged_frame.merge(human_dz, left_on="scale", right_on="condition", how="left")
    merged_frame["abs_dz_gap"] = (merged_frame["model_dz"] - merged_frame["overall_human_dz"]).abs()

    top_rows = (
        combined_icc.sort_values("icc_2_1", ascending=False)
        .groupby(["participant_gender_label", "scale_label"], as_index=False)
        .first()
    )
    top_rows.to_csv(OUTPUT_DIR / "item_level_top_icc_models.csv", index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.4, 4.8),
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )
    scatter_frame = merged_frame.dropna(subset=["abs_dz_gap", "icc_2_1"]).copy()
    scatter_frame["scale_display"] = scatter_frame["scale"].map(SCALE_LABELS).fillna(scatter_frame["scale_label"])
    sns.scatterplot(
        data=scatter_frame,
        x="abs_dz_gap",
        y="icc_2_1",
        hue="scale_display",
        style="scale_display",
        s=72,
        ax=axes[0],
        palette=["#4C78A8", "#F58518", "#54A24B"],
    )
    axes[0].axhline(0.50, color="#B23A48", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel(r"Absolute aggregate gap vs. overall humans ($|d_z^{model}-d_z^{human}|$)")
    axes[0].set_ylabel("Item-level ICC(2,1)")
    axes[0].set_title("Aggregate closeness does not imply\nitem-level agreement", fontsize=10.5)
    axes[0].legend(frameon=False, title="")
    axes[0].set_ylim(-0.08, 0.55)

    heatmap_frame = combined_icc.copy()
    heatmap_frame["reference"] = heatmap_frame["participant_gender_label"].replace(
        {"男性被试": "Male participants", "女性被试": "Female participants"}
    )
    heatmap_frame["scale_display"] = heatmap_frame["scale"].map(SCALE_LABELS).fillna(
        heatmap_frame["scale_label"].replace(
            {"7-point Likert": "7-point", "0-100 slider": "Slider"}
        )
    )
    heatmap_frame["cell"] = heatmap_frame["reference"] + "\n" + heatmap_frame["scale_display"]
    top_model_frame = (
        heatmap_frame.sort_values("icc_2_1", ascending=False)
        .groupby("cell", as_index=False)
        .first()
    )
    top_model_frame["model_display"] = top_model_frame["model"].map(display_model_label)
    matrix = top_model_frame.pivot_table(
        index="reference",
        columns="scale_display",
        values="icc_2_1",
        aggfunc="max",
    ).reindex(index=["Overall humans", "Female participants", "Male participants"])
    scale_columns = [label for label in ["3-point", "7-point", "Slider"] if label in matrix.columns]
    matrix = matrix.reindex(columns=scale_columns)
    annotations = matrix.copy().astype(object)
    for reference_label in matrix.index:
        for scale_label in matrix.columns:
            row = top_model_frame.loc[
                (top_model_frame["reference"] == reference_label)
                & (top_model_frame["scale_display"] == scale_label)
            ]
            if row.empty:
                annotations.loc[reference_label, scale_label] = ""
            else:
                annotations.loc[reference_label, scale_label] = (
                    f"{row['icc_2_1'].iloc[0]:.2f}\n{row['model_display'].iloc[0]}"
                )
    sns.heatmap(
        matrix,
        ax=axes[1],
        cmap="Blues",
        vmin=0,
        vmax=0.50,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 8.4},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Best ICC(2,1) in cell"},
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].set_title("The closest model changes across\nreference groups and scales", fontsize=10.5)
    axes[1].tick_params(axis="x", labelrotation=0)
    axes[1].tick_params(axis="y", labelrotation=0, labelsize=9.2, pad=2)
    figure.suptitle("Item-level agreement remains weak even when aggregate effects match", fontsize=13.0, y=1.03)
    figure.subplots_adjust(wspace=0.35)
    save_figure(figure, "fig_r6_aggregate_item_dissociation")


def main() -> None:
    configure_plotting()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    item_frame = pd.read_csv(ITEM_DELTA_PATH)
    dz_stats = pd.read_csv(DZ_STATS_PATH)
    participant_gender_icc_frame = pd.read_csv(ICC_PATH)
    overall_icc_frame = pd.read_csv(OVERALL_ICC_PATH)

    make_reference_group_bootstrap_forest(item_frame)
    make_domain_alignment_forest()
    make_scale_response_trajectory(dz_stats)
    make_aggregate_item_dissociation(
        participant_gender_icc_frame, overall_icc_frame, dz_stats
    )
    print(f"Wrote revised paper figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
