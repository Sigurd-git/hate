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
NATURAL_GENDER_TARGET_PATH = (
    PROJECT_ROOT / "artifacts" / "placeholder_a_1a_gender_target_mean_score.csv"
)

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
MODEL_LABEL_ALIASES = {
    "Claude Opus 4.5": "Claude-4.5",
    "DeepSeek R1": "DeepSeek-R1",
    "DeepSeek V3.2": "DeepSeek-V3.2",
    "GLM 4.6": "GLM-4.6",
    "Gemma 4 31B": "Gemma-4-31B",
    "Kimi K2": "Kimi-K2",
    "Llama Maverick": "Llama-4-Maverick",
    "Qwen 2.5 72B": "Qwen-2.5-72B",
    "GPT-5.1": "GPT-5.1",
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


def canonical_model_label(label: str) -> str:
    return MODEL_LABEL_ALIASES.get(label, label)


def icc_2_1_two_raters(first_values: np.ndarray, second_values: np.ndarray) -> float:
    paired_frame = pd.DataFrame({"first": first_values, "second": second_values}).dropna()
    if len(paired_frame) < 2:
        return float("nan")
    values = paired_frame.to_numpy(dtype=float)
    n_targets, n_raters = values.shape
    target_means = values.mean(axis=1, keepdims=True)
    rater_means = values.mean(axis=0, keepdims=True)
    grand_mean = values.mean()
    ss_targets = n_raters * np.sum((target_means - grand_mean) ** 2)
    ss_raters = n_targets * np.sum((rater_means - grand_mean) ** 2)
    ss_error = np.sum((values - target_means - rater_means + grand_mean) ** 2)
    ms_targets = ss_targets / (n_targets - 1)
    ms_raters = ss_raters / (n_raters - 1)
    ms_error = ss_error / ((n_targets - 1) * (n_raters - 1))
    denominator = ms_targets + (n_raters - 1) * ms_error + (
        n_raters * (ms_raters - ms_error) / n_targets
    )
    if np.isclose(denominator, 0.0):
        return float("nan")
    return float((ms_targets - ms_error) / denominator)


def save_figure(figure: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(figure)


def make_natural_gender_target_confound_figure() -> None:
    natural_frame = pd.read_csv(NATURAL_GENDER_TARGET_PATH)
    english_frame = natural_frame.loc[natural_frame["language"] == "en"].copy()
    english_frame["model_id"] = english_frame["model_label"].map(canonical_model_label)
    english_frame["model_display"] = english_frame["model_id"].map(display_model_label)
    english_frame["model_order"] = english_frame["model_id"].map(
        {model_id: index for index, model_id in enumerate(MODEL_ORDER)}
    )
    english_frame["setting_label"] = english_frame["setting"].map(
        {"zeroshot": "Zero-shot", "cot": "Chain-of-thought"}
    )
    english_frame["target_label"] = english_frame["gender_target"].map(
        {"female": "Female-targeted", "male": "Male-targeted"}
    )
    english_frame = english_frame.sort_values(["setting", "model_order", "gender_target"])

    summary_rows = []
    for setting, setting_frame in english_frame.groupby("setting", sort=False):
        pivot_frame = setting_frame.pivot_table(
            index="model_id", columns="gender_target", values="mean_score"
        )
        difference = pivot_frame["male"] - pivot_frame["female"]
        summary_rows.append(
            {
                "setting": setting,
                "female_targeted_n": int(
                    setting_frame.loc[setting_frame["gender_target"] == "female", "n"].iloc[0]
                ),
                "male_targeted_n": int(
                    setting_frame.loc[setting_frame["gender_target"] == "male", "n"].iloc[0]
                ),
                "female_targeted_model_mean": float(pivot_frame["female"].mean()),
                "male_targeted_model_mean": float(pivot_frame["male"].mean()),
                "mean_male_minus_female": float(difference.mean()),
                "models_with_male_higher": int((difference > 0).sum()),
                "model_count": int(difference.shape[0]),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "natural_gender_target_descriptive_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    palette = {"Female-targeted": "#4C78A8", "Male-targeted": "#E45756"}
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.2), sharey=True)
    for axis, setting in zip(axes, ["zeroshot", "cot"], strict=True):
        panel_frame = english_frame.loc[english_frame["setting"] == setting].copy()
        model_labels = [
            display_model_label(model_id)
            for model_id in MODEL_ORDER
            if model_id in set(panel_frame["model_id"])
        ]
        x_positions = np.arange(len(model_labels))
        bar_width = 0.36
        for offset, target_label in [(-bar_width / 2, "Female-targeted"), (bar_width / 2, "Male-targeted")]:
            target_frame = (
                panel_frame.loc[panel_frame["target_label"] == target_label]
                .set_index("model_display")
                .reindex(model_labels)
            )
            axis.bar(
                x_positions + offset,
                target_frame["mean_score"],
                width=bar_width,
                color=palette[target_label],
                label=target_label,
                edgecolor="white",
                linewidth=0.5,
            )
            axis.errorbar(
                x_positions + offset,
                target_frame["mean_score"],
                yerr=target_frame["sem_score"],
                fmt="none",
                ecolor="#2B2B2B",
                elinewidth=0.8,
                capsize=2.2,
                capthick=0.8,
            )
        sample_sizes = (
            panel_frame[["target_label", "n"]]
            .drop_duplicates()
            .set_index("target_label")["n"]
            .to_dict()
        )
        axis.set_title(
            f"{panel_frame['setting_label'].iloc[0]} "
            f"(female n={sample_sizes['Female-targeted']}, male n={sample_sizes['Male-targeted']})",
            fontsize=11.0,
        )
        axis.set_xticks(x_positions)
        axis.set_xticklabels(model_labels, rotation=42, ha="right")
        axis.set_ylim(0, 5)
        axis.set_xlabel("")
        axis.grid(axis="y", color="#E6EAF0")
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean attack rating in HateXplain (0-5)")
    axes[1].legend(frameon=False, loc="upper left")
    figure.suptitle(
        "Natural gender-target contrasts reverse direction because target and content are confounded",
        fontsize=13.0,
        y=1.02,
    )
    figure.subplots_adjust(wspace=0.08)
    save_figure(figure, "fig_r1_natural_gender_target_confound")


def make_direction_agreement_forest(item_frame: pd.DataFrame) -> None:
    relevant_frame = item_frame.loc[item_frame["condition"].isin(SCALE_ORDER)].copy()
    rater_label_frame = (
        relevant_frame[["rater_id", "rater_label", "rater_kind"]]
        .drop_duplicates()
        .copy()
    )
    rater_label_frame["rater_sort_label"] = rater_label_frame["rater_label"].map(
        canonical_model_label
    )
    rater_label_frame["display_label"] = rater_label_frame["rater_sort_label"].map(
        display_model_label
    )
    human_display_labels = {
        "human_all": "Overall humans",
        "human_female": "Female participants",
        "human_male": "Male participants",
    }
    rater_label_frame.loc[
        rater_label_frame["rater_id"].isin(human_display_labels), "display_label"
    ] = rater_label_frame["rater_id"].map(human_display_labels)

    rater_order = HUMAN_REFERENCE_ORDER + [
        rater_id
        for rater_id, sort_label in (
            rater_label_frame.set_index("rater_id")["rater_sort_label"].to_dict()
        ).items()
        if sort_label in MODEL_ORDER
    ]
    rater_order = HUMAN_REFERENCE_ORDER + sorted(
        [rater_id for rater_id in rater_order if rater_id not in HUMAN_REFERENCE_ORDER],
        key=lambda rater_id: MODEL_ORDER.index(
            rater_label_frame.set_index("rater_id").loc[rater_id, "rater_sort_label"]
        ),
    )
    label_lookup = rater_label_frame.set_index("rater_id")["display_label"].to_dict()
    kind_lookup = rater_label_frame.set_index("rater_id")["rater_kind"].to_dict()

    rng = np.random.default_rng(BOOTSTRAP_SEED + 31)
    summary_rows: list[dict[str, object]] = []
    for (condition, rater_id), group in relevant_frame.groupby(["condition", "rater_id"]):
        values = group["delta_female_minus_male_norm"].dropna().to_numpy(dtype=float)
        bootstrap_means = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
        for iteration_index in range(BOOTSTRAP_ITERATIONS):
            bootstrap_sample = rng.choice(values, size=values.size, replace=True)
            bootstrap_means[iteration_index] = float(np.mean(bootstrap_sample))
        summary_rows.append(
            {
                "condition": condition,
                "scale_label": SCALE_LABELS[condition],
                "rater_id": rater_id,
                "rater_label": label_lookup[rater_id],
                "rater_kind": kind_lookup[rater_id],
                "mean_delta_norm": float(np.mean(values)),
                "ci_low": float(np.quantile(bootstrap_means, 0.025)),
                "ci_high": float(np.quantile(bootstrap_means, 0.975)),
                "female_higher_items": int((values > 0).sum()),
                "male_higher_items": int((values < 0).sum()),
                "tied_items": int(np.isclose(values, 0.0).sum()),
                "n_items": int(values.size),
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(
        OUTPUT_DIR / "direction_agreement_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    y_labels = [label_lookup[rater_id] for rater_id in rater_order]
    y_positions = np.arange(len(rater_order))
    y_lookup = {rater_id: position for position, rater_id in enumerate(rater_order)}
    color_lookup = {"human": "#1F77B4", "model": "#E45756"}
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 7.2), sharey=True, sharex=True)
    for axis, condition in zip(axes, SCALE_ORDER, strict=True):
        panel_frame = summary_frame.loc[summary_frame["condition"] == condition].copy()
        for _, row in panel_frame.iterrows():
            y_position = y_lookup[row["rater_id"]]
            axis.errorbar(
                row["mean_delta_norm"],
                y_position,
                xerr=[
                    [row["mean_delta_norm"] - row["ci_low"]],
                    [row["ci_high"] - row["mean_delta_norm"]],
                ],
                fmt="o",
                markersize=4.8,
                color=color_lookup[row["rater_kind"]],
                ecolor=color_lookup[row["rater_kind"]],
                elinewidth=1.0,
                capsize=2.4,
                zorder=3,
            )
        axis.axvline(0, color="#333333", linewidth=0.9)
        axis.set_title(SCALE_LABELS[condition], fontsize=11.0)
        axis.set_xlabel(r"Mean $\Delta_{F-M}$ / scale maximum")
        axis.set_xlim(-0.01, 0.17)
        axis.grid(axis="x", color="#E6EAF0")
        axis.grid(axis="y", visible=False)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels)
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1F77B4", label="Humans", markersize=6),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#E45756", label="LLMs", markersize=6),
    ]
    axes[2].legend(handles=handles, frameon=False, loc="lower right")
    figure.suptitle(
        "All human and LLM cells rate female-targeted paired items as more attacking on average",
        fontsize=13.0,
        y=1.02,
    )
    figure.subplots_adjust(wspace=0.08)
    save_figure(figure, "fig_r2_direction_agreement_forest")


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


def make_domain_alignment_errorbar_forest(item_frame: pd.DataFrame) -> None:
    # Project convention: ten-domain comparisons must be shown as forest plots,
    # not heatmaps. This variant follows the original R4 style by adding 95% CIs.
    relevant_frame = item_frame.loc[
        item_frame["condition"].isin(SCALE_ORDER)
        & item_frame["dimension_1"].isin(DOMAIN_ORDER)
    ].copy()

    human_frame = relevant_frame.loc[
        relevant_frame["rater_id"].isin(["human_all", "human_female", "human_male"])
    ].copy()
    human_frame["rater_group"] = human_frame["rater_id"].map(
        {
            "human_all": "Overall humans",
            "human_female": "Female participants",
            "human_male": "Male participants",
        }
    )

    model_item_median = (
        relevant_frame.loc[relevant_frame["rater_kind"] == "model"]
        .groupby(["condition", "item_id", "dimension_1"], as_index=False)
        .agg(delta_female_minus_male_norm=("delta_female_minus_male_norm", "median"))
        .assign(rater_group="LLM median")
    )
    bootstrap_source = pd.concat(
        [
            human_frame[
                ["condition", "item_id", "dimension_1", "delta_female_minus_male_norm", "rater_group"]
            ],
            model_item_median[
                ["condition", "item_id", "dimension_1", "delta_female_minus_male_norm", "rater_group"]
            ],
        ],
        ignore_index=True,
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED + 17)
    summary_rows: list[dict[str, object]] = []
    for (condition, dimension, rater_group), group in bootstrap_source.groupby(
        ["condition", "dimension_1", "rater_group"], sort=False
    ):
        values = group["delta_female_minus_male_norm"].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        bootstrap_means = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
        for iteration_index in range(BOOTSTRAP_ITERATIONS):
            bootstrap_sample = rng.choice(values, size=values.size, replace=True)
            bootstrap_means[iteration_index] = float(np.mean(bootstrap_sample))
        summary_rows.append(
            {
                "condition": condition,
                "dimension_1": dimension,
                "rater_group": rater_group,
                "mean_delta_norm": float(np.mean(values)),
                "ci_low": float(np.quantile(bootstrap_means, 0.025)),
                "ci_high": float(np.quantile(bootstrap_means, 0.975)),
                "n_items": int(values.size),
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(OUTPUT_DIR / "domain_alignment_errorbar_summary.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(13.8, 7.0), sharey=True)
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
    x_limit = max(0.45, float(summary_frame["ci_high"].max()) * 1.12)
    for axis, scale_key in zip(axes, SCALE_ORDER, strict=True):
        panel_frame = summary_frame.loc[summary_frame["condition"] == scale_key].copy()
        for domain in DOMAIN_ORDER:
            axis.axhline(y_positions[domain], color="#e5e7eb", linewidth=0.7, zorder=0)
        axis.axvline(0, color="#6b7280", linestyle="--", linewidth=0.9, zorder=1)
        for rater_group in ["Overall humans", "Female participants", "Male participants", "LLM median"]:
            rater_frame = panel_frame.loc[panel_frame["rater_group"] == rater_group].copy()
            rater_frame["y"] = (
                rater_frame["dimension_1"].map(y_positions) + vertical_offsets[rater_group]
            )
            lower_error = rater_frame["mean_delta_norm"] - rater_frame["ci_low"]
            upper_error = rater_frame["ci_high"] - rater_frame["mean_delta_norm"]
            axis.errorbar(
                rater_frame["mean_delta_norm"],
                rater_frame["y"],
                xerr=np.vstack([lower_error, upper_error]),
                fmt=marker_map[rater_group],
                markersize=5.0,
                color=palette[rater_group],
                ecolor=palette[rater_group],
                elinewidth=1.0,
                capsize=2.2,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=rater_group,
                zorder=3,
            )
        axis.set_title(SCALE_LABELS[scale_key], fontsize=11.5)
        axis.set_xlabel(r"Mean $\Delta_{F-M}$ / scale max")
        axis.set_xlim(-0.08, x_limit)
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
    figure.suptitle(r"Per-domain mean $\Delta_{F-M}$ with 95% CI", fontsize=13.0, y=0.99)
    figure.subplots_adjust(bottom=0.13, top=0.91, wspace=0.18)
    save_figure(figure, "fig_r4_domain_alignment_errorbar_forest")


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


def make_multilevel_similarity_profile(
    dz_stats: pd.DataFrame,
    participant_gender_icc_frame: pd.DataFrame,
    overall_icc_frame: pd.DataFrame,
) -> None:
    human_dz = dz_stats.loc[
        dz_stats["rater_id"].isin(HUMAN_REFERENCE_ORDER),
        ["rater_id", "condition", "cohens_dz"],
    ].rename(columns={"cohens_dz": "human_dz"})
    human_dz["human_reference"] = human_dz["rater_id"].map(HUMAN_REFERENCE_LABELS)

    model_dz = dz_stats.loc[
        dz_stats["rater_kind"] == "model",
        ["rater_label", "condition", "cohens_dz"],
    ].rename(columns={"rater_label": "model", "cohens_dz": "model_dz"})
    model_dz["model"] = model_dz["model"].map(canonical_model_label)
    aggregate_closeness = human_dz.merge(model_dz, on="condition", how="inner")
    aggregate_closeness["scale_label"] = aggregate_closeness["condition"].map(SCALE_LABELS)
    aggregate_closeness["signed_dz_gap"] = aggregate_closeness["model_dz"] - aggregate_closeness["human_dz"]
    aggregate_closeness["abs_dz_gap"] = aggregate_closeness["signed_dz_gap"].abs()
    aggregate_closeness["aggregate_similarity"] = 1 / (1 + aggregate_closeness["abs_dz_gap"])
    aggregate_closeness["model_display"] = aggregate_closeness["model"].map(display_model_label)
    aggregate_closeness.to_csv(
        OUTPUT_DIR / "aggregate_level_human_model_closeness.csv",
        index=False,
        encoding="utf-8-sig",
    )

    human_domain = pd.read_csv(HUMAN_DOMAIN_PATH)
    model_domain = pd.read_csv(MODEL_DOMAIN_PATH)
    model_domain["model"] = model_domain["model_label"].map(canonical_model_label)
    domain_rows: list[dict[str, object]] = []
    for human_id in HUMAN_REFERENCE_ORDER:
        for scale_key in SCALE_ORDER:
            human_vector = (
                human_domain.loc[
                    (human_domain["rater_id"] == human_id)
                    & (human_domain["condition"] == scale_key),
                    ["dimension_1", "dz"],
                ]
                .set_index("dimension_1")
                .reindex(DOMAIN_ORDER)["dz"]
            )
            for model in MODEL_ORDER:
                model_vector = (
                    model_domain.loc[
                        (model_domain["model"] == model)
                        & (model_domain["scale_key"] == scale_key),
                        ["一级-攻击领域", "cohens_dz"],
                    ]
                    .rename(columns={"一级-攻击领域": "dimension_1"})
                    .set_index("dimension_1")
                    .reindex(DOMAIN_ORDER)["cohens_dz"]
                )
                paired_domain = pd.DataFrame({"human": human_vector, "model": model_vector}).dropna()
                domain_rows.append(
                    {
                        "human_reference_id": human_id,
                        "human_reference": HUMAN_REFERENCE_LABELS[human_id],
                        "scale": scale_key,
                        "scale_label": SCALE_LABELS[scale_key],
                        "model": model,
                        "model_display": display_model_label(model),
                        "n_domains": len(paired_domain),
                        "spearman_rho": paired_domain["human"].corr(
                            paired_domain["model"], method="spearman"
                        ),
                        "pearson_r": paired_domain["human"].corr(paired_domain["model"]),
                        "icc_2_1": icc_2_1_two_raters(
                            paired_domain["human"].to_numpy(dtype=float),
                            paired_domain["model"].to_numpy(dtype=float),
                        ),
                    }
                )
    domain_similarity = pd.DataFrame(domain_rows)
    domain_similarity.to_csv(
        OUTPUT_DIR / "domain_level_human_model_similarity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall_icc = overall_icc_frame.copy()
    overall_icc["human_reference"] = "Overall humans"
    overall_icc["human_reference_id"] = "human_all"
    participant_icc = participant_gender_icc_frame.copy()
    participant_icc["human_reference"] = participant_icc["participant_gender_label"].replace(
        {"男性被试": "Male participants", "女性被试": "Female participants"}
    )
    participant_icc["human_reference_id"] = participant_icc["human_reference"].map(
        {
            "Male participants": "human_male",
            "Female participants": "human_female",
        }
    )
    item_similarity = pd.concat(
        [
            overall_icc[["human_reference_id", "human_reference", "scale", "scale_label", "model", "icc_2_1", "spearman_rho"]],
            participant_icc[["human_reference_id", "human_reference", "scale", "scale_label", "model", "icc_2_1", "spearman_rho"]],
        ],
        ignore_index=True,
    )
    item_similarity["scale_label"] = item_similarity["scale"].map(SCALE_LABELS)
    item_similarity["model"] = item_similarity["model"].map(canonical_model_label)
    item_similarity["model_display"] = item_similarity["model"].map(display_model_label)
    item_similarity.to_csv(
        OUTPUT_DIR / "item_level_human_model_similarity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top_aggregate = (
        aggregate_closeness.sort_values("abs_dz_gap")
        .groupby(["human_reference", "condition"], as_index=False)
        .first()
        .rename(columns={"model_display": "aggregate_closest_model"})
    )
    top_domain = (
        domain_similarity.sort_values("spearman_rho", ascending=False)
        .groupby(["human_reference", "scale"], as_index=False)
        .first()
        .rename(
            columns={
                "model_display": "domain_closest_model",
                "spearman_rho": "domain_spearman_rho",
            }
        )
    )
    top_item = (
        item_similarity.sort_values("icc_2_1", ascending=False)
        .groupby(["human_reference", "scale"], as_index=False)
        .first()
        .rename(columns={"model_display": "item_closest_model", "icc_2_1": "item_icc_2_1"})
    )
    top_model_profile = (
        top_aggregate.merge(
            top_domain,
            left_on=["human_reference", "condition"],
            right_on=["human_reference", "scale"],
            how="inner",
            suffixes=("", "_domain"),
        )
        .merge(
            top_item,
            left_on=["human_reference", "condition"],
            right_on=["human_reference", "scale"],
            how="inner",
            suffixes=("", "_item"),
        )
    )
    top_model_profile = top_model_profile[
        [
            "human_reference",
            "scale_label",
            "aggregate_closest_model",
            "abs_dz_gap",
            "domain_closest_model",
            "domain_spearman_rho",
            "item_closest_model",
            "item_icc_2_1",
        ]
    ]
    top_model_profile.to_csv(
        OUTPUT_DIR / "multilevel_top_model_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows: list[dict[str, object]] = []
    for human_reference in HUMAN_REFERENCE_LABELS.values():
        for scale_key in SCALE_ORDER:
            scale_label = SCALE_LABELS[scale_key]
            aggregate_panel = aggregate_closeness.loc[
                (aggregate_closeness["human_reference"] == human_reference)
                & (aggregate_closeness["condition"] == scale_key)
            ]
            domain_panel = domain_similarity.loc[
                (domain_similarity["human_reference"] == human_reference)
                & (domain_similarity["scale"] == scale_key)
            ]
            item_panel = item_similarity.loc[
                (item_similarity["human_reference"] == human_reference)
                & (item_similarity["scale"] == scale_key)
            ]
            summary_rows.extend(
                [
                    {
                        "human_reference": human_reference,
                        "scale": scale_key,
                        "scale_label": scale_label,
                        "analysis_level": "Aggregate |dz gap|",
                        "median_value": aggregate_panel["abs_dz_gap"].median(),
                        "best_value": aggregate_panel["abs_dz_gap"].min(),
                        "best_model": aggregate_panel.sort_values("abs_dz_gap")["model_display"].iloc[0],
                        "direction": "lower is closer",
                    },
                    {
                        "human_reference": human_reference,
                        "scale": scale_key,
                        "scale_label": scale_label,
                        "analysis_level": "Domain Spearman rho",
                        "median_value": domain_panel["spearman_rho"].median(),
                        "best_value": domain_panel["spearman_rho"].max(),
                        "best_model": domain_panel.sort_values("spearman_rho", ascending=False)["model_display"].iloc[0],
                        "direction": "higher is closer",
                    },
                    {
                        "human_reference": human_reference,
                        "scale": scale_key,
                        "scale_label": scale_label,
                        "analysis_level": "Item ICC(2,1)",
                        "median_value": item_panel["icc_2_1"].median(),
                        "best_value": item_panel["icc_2_1"].max(),
                        "best_model": item_panel.sort_values("icc_2_1", ascending=False)["model_display"].iloc[0],
                        "direction": "higher is closer",
                    },
                ]
            )
    multilevel_summary = pd.DataFrame(summary_rows)
    multilevel_summary.to_csv(
        OUTPUT_DIR / "multilevel_similarity_descriptive_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_frame = multilevel_summary.copy()
    plot_frame["cell"] = plot_frame["human_reference"] + "\n" + plot_frame["scale_label"]
    cell_order = [
        f"{reference}\n{scale}"
        for reference in ["Overall humans", "Female participants", "Male participants"]
        for scale in ["3-point", "7-point", "Slider"]
    ]
    plot_frame["cell"] = pd.Categorical(plot_frame["cell"], categories=cell_order, ordered=True)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.4, 6.2),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.1]},
    )
    panel_specs = [
        ("Aggregate |dz gap|", r"Closest aggregate $|d_z|$ gap", "best_value", "#4C78A8", (0.0, 0.30), True),
        ("Domain Spearman rho", r"Best 10-domain Spearman $\rho$", "best_value", "#F58518", (0.0, 1.0), False),
    ]
    for axis, (level, title, x_column, color, x_limits, lower_is_better) in zip(
        axes, panel_specs, strict=True
    ):
        panel_frame = plot_frame.loc[plot_frame["analysis_level"] == level].copy()
        axis.scatter(
            panel_frame[x_column],
            panel_frame["cell"],
            s=62,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for _, row in panel_frame.iterrows():
            text_offset = 0.012
            horizontal_alignment = "left"
            axis.text(
                row[x_column] + text_offset,
                row["cell"],
                row["best_model"],
                va="center",
                ha=horizontal_alignment,
                fontsize=7.4,
            )
        axis.set_title(title, fontsize=10.8)
        axis.set_xlabel("Lower is closer" if lower_is_better else "Higher is closer")
        axis.set_xlim(*x_limits)
        axis.grid(axis="x", color="#E6EAF0")
        axis.grid(axis="y", visible=False)
        axis.tick_params(axis="y", labelsize=9.0)
    axes[0].set_ylabel("")
    figure.suptitle("The most human-like model depends on reference group, scale, and analysis level", fontsize=13.0, y=1.02)
    figure.subplots_adjust(wspace=0.12)
    save_figure(figure, "fig_r6_similarity_by_analysis_level_forest")


def main() -> None:
    configure_plotting()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    item_frame = pd.read_csv(ITEM_DELTA_PATH)
    dz_stats = pd.read_csv(DZ_STATS_PATH)
    participant_gender_icc_frame = pd.read_csv(ICC_PATH)
    overall_icc_frame = pd.read_csv(OVERALL_ICC_PATH)

    make_natural_gender_target_confound_figure()
    make_direction_agreement_forest(item_frame)
    make_reference_group_bootstrap_forest(item_frame)
    make_domain_alignment_forest()
    make_domain_alignment_errorbar_forest(item_frame)
    make_scale_response_trajectory(dz_stats)
    make_aggregate_item_dissociation(
        participant_gender_icc_frame, overall_icc_frame, dz_stats
    )
    make_multilevel_similarity_profile(
        dz_stats, participant_gender_icc_frame, overall_icc_frame
    )
    print(f"Wrote revised paper figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
