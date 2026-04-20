"""Generate the three placeholder figures referenced in the Results manuscript.

Placeholder A:
    Study-1a subgroup F1/Brier stratified by hate-target topic. For Chinese,
    the validated dataset carries a ``topic`` column with values
    {race, region, gender, lgbt}; rows are partitioned into a
    gender-related subset (gender + lgbt) versus a race/region subset plus
    the overall baseline. For English, the merged validated artifact does
    not preserve the HateXplain ``target`` annotation, so this script joins
    HateXplain's ``dataset.json`` (downloaded to
    ``data/external/hatexplain/dataset.json``) on ``post_id`` and partitions
    HateXplain-sourced rows by whether any annotator labelled the post with
    a gender-related target (Women / Men / Homosexual / Heterosexual /
    Bisexual / Asexual). HASOC-sourced rows (whose ``post_id`` starts with
    ``hasoc_`` or consists of a bare numeric id) carry no target-group
    label and therefore contribute only to the "all" baseline.
    F1 uses the score>=1 -> positive rule; Brier uses the p=score/5
    mapping.

Placeholder B:
    Pairwise scatter across the three 1b rating scales (attack_3pt,
    attack_7pt_likert, attack_slider_0_100) showing Cohen's d_z per model.
    Consistency with y=x indicates scale-invariance of the female > male
    effect.

Placeholder C:
    Model x level-1 attack-domain d_z heatmap (primary scale: 7pt Likert),
    annotated with significance markers from the FDR-corrected Wilcoxon q
    values already present in stats_level1.csv.

Run via ``uv run python generate_placeholder_figures.py``. Outputs land next
to the existing artifacts/ and outputs/group_swap_1b/.../ files so that
results_1a_1b.tex can reference them without path changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
B_FIG_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "group_swap_1b"
    / "1b_groupswap_demensionsentence"
    / "analysis_1b"
    / "figures"
)


# Match the model ordering used by data/12_1b_plot.py so panel numbers line up.
MODEL_ORDER = [
    "openai_gpt-5.1",
    "anthropic_claude-opus-4.5",
    "z-ai_glm-4.6",
    "meta-llama_llama-4-maverick",
    "deepseek_deepseek-r1-0528",
    "deepseek_deepseek-v3.2-exp",
    "moonshotai_kimi-k2-thinking",
    "qwen_qwen-2.5-72b-instruct",
    "google_gemma-4-31b-it",
]
MODEL_LABEL_MAP = {
    "openai_gpt-5.1": "GPT-5.1",
    "anthropic_claude-opus-4.5": "Claude Opus 4.5",
    "z-ai_glm-4.6": "GLM 4.6",
    "meta-llama_llama-4-maverick": "Llama Maverick",
    "deepseek_deepseek-r1-0528": "DeepSeek R1",
    "deepseek_deepseek-v3.2-exp": "DeepSeek V3.2",
    "moonshotai_kimi-k2-thinking": "Kimi K2",
    "qwen_qwen-2.5-72b-instruct": "Qwen 2.5 72B",
    "google_gemma-4-31b-it": "Gemma 4 31B",
}

# Column name abbreviations used in the merged 1a artifact. The 1a pipeline
# historically maps ``openai_gpt-5.1`` -> ``chatgpt5.1`` and
# ``anthropic_claude-opus-4.5`` -> ``claude4.5``. This script recovers the
# canonical model_prefix keys used elsewhere for consistent labelling.
ONE_A_COLUMN_ALIAS_MAP = {
    "chatgpt5.1": "openai_gpt-5.1",
    "claude4.5": "anthropic_claude-opus-4.5",
}


SCALE_ORDER = ["attack_3pt", "attack_7pt_likert", "attack_slider_0_100"]
SCALE_LABEL_MAP = {
    "attack_3pt": "3 点",
    "attack_7pt_likert": "7 点 Likert",
    "attack_slider_0_100": "0-100 连续",
}

LEVEL1_ORDER = [
    "性化攻击（性羞辱）",
    "外貌形象攻击",
    "性别角色/性别表达攻击",
    "道德品行攻击",
    "人际关系攻击",
    "情绪稳定攻击",
    "能力才干攻击",
    "智力理性攻击",
    "社会地位攻击",
    "经济资源攻击",
]


# Gender-related HateXplain target labels. Any post whose annotators list
# at least one of these values is routed into the "gender+lgbt" subset; any
# post that has at least one non-gender community target (e.g., African,
# Islam, Jewish) is routed into the "race+religion" subset. A post can land
# in both subsets if different annotators chose different targets; this is
# rare and we tolerate the overlap because the union still cleanly separates
# gender content from non-gender content within the subset it counts into.
GENDER_RELATED_HATEXPLAIN_TARGETS = frozenset(
    {"Women", "Men", "Homosexual", "Heterosexual", "Bisexual", "Asexual"}
)
NON_GENDER_COMMUNITY_TARGETS = frozenset(
    {
        "African",
        "Arab",
        "Asian",
        "Caucasian",
        "Christian",
        "Hindu",
        "Hispanic",
        "Indian",
        "Indigenous",
        "Islam",
        "Jewish",
        "Refugee",
        "Buddhism",
        "Nonreligious",
        "Minority",
        "Disability",
        "Economic",
    }
)
HATEXPLAIN_JSON_PATH = (
    PROJECT_ROOT / "data" / "external" / "hatexplain" / "dataset.json"
)


# ---------------------------------------------------------------------------
# Placeholder A: Study 1a F1/Brier stratified by hate-target topic (zh + en)
# ---------------------------------------------------------------------------


def scores_to_binary_predictions(score_column: pd.Series) -> pd.Series:
    """Map 0-5 discrete LLM scores to binary predictions (>=1 -> positive)."""
    return (score_column.fillna(0) >= 1).astype(int)


def scores_to_probabilities(score_column: pd.Series) -> pd.Series:
    """Map 0-5 discrete LLM scores to probabilities via p = score / 5."""
    return score_column.fillna(0).clip(lower=0, upper=5) / 5.0


def brier_score_from_probabilities(
    predicted_probabilities: pd.Series, binary_labels: pd.Series
) -> float:
    """Compute mean squared error between predicted probabilities and labels."""
    aligned_probabilities = predicted_probabilities.astype(float)
    aligned_labels = binary_labels.astype(int)
    return float(((aligned_probabilities - aligned_labels) ** 2).mean())


def parse_model_setting_from_column(column_name: str) -> tuple[str, str]:
    """Split ``{model_alias}_{setting}_score`` into (model_prefix, setting).

    ``setting`` is always either ``zeroshot`` or ``cot``. The model alias
    uses 1a-specific shorthand (see ``ONE_A_COLUMN_ALIAS_MAP``) that is
    normalised back to the canonical ``MODEL_ORDER`` key.
    """
    assert column_name.endswith("_score"), column_name
    stem = column_name[: -len("_score")]
    if stem.endswith("_zeroshot"):
        model_alias = stem[: -len("_zeroshot")]
        setting = "zeroshot"
    elif stem.endswith("_cot"):
        model_alias = stem[: -len("_cot")]
        setting = "cot"
    else:
        raise ValueError(f"Unrecognised column suffix: {column_name}")
    model_prefix = ONE_A_COLUMN_ALIAS_MAP.get(model_alias, model_alias)
    return model_prefix, setting


def compute_stratified_one_a_metrics(
    validated_frame: pd.DataFrame,
    subset_definitions: dict[str, pd.Series],
    language: str,
    label_column: str = "label/2classes",
) -> pd.DataFrame:
    """Return long-form F1/Brier per (language, model, setting, subgroup)."""
    metric_rows: list[dict] = []
    score_columns = [c for c in validated_frame.columns if c.endswith("_score")]
    for score_column in score_columns:
        model_prefix, setting = parse_model_setting_from_column(score_column)
        model_label = MODEL_LABEL_MAP.get(model_prefix, model_prefix)
        for subgroup_name, subgroup_mask in subset_definitions.items():
            subgroup_frame = validated_frame.loc[subgroup_mask]
            if subgroup_frame.empty:
                continue
            ground_truth = subgroup_frame[label_column].astype(int)
            prediction_series = scores_to_binary_predictions(
                subgroup_frame[score_column]
            )
            probability_series = scores_to_probabilities(
                subgroup_frame[score_column]
            )
            metric_rows.append(
                {
                    "language": language,
                    "subgroup": subgroup_name,
                    "model_prefix": model_prefix,
                    "model_label": model_label,
                    "setting": setting,
                    "n": int(len(subgroup_frame)),
                    "f1": f1_score(ground_truth, prediction_series, zero_division=0),
                    "brier": brier_score_from_probabilities(
                        probability_series, ground_truth
                    ),
                }
            )
    return pd.DataFrame(metric_rows)


def load_hatexplain_post_id_to_target_category(
    hatexplain_json_path: Path = HATEXPLAIN_JSON_PATH,
) -> dict[str, str]:
    """Return a mapping from HateXplain post_id to 'gender+lgbt' / 'race+religion'.

    Rule: if any annotator labelled the post with a gender-related target,
    the post is marked 'gender+lgbt'. Otherwise, if any annotator labelled
    it with a non-gender community target, it is marked 'race+religion'.
    Posts where all annotators chose 'None' / 'Other' are not in the output.
    """
    if not hatexplain_json_path.exists():
        raise FileNotFoundError(
            f"HateXplain dataset.json missing at {hatexplain_json_path}. "
            "Download it first (see module docstring)."
        )
    with hatexplain_json_path.open("r", encoding="utf-8") as handle:
        hatexplain_raw = json.load(handle)

    post_id_to_category: dict[str, str] = {}
    for post_id, entry in hatexplain_raw.items():
        all_targets: set[str] = set()
        for annotator_entry in entry.get("annotators", []):
            for target_label in annotator_entry.get("target", []):
                all_targets.add(target_label)
        gender_overlap = all_targets & GENDER_RELATED_HATEXPLAIN_TARGETS
        non_gender_overlap = all_targets & NON_GENDER_COMMUNITY_TARGETS
        if gender_overlap:
            post_id_to_category[post_id] = "gender+lgbt"
        elif non_gender_overlap:
            post_id_to_category[post_id] = "race+religion"
    return post_id_to_category


def plot_placeholder_a(
    stratified_frame: pd.DataFrame,
    output_pdf_path: Path,
) -> None:
    """Save a 4x2 grid (language x metric rows) x (setting columns)."""
    setting_order = ["zeroshot", "cot"]
    subgroup_order_zh = ["all", "gender+lgbt", "race+region"]
    # For English the non-gender subset is race + religion rather than
    # race + region (China-origin labelling convention) because HateXplain
    # target labels include religion.
    subgroup_order_en = ["all", "gender+lgbt", "race+religion"]
    palette = sns.color_palette("colorblind", n_colors=3)

    row_specs = [
        ("zh", "f1"),
        ("zh", "brier"),
        ("en", "f1"),
        ("en", "brier"),
    ]

    model_order_in_plot = [
        MODEL_LABEL_MAP[model_prefix]
        for model_prefix in MODEL_ORDER
        if MODEL_LABEL_MAP[model_prefix]
        in stratified_frame["model_label"].unique().tolist()
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14.0, 15.5), sharex=True)
    for row_index, (language, metric_name) in enumerate(row_specs):
        subgroup_order = (
            subgroup_order_zh if language == "zh" else subgroup_order_en
        )
        for col_index, setting_name in enumerate(setting_order):
            axis = axes[row_index, col_index]
            plot_frame = stratified_frame[
                (stratified_frame["language"] == language)
                & (stratified_frame["setting"] == setting_name)
            ].copy()
            sns.barplot(
                data=plot_frame,
                x="model_label",
                y=metric_name,
                hue="subgroup",
                order=model_order_in_plot,
                hue_order=subgroup_order,
                palette=palette,
                ax=axis,
            )
            axis.set_xlabel("")
            metric_display_label = "F1" if metric_name == "f1" else "Brier"
            if metric_name == "f1":
                axis.set_ylabel("F1")
                axis.set_ylim(0.0, 1.0)
            else:
                axis.set_ylabel("Brier")
                axis.set_ylim(0.0, 0.5)
            setting_display_label = (
                "Zero-shot" if setting_name == "zeroshot" else "CoT"
            )
            language_display_label = "中文" if language == "zh" else "英文"
            axis.set_title(
                f"{language_display_label} · {metric_display_label} · "
                f"{setting_display_label}"
            )
            for tick_label in axis.get_xticklabels():
                tick_label.set_rotation(35)
                tick_label.set_ha("right")
            # Keep one legend per language so users can see the subset
            # renaming ("race+region" for ZH vs "race+religion" for EN).
            keep_legend = (row_index, col_index) in {(0, 1), (2, 1)}
            if not keep_legend:
                legend_handle = axis.get_legend()
                if legend_handle is not None:
                    legend_handle.remove()
            else:
                axis.legend(title="子集", loc="upper right", fontsize=9)
    fig.suptitle(
        "研究一(a) 按目标群体主题分层的 F1 与 Brier（上：中文；下：英文 HateXplain 子集）",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf_path, bbox_inches="tight")
    fig.savefig(output_pdf_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_zh_subset_definitions(
    validated_zh_frame: pd.DataFrame,
) -> dict[str, pd.Series]:
    topic_series = validated_zh_frame["topic"].fillna("")
    return {
        "all": pd.Series(True, index=validated_zh_frame.index),
        "gender+lgbt": topic_series.isin(["gender", "lgbt"]),
        "race+region": topic_series.isin(["race", "region"]),
    }


def _build_en_subset_definitions(
    validated_en_frame: pd.DataFrame,
    post_id_to_category: dict[str, str],
) -> dict[str, pd.Series]:
    """HASOC-sourced rows (no target label) only contribute to the baseline.

    A row goes into the 'gender+lgbt' or 'race+religion' subset only when
    its post_id can be located in HateXplain's target annotations.
    """
    post_id_series = validated_en_frame["post_id"].astype(str)
    resolved_category_series = post_id_series.map(post_id_to_category)
    return {
        "all": pd.Series(True, index=validated_en_frame.index),
        "gender+lgbt": resolved_category_series == "gender+lgbt",
        "race+religion": resolved_category_series == "race+religion",
    }


def generate_placeholder_a() -> Path:
    """Compute F1/Brier stratified by target topic for both zh and en."""
    validated_zh_frame = pd.read_excel(
        ARTIFACTS_DIR / "merged_zh_models_validated.xlsx"
    )
    validated_en_frame = pd.read_excel(
        ARTIFACTS_DIR / "merged_en_models_validated.xlsx"
    )

    zh_subset_definitions = _build_zh_subset_definitions(validated_zh_frame)
    post_id_to_category = load_hatexplain_post_id_to_target_category()
    en_subset_definitions = _build_en_subset_definitions(
        validated_en_frame, post_id_to_category
    )

    en_gender_count = int(en_subset_definitions["gender+lgbt"].sum())
    en_non_gender_count = int(en_subset_definitions["race+religion"].sum())
    en_total = len(validated_en_frame)
    print(
        f"[A] EN subset sizes: gender+lgbt={en_gender_count}, "
        f"race+religion={en_non_gender_count}, "
        f"unresolved (HASOC or non-targeted)={en_total - en_gender_count - en_non_gender_count}, "
        f"total={en_total}"
    )

    zh_metrics_frame = compute_stratified_one_a_metrics(
        validated_zh_frame, zh_subset_definitions, language="zh"
    )
    en_metrics_frame = compute_stratified_one_a_metrics(
        validated_en_frame, en_subset_definitions, language="en"
    )
    stratified_frame = pd.concat([zh_metrics_frame, en_metrics_frame], ignore_index=True)

    long_csv_path = ARTIFACTS_DIR / "placeholder_a_1a_topic_stratified_metrics.csv"
    stratified_frame.sort_values(
        by=["language", "setting", "model_prefix", "subgroup"]
    ).to_csv(long_csv_path, index=False)

    pdf_path = ARTIFACTS_DIR / "placeholder_a_1a_topic_stratified_f1_brier.pdf"
    plot_placeholder_a(stratified_frame, pdf_path)
    return pdf_path


# ---------------------------------------------------------------------------
# Placeholder B: three-scale d_z consistency scatter
# ---------------------------------------------------------------------------


def load_three_scale_overall_stats() -> pd.DataFrame:
    """Load stats_overall.csv for each rating scale into a long frame."""
    frames: list[pd.DataFrame] = []
    for scale_name in SCALE_ORDER:
        stats_path = B_FIG_ROOT / scale_name / "stats_overall.csv"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Missing stats_overall.csv for scale={scale_name} at {stats_path}"
            )
        stats_frame = pd.read_csv(stats_path)
        stats_frame = stats_frame.loc[
            :, ["model_prefix", "model_label", "cohens_dz"]
        ].copy()
        stats_frame["scale"] = scale_name
        frames.append(stats_frame)
    return pd.concat(frames, ignore_index=True)


def plot_placeholder_b(long_stats_frame: pd.DataFrame, output_pdf_path: Path) -> None:
    """Three pairwise scatters of Cohen's d_z across rating scales."""
    wide_stats_frame = long_stats_frame.pivot(
        index=["model_prefix", "model_label"],
        columns="scale",
        values="cohens_dz",
    ).reset_index()

    scale_axis_pairs = [
        ("attack_7pt_likert", "attack_3pt"),
        ("attack_7pt_likert", "attack_slider_0_100"),
        ("attack_3pt", "attack_slider_0_100"),
    ]
    palette = sns.color_palette("colorblind", n_colors=len(MODEL_ORDER))
    model_to_color = {
        model_prefix: palette[idx] for idx, model_prefix in enumerate(MODEL_ORDER)
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for axis, (x_scale, y_scale) in zip(axes, scale_axis_pairs):
        for _, row in wide_stats_frame.iterrows():
            axis.scatter(
                row[x_scale],
                row[y_scale],
                s=70,
                color=model_to_color[row["model_prefix"]],
                edgecolor="black",
                linewidth=0.5,
                label=row["model_label"],
            )
        combined_values = pd.concat([wide_stats_frame[x_scale], wide_stats_frame[y_scale]])
        axis_min = float(combined_values.min()) - 0.05
        axis_max = float(combined_values.max()) + 0.05
        axis.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            linestyle="--",
            linewidth=1.0,
            color="grey",
        )
        axis.set_xlim(axis_min, axis_max)
        axis.set_ylim(axis_min, axis_max)
        axis.set_xlabel(f"$d_z$（{SCALE_LABEL_MAP[x_scale]}）")
        axis.set_ylabel(f"$d_z$（{SCALE_LABEL_MAP[y_scale]}）")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, linestyle=":", alpha=0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(labels)),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "研究一(b) 三种评分尺度下 $d_z$ 的跨模型一致性（7 点 Likert 为主分析）",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf_path, bbox_inches="tight")
    fig.savefig(output_pdf_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_placeholder_b() -> Path:
    long_stats_frame = load_three_scale_overall_stats()
    pdf_path = B_FIG_ROOT / "placeholder_b_three_scale_dz_scatter.pdf"
    plot_placeholder_b(long_stats_frame, pdf_path)
    return pdf_path


# ---------------------------------------------------------------------------
# Placeholder C: model x level-1 d_z heatmap (7pt primary)
# ---------------------------------------------------------------------------


def _ordered_categorical(
    all_values: Iterable[str], reference_order: list[str]
) -> list[str]:
    """Keep ``reference_order`` items that exist in ``all_values``, in order."""
    values_present = set(all_values)
    ordered = [value for value in reference_order if value in values_present]
    leftovers = sorted(values_present - set(ordered))
    return ordered + leftovers


def plot_placeholder_c(level1_frame: pd.DataFrame, output_pdf_path: Path) -> None:
    """Draw model x level-1 d_z heatmap with significance stars."""
    dz_pivot = level1_frame.pivot(
        index="model_prefix", columns="一级-攻击领域", values="cohens_dz"
    )
    q_pivot = level1_frame.pivot(
        index="model_prefix", columns="一级-攻击领域", values="wilcoxon_p_fdr"
    )

    row_order = _ordered_categorical(dz_pivot.index.tolist(), MODEL_ORDER)
    col_order = _ordered_categorical(dz_pivot.columns.tolist(), LEVEL1_ORDER)
    dz_pivot = dz_pivot.reindex(index=row_order, columns=col_order)
    q_pivot = q_pivot.reindex(index=row_order, columns=col_order)

    annotation_matrix = np.empty_like(dz_pivot.values, dtype=object)
    for i in range(dz_pivot.shape[0]):
        for j in range(dz_pivot.shape[1]):
            dz_value = dz_pivot.values[i, j]
            q_value = q_pivot.values[i, j]
            if pd.isna(dz_value):
                annotation_matrix[i, j] = ""
                continue
            if pd.isna(q_value):
                marker = ""
            elif q_value < 0.001:
                marker = "***"
            elif q_value < 0.01:
                marker = "**"
            elif q_value < 0.05:
                marker = "*"
            else:
                marker = ""
            annotation_matrix[i, j] = f"{dz_value:.2f}{marker}"

    # Symmetric diverging cmap centred at 0 so negative d_z (if present) is
    # visually distinct from positive d_z.
    abs_extreme = float(np.nanmax(np.abs(dz_pivot.values)))
    vmin_value = -abs_extreme
    vmax_value = abs_extreme

    display_row_labels = [MODEL_LABEL_MAP.get(key, key) for key in dz_pivot.index]

    fig, axis = plt.subplots(figsize=(11.0, 6.0))
    sns.heatmap(
        dz_pivot.values,
        annot=annotation_matrix,
        fmt="",
        cmap="RdBu_r",
        center=0.0,
        vmin=vmin_value,
        vmax=vmax_value,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Cohen's $d_z$"},
        xticklabels=list(dz_pivot.columns),
        yticklabels=display_row_labels,
        ax=axis,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_title(
        "研究一(b) 模型 × 一级攻击领域 $d_z$ 热图（7 点 Likert；"
        "$*$ $q<0.05$，$**$ $q<0.01$，$***$ $q<0.001$，BH-FDR 校正）",
        fontsize=12,
    )
    for tick_label in axis.get_xticklabels():
        tick_label.set_rotation(30)
        tick_label.set_ha("right")
    for tick_label in axis.get_yticklabels():
        tick_label.set_rotation(0)
    fig.tight_layout()
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf_path, bbox_inches="tight")
    fig.savefig(output_pdf_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_placeholder_c() -> Path:
    level1_csv = B_FIG_ROOT / "attack_7pt_likert" / "stats_level1.csv"
    level1_frame = pd.read_csv(level1_csv)
    pdf_path = B_FIG_ROOT / "placeholder_c_model_by_level1_dz_heatmap.pdf"
    plot_placeholder_c(level1_frame, pdf_path)
    return pdf_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    sns.set_theme(context="paper", style="whitegrid")
    # Force the matplotlib font stack to a CJK-capable choice so axis labels
    # render correctly regardless of the system default.
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    placeholder_a_path = generate_placeholder_a()
    print(f"[A] wrote {placeholder_a_path}")
    placeholder_b_path = generate_placeholder_b()
    print(f"[B] wrote {placeholder_b_path}")
    placeholder_c_path = generate_placeholder_c()
    print(f"[C] wrote {placeholder_c_path}")


if __name__ == "__main__":
    main()
