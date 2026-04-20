"""Generate the three placeholder figures referenced in the Results manuscript.

Placeholder A:
    Study-1a mean LLM attack score on female-targeted vs male-targeted
    posts. This is the less-rigorous natural-data counterpart to Study 1b
    and is intended to motivate the paired-design in 1b: under natural
    data (i.e., without the 1b paired-sentence control), do LLMs already
    score posts that target women higher than posts that target men?
    For English, HateXplain's ``target`` annotation is used directly:
    a post is female-targeted iff any annotator listed ``Women``, and
    male-targeted iff any annotator listed ``Men``. Posts listed with
    both (or neither) are dropped. For Chinese, the dataset carries no
    male/female target field, so this script heuristically classifies
    posts in the ``gender`` + ``lgbt`` topic subset by presence of
    unambiguous female vs male lexical markers (e.g., 女人/她/妹 vs
    男人/哥/弟); the generic pronoun 他 is excluded because it can refer
    to either sex. Posts with both or neither marker type are dropped.
    The metric plotted is the mean 0--5 LLM attack score per
    (language, model, setting, gender-target) cell.

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


# HateXplain target labels used to identify male- vs female-targeted posts.
# A post is female-targeted iff any annotator listed "Women" as a target,
# and male-targeted iff any annotator listed "Men". Posts listed with both
# (rare) or neither are dropped from the comparison.
FEMALE_TARGET_LABEL = "Women"
MALE_TARGET_LABEL = "Men"
HATEXPLAIN_JSON_PATH = (
    PROJECT_ROOT / "data" / "external" / "hatexplain" / "dataset.json"
)


# Chinese lexical markers used to heuristically assign a gender target to
# posts in the zh `gender`+`lgbt` topic subset. Only unambiguous markers
# are included: the generic pronoun 他 is deliberately excluded because
# in Chinese it can refer to either sex; 她 is kept since it is written
# exclusively to denote female referents. Multi-character markers are
# listed before shorter substrings so substring matches behave as
# expected under simple ``str.contains``.
FEMALE_MARKERS_ZH: tuple[str, ...] = (
    "女人", "女性", "女生", "女士", "女子", "女孩", "女友", "女朋友",
    "妇女", "妇人", "太太", "老婆", "妻子", "媳妇", "姑娘", "少女",
    "美女", "母亲", "妈妈", "姐姐", "妹妹", "阿姨", "娘", "她",
)
MALE_MARKERS_ZH: tuple[str, ...] = (
    "男人", "男性", "男生", "男士", "男子", "男孩", "男友", "男朋友",
    "丈夫", "老公", "父亲", "爸爸", "儿子", "哥哥", "弟弟", "叔叔",
    "舅舅", "帅哥", "小伙",
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


def compute_one_a_mean_scores_by_gender_target(
    validated_frame: pd.DataFrame,
    gender_target_series: pd.Series,
    language: str,
) -> pd.DataFrame:
    """Return long-form mean 0-5 score per (language, model, setting, gender).

    ``gender_target_series`` must be aligned with ``validated_frame``'s index
    and carry either 'female', 'male', or NaN. Rows with NaN are dropped
    before aggregation.
    """
    metric_rows: list[dict] = []
    score_columns = [c for c in validated_frame.columns if c.endswith("_score")]
    gender_assigned_mask = gender_target_series.notna()
    assigned_frame = validated_frame.loc[gender_assigned_mask].copy()
    assigned_frame["gender_target"] = gender_target_series.loc[gender_assigned_mask]
    for score_column in score_columns:
        model_prefix, setting = parse_model_setting_from_column(score_column)
        model_label = MODEL_LABEL_MAP.get(model_prefix, model_prefix)
        for gender_target_value in ("female", "male"):
            subgroup_frame = assigned_frame.loc[
                assigned_frame["gender_target"] == gender_target_value
            ]
            if subgroup_frame.empty:
                continue
            score_series = subgroup_frame[score_column].astype(float)
            metric_rows.append(
                {
                    "language": language,
                    "gender_target": gender_target_value,
                    "model_prefix": model_prefix,
                    "model_label": model_label,
                    "setting": setting,
                    "n": int(score_series.notna().sum()),
                    "mean_score": float(score_series.mean()),
                    "sem_score": float(score_series.sem()),
                }
            )
    return pd.DataFrame(metric_rows)


def load_hatexplain_post_id_to_gender_target(
    hatexplain_json_path: Path = HATEXPLAIN_JSON_PATH,
) -> dict[str, str]:
    """Return a mapping from HateXplain post_id to 'female' / 'male'.

    Rule: if any annotator listed ``Women`` as a target but none listed
    ``Men``, the post is 'female'. Symmetrically for 'male'. Posts where
    annotators listed BOTH ``Women`` and ``Men`` are dropped as ambiguous,
    as are posts where neither label appears.
    """
    if not hatexplain_json_path.exists():
        raise FileNotFoundError(
            f"HateXplain dataset.json missing at {hatexplain_json_path}. "
            "Download it first (see module docstring)."
        )
    with hatexplain_json_path.open("r", encoding="utf-8") as handle:
        hatexplain_raw = json.load(handle)

    post_id_to_gender_target: dict[str, str] = {}
    for post_id, entry in hatexplain_raw.items():
        all_targets: set[str] = set()
        for annotator_entry in entry.get("annotators", []):
            for target_label in annotator_entry.get("target", []):
                all_targets.add(target_label)
        has_female_target = FEMALE_TARGET_LABEL in all_targets
        has_male_target = MALE_TARGET_LABEL in all_targets
        if has_female_target and not has_male_target:
            post_id_to_gender_target[post_id] = "female"
        elif has_male_target and not has_female_target:
            post_id_to_gender_target[post_id] = "male"
    return post_id_to_gender_target


def classify_zh_text_by_gender_markers(text_value: str) -> str | None:
    """Classify a Chinese post as 'female', 'male', or None.

    The heuristic counts unambiguous female vs male lexical markers in the
    text (see ``FEMALE_MARKERS_ZH`` and ``MALE_MARKERS_ZH``). The post is
    labelled by whichever marker class has the strictly higher count; ties
    and zero-match posts return None so they are excluded from the plot.
    The generic pronoun 他 is intentionally NOT in either list because in
    Chinese it can refer to any sex (unlike 她 which is female-only).
    """
    if not isinstance(text_value, str) or not text_value:
        return None
    female_marker_count = sum(text_value.count(marker) for marker in FEMALE_MARKERS_ZH)
    male_marker_count = sum(text_value.count(marker) for marker in MALE_MARKERS_ZH)
    if female_marker_count == 0 and male_marker_count == 0:
        return None
    if female_marker_count == male_marker_count:
        return None
    return "female" if female_marker_count > male_marker_count else "male"


def plot_placeholder_a(
    gender_mean_frame: pd.DataFrame,
    output_pdf_path: Path,
) -> None:
    """Save a 1x2 grid (setting columns) of EN-only mean-score bars.

    x-axis: model. hue: gender target (female vs male). y-axis: mean 0-5
    LLM attack score. The ZH heuristic is not shown because the ZH dataset
    lacks a male/female target field and the lexical-marker heuristic
    produced neither a consistent nor interpretable signal. For EN the
    HateXplain target annotation is used directly; the expected role of
    this panel is to show that natural-data female- vs male-targeted
    samples are confounded (sample composition differs) so the natural
    comparison cannot by itself reveal the gender-of-target bias — which
    is what motivates the paired-sentence design in Study 1b.
    """
    english_frame = gender_mean_frame[gender_mean_frame["language"] == "en"].copy()
    setting_order = ["zeroshot", "cot"]
    gender_target_order = ["female", "male"]
    gender_target_display = {"female": "针对女性", "male": "针对男性"}
    palette = {
        "female": sns.color_palette("colorblind")[3],  # reddish
        "male": sns.color_palette("colorblind")[0],  # bluish
    }

    available_model_labels = english_frame["model_label"].unique().tolist()
    model_order_in_plot = [
        MODEL_LABEL_MAP[model_prefix]
        for model_prefix in MODEL_ORDER
        if MODEL_LABEL_MAP[model_prefix] in available_model_labels
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), sharex=True, sharey=True)
    for col_index, setting_name in enumerate(setting_order):
        axis = axes[col_index]
        plot_frame = english_frame[english_frame["setting"] == setting_name].copy()
        sns.barplot(
            data=plot_frame,
            x="model_label",
            y="mean_score",
            hue="gender_target",
            order=model_order_in_plot,
            hue_order=gender_target_order,
            palette=palette,
            ax=axis,
            errorbar=None,
        )
        axis.set_xlabel("")
        axis.set_ylabel("平均 LLM 攻击打分 (0-5)")
        axis.set_ylim(0.0, 5.0)
        setting_display_label = (
            "Zero-shot" if setting_name == "zeroshot" else "CoT"
        )
        sample_sizes_frame = (
            plot_frame.groupby("gender_target")["n"].first().to_dict()
        )
        female_n_text = sample_sizes_frame.get("female", 0)
        male_n_text = sample_sizes_frame.get("male", 0)
        axis.set_title(
            f"英文 HateXplain · {setting_display_label}"
            f"（女={female_n_text}，男={male_n_text}）"
        )
        for tick_label in axis.get_xticklabels():
            tick_label.set_rotation(35)
            tick_label.set_ha("right")
        legend_handle = axis.get_legend()
        if legend_handle is not None:
            if col_index == 1:
                handles_list, _ = axis.get_legend_handles_labels()
                axis.legend(
                    handles_list,
                    [gender_target_display[g] for g in gender_target_order],
                    title="目标群体",
                    loc="upper right",
                    fontsize=9,
                )
            else:
                legend_handle.remove()
    fig.suptitle(
        "研究一(a) 自然数据（HateXplain）下 LLM 对针对女性 vs 男性帖子的平均攻击打分"
        "——样本构成未受控，偏差方向被混淆",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf_path, bbox_inches="tight")
    fig.savefig(output_pdf_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_zh_gender_target_series(
    validated_zh_frame: pd.DataFrame,
) -> pd.Series:
    """Assign 'female' / 'male' / NaN to each zh row via lexical markers.

    The classifier is only applied to rows whose ``topic`` is in
    {gender, lgbt} — the subsets most likely to contain gender-specific
    targets. Rows outside those topics are returned as NaN so they are
    excluded from the plot.
    """
    topic_series = validated_zh_frame["topic"].fillna("")
    eligible_mask = topic_series.isin(["gender", "lgbt"])
    text_column = _pick_zh_text_column(validated_zh_frame)
    gender_target_values = [None] * len(validated_zh_frame)
    for row_index, is_eligible in enumerate(eligible_mask.tolist()):
        if not is_eligible:
            continue
        text_value = validated_zh_frame.iloc[row_index][text_column]
        gender_target_values[row_index] = classify_zh_text_by_gender_markers(
            text_value
        )
    return pd.Series(
        gender_target_values, index=validated_zh_frame.index, dtype=object
    )


def _pick_zh_text_column(validated_zh_frame: pd.DataFrame) -> str:
    """Return the most plausible Chinese post-text column in the zh frame."""
    preferred_candidates = [
        "text", "post", "content", "sentence", "句子", "文本", "内容",
    ]
    for candidate in preferred_candidates:
        if candidate in validated_zh_frame.columns:
            return candidate
    # Fallback: pick the first object-dtype column that is not a known
    # metadata column so we never silently misclassify.
    metadata_columns = {
        "sample_index", "post_id", "label", "label/2classes", "topic",
        "仇恨类型（PRFN/HATE/OFFN）", "有无明确对象（TIN/UNT）",
    }
    for column_name in validated_zh_frame.columns:
        if column_name in metadata_columns:
            continue
        if validated_zh_frame[column_name].dtype == object:
            return column_name
    raise KeyError(
        "Could not locate a Chinese post-text column in merged_zh_models_validated.xlsx"
    )


def _build_en_gender_target_series(
    validated_en_frame: pd.DataFrame,
    post_id_to_gender_target: dict[str, str],
) -> pd.Series:
    """Assign 'female' / 'male' / NaN to each en row from HateXplain targets."""
    post_id_series = validated_en_frame["post_id"].astype(str)
    return post_id_series.map(post_id_to_gender_target)


def generate_placeholder_a() -> Path:
    """Compute mean LLM scores on female- vs male-targeted posts per model."""
    validated_zh_frame = pd.read_excel(
        ARTIFACTS_DIR / "merged_zh_models_validated.xlsx"
    )
    validated_en_frame = pd.read_excel(
        ARTIFACTS_DIR / "merged_en_models_validated.xlsx"
    )

    zh_gender_target_series = _build_zh_gender_target_series(validated_zh_frame)
    post_id_to_gender_target = load_hatexplain_post_id_to_gender_target()
    en_gender_target_series = _build_en_gender_target_series(
        validated_en_frame, post_id_to_gender_target
    )

    zh_female_count = int((zh_gender_target_series == "female").sum())
    zh_male_count = int((zh_gender_target_series == "male").sum())
    en_female_count = int((en_gender_target_series == "female").sum())
    en_male_count = int((en_gender_target_series == "male").sum())
    print(
        f"[A] ZH gender-target coverage: female={zh_female_count}, "
        f"male={zh_male_count}, unassigned="
        f"{len(validated_zh_frame) - zh_female_count - zh_male_count}, "
        f"total={len(validated_zh_frame)}"
    )
    print(
        f"[A] EN gender-target coverage: female={en_female_count}, "
        f"male={en_male_count}, unassigned="
        f"{len(validated_en_frame) - en_female_count - en_male_count}, "
        f"total={len(validated_en_frame)}"
    )

    zh_metrics_frame = compute_one_a_mean_scores_by_gender_target(
        validated_zh_frame, zh_gender_target_series, language="zh"
    )
    en_metrics_frame = compute_one_a_mean_scores_by_gender_target(
        validated_en_frame, en_gender_target_series, language="en"
    )
    gender_mean_frame = pd.concat(
        [zh_metrics_frame, en_metrics_frame], ignore_index=True
    )

    long_csv_path = (
        ARTIFACTS_DIR / "placeholder_a_1a_gender_target_mean_score.csv"
    )
    gender_mean_frame.sort_values(
        by=["language", "setting", "model_prefix", "gender_target"]
    ).to_csv(long_csv_path, index=False)

    pdf_path = ARTIFACTS_DIR / "placeholder_a_1a_gender_target_mean_score.pdf"
    plot_placeholder_a(gender_mean_frame, pdf_path)
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
