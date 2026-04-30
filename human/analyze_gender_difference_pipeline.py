from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


RNG = np.random.default_rng(20260411)
CONDITION_ORDER: list[str] = [
    "attack_3pt",
    "attack_7pt_likert",
    "attack_slider_0_100",
]
CONDITION_LABEL_MAP = {
    "attack_3pt": "3点评分",
    "attack_7pt_likert": "7点评分",
    "attack_slider_0_100": "滑动条评分",
}
CONDITION_SCALE_LIMITS = {
    "attack_3pt": (0.0, 2.0),
    "attack_7pt_likert": (0.0, 6.0),
    "attack_slider_0_100": (0.0, 100.0),
}
VERSION_ORDER = ["男人版", "女人版"]
VERSION_LABEL_MAP = {"男人版": "男版", "女人版": "女版"}
VERSION_COLOR_MAP = {
    "男人版": "#4C607A",
    "女人版": "#B26A6A",
}
DIRECTION_ORDER = ["女人版更高", "两者相同", "男人版更高"]
DIRECTION_LABEL_MAP = {
    "女人版更高": "女版更高",
    "两者相同": "两者相同",
    "男人版更高": "男版更高",
}
DIRECTION_COLOR_MAP = {
    "女人版更高": "#B26A6A",
    "两者相同": "#B7B0A7",
    "男人版更高": "#4C607A",
}
DIMENSION_LEVEL_MAP: list[tuple[str, str, str]] = [
    ("一级维度", "dimension_1", "一级-攻击领域"),
    ("二级维度", "dimension_2", "二级-攻击属性"),
    ("三级维度", "dimension_3", "三级-攻击表达策略"),
]
JOURNAL_COLORS = {
    "text": "#222222",
    "muted_text": "#6B655E",
    "grid": "#DDD8CF",
    "spine": "#CFC8BE",
    "background": "#FCFBF9",
    "connector": "#B9B1A8",
    "neutral": "#7E7A75",
    "accent_gold": "#C8A96B",
    "black": "#1E1E1E",
    "soft_white": "#FEFDFC",
}
REQUIRED_COLUMNS: Sequence[str] = (
    "condition",
    "template_id",
    "shown_version",
    "shown_text",
    "dimension_1",
    "dimension_2",
    "dimension_3",
    "response_value",
    "is_repeat_trial",
)
ANALYSIS_WORKBOOK_NAME = "human_gender_difference_analysis.xlsx"
PARTICIPANT_GENDER_CHOICES = ("男", "女")
PARTICIPANT_GENDER_LABEL_MAP = {
    "男": "男性被试",
    "女": "女性被试",
}
PARTICIPANT_GENDER_OUTPUT_SLUG_MAP = {
    "男": "male_raters",
    "女": "female_raters",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze human aggression ratings by comparing female-vs-male sentence versions "
            "within each of the three rating conditions."
        )
    )
    parser.add_argument(
        "--input-long-path",
        type=Path,
        default=Path("outputs/final_clean_long.csv"),
        help="Path to the cleaned long-format human rating file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for analysis tables and figures. Defaults to a sibling folder named "
            "<input_stem>_gender_difference_analysis."
        ),
    )
    parser.add_argument(
        "--include-repeat-trials",
        action="store_true",
        help="Include repeat trials in the analysis. By default, repeat trials are excluded.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=3000,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--top-level1",
        type=int,
        default=10,
        help="Maximum number of level-1 dimensions to show per forest-plot panel.",
    )
    parser.add_argument(
        "--gender",
        choices=PARTICIPANT_GENDER_CHOICES,
        default=None,
        help="Restrict the analysis to one participant gender subgroup.",
    )
    return parser.parse_args()


def condition_label(condition_name: str) -> str:
    return CONDITION_LABEL_MAP.get(condition_name, condition_name)


def version_label(version_name: str) -> str:
    return VERSION_LABEL_MAP.get(version_name, version_name)


def direction_label(direction_name: str) -> str:
    return DIRECTION_LABEL_MAP.get(direction_name, direction_name)


def participant_gender_label(participant_gender: str | None) -> str:
    if participant_gender is None:
        return "全部被试"
    return PARTICIPANT_GENDER_LABEL_MAP.get(participant_gender, participant_gender)


def build_default_output_dir(input_long_path: Path, participant_gender: str | None) -> Path:
    if participant_gender is None:
        return input_long_path.parent / f"{input_long_path.stem}_gender_difference_analysis"
    output_slug = PARTICIPANT_GENDER_OUTPUT_SLUG_MAP.get(participant_gender, participant_gender)
    return input_long_path.parent / f"{input_long_path.stem}_{output_slug}_gender_difference_analysis"


def ensure_required_columns(dataframe: pd.DataFrame, required_columns: Iterable[str], context: str) -> None:
    missing_columns = [column_name for column_name in required_columns if column_name not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {context}: {missing_columns}")


def normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text_value = str(value).strip()
    if not text_value or text_value.lower() in {"nan", "none"}:
        return None
    return text_value


def load_human_long_frame(input_path: Path, include_repeat_trials: bool, participant_gender: str | None = None) -> pd.DataFrame:
    dataframe = pd.read_csv(input_path, encoding="utf-8-sig", dtype=str, low_memory=False)
    ensure_required_columns(dataframe, REQUIRED_COLUMNS, str(input_path))
    if participant_gender is not None and "gender" not in dataframe.columns:
        raise ValueError(f"Missing required column for gender filtering in {input_path}: ['gender']")

    if not include_repeat_trials:
        dataframe = dataframe.loc[dataframe["is_repeat_trial"].astype(str).str.upper() == "FALSE"].copy()

    dataframe["response_value_num"] = pd.to_numeric(dataframe["response_value"], errors="coerce")
    dataframe = dataframe.loc[dataframe["response_value_num"].notna()].copy()
    dataframe = dataframe.loc[dataframe["shown_version"].isin(VERSION_ORDER)].copy()

    for text_column in ["template_id", "shown_text", "dimension_1", "dimension_2", "dimension_3", "shown_version", "condition"]:
        dataframe[text_column] = dataframe[text_column].map(normalize_text)
    if "gender" in dataframe.columns:
        dataframe["gender"] = dataframe["gender"].map(normalize_text)

    if participant_gender is not None:
        dataframe = dataframe.loc[dataframe["gender"] == participant_gender].copy()

    dataframe = dataframe.dropna(subset=["condition", "template_id", "shown_version", "shown_text", "dimension_1"])
    return dataframe


def aggregate_template_scores(long_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated human ratings into one mean score per template/version/condition."""
    grouped = (
        long_frame.groupby(["condition", "template_id", "shown_version"], dropna=False)
        .agg(
            n_ratings=("response_value_num", "size"),
            mean_score=("response_value_num", "mean"),
            median_score=("response_value_num", "median"),
            sd_score=("response_value_num", "std"),
            min_score=("response_value_num", "min"),
            max_score=("response_value_num", "max"),
            shown_text=("shown_text", lambda values: next((value for value in values if pd.notna(value)), None)),
            dimension_1=("dimension_1", lambda values: next((value for value in values if pd.notna(value)), None)),
            dimension_2=("dimension_2", lambda values: next((value for value in values if pd.notna(value)), None)),
            dimension_3=("dimension_3", lambda values: next((value for value in values if pd.notna(value)), None)),
        )
        .reset_index()
    )
    grouped["sd_score"] = grouped["sd_score"].fillna(0.0)
    return grouped


def build_same_sentence_difference_frame(template_score_frame: pd.DataFrame) -> pd.DataFrame:
    """Pair male and female versions of the same template inside each rating condition."""
    male_frame = (
        template_score_frame.loc[template_score_frame["shown_version"] == "男人版"]
        .rename(
            columns={
                "shown_text": "男人句子",
                "n_ratings": "男人版评分人数",
                "mean_score": "男人版攻击性评分",
                "median_score": "男人版评分中位数",
                "sd_score": "男人版评分标准差",
                "min_score": "男人版评分最小值",
                "max_score": "男人版评分最大值",
            }
        )
        .drop(columns=["shown_version"])
    )
    female_frame = (
        template_score_frame.loc[template_score_frame["shown_version"] == "女人版"]
        .rename(
            columns={
                "shown_text": "女人句子",
                "n_ratings": "女人版评分人数",
                "mean_score": "女人版攻击性评分",
                "median_score": "女人版评分中位数",
                "sd_score": "女人版评分标准差",
                "min_score": "女人版评分最小值",
                "max_score": "女人版评分最大值",
            }
        )
        .drop(columns=["shown_version"])
    )

    merged = male_frame.merge(
        female_frame,
        on=["condition", "template_id", "dimension_1", "dimension_2", "dimension_3"],
        how="inner",
        validate="one_to_one",
    )
    merged["条件标签"] = merged["condition"].map(condition_label)
    merged["女减男分差"] = merged["女人版攻击性评分"] - merged["男人版攻击性评分"]
    merged["绝对分差"] = merged["女减男分差"].abs()
    merged["分差方向"] = np.select(
        [merged["女减男分差"] > 0, merged["女减男分差"] < 0],
        ["女人版更高", "男人版更高"],
        default="两者相同",
    )

    ordered_columns = [
        "condition",
        "条件标签",
        "template_id",
        "dimension_1",
        "dimension_2",
        "dimension_3",
        "男人句子",
        "女人句子",
        "男人版评分人数",
        "女人版评分人数",
        "男人版攻击性评分",
        "女人版攻击性评分",
        "男人版评分中位数",
        "女人版评分中位数",
        "男人版评分标准差",
        "女人版评分标准差",
        "男人版评分最小值",
        "男人版评分最大值",
        "女人版评分最小值",
        "女人版评分最大值",
        "女减男分差",
        "绝对分差",
        "分差方向",
    ]
    return merged.loc[:, ordered_columns].sort_values(["condition", "绝对分差", "template_id"], ascending=[True, False, True]).reset_index(drop=True)


def build_gender_score_long_frame(difference_frame: pd.DataFrame) -> pd.DataFrame:
    male_frame = difference_frame[
        [
            "condition",
            "条件标签",
            "template_id",
            "dimension_1",
            "dimension_2",
            "dimension_3",
            "男人句子",
            "男人版评分人数",
            "男人版攻击性评分",
            "男人版评分中位数",
            "男人版评分标准差",
        ]
    ].copy()
    male_frame.columns = [
        "condition",
        "条件标签",
        "template_id",
        "dimension_1",
        "dimension_2",
        "dimension_3",
        "句子",
        "评分人数",
        "攻击性评分",
        "评分中位数",
        "评分标准差",
    ]
    male_frame["版本"] = "男人版"

    female_frame = difference_frame[
        [
            "condition",
            "条件标签",
            "template_id",
            "dimension_1",
            "dimension_2",
            "dimension_3",
            "女人句子",
            "女人版评分人数",
            "女人版攻击性评分",
            "女人版评分中位数",
            "女人版评分标准差",
        ]
    ].copy()
    female_frame.columns = [
        "condition",
        "条件标签",
        "template_id",
        "dimension_1",
        "dimension_2",
        "dimension_3",
        "句子",
        "评分人数",
        "攻击性评分",
        "评分中位数",
        "评分标准差",
    ]
    female_frame["版本"] = "女人版"

    long_frame = pd.concat([male_frame, female_frame], ignore_index=True)
    return long_frame[[
        "condition",
        "条件标签",
        "template_id",
        "版本",
        "句子",
        "dimension_1",
        "dimension_2",
        "dimension_3",
        "评分人数",
        "攻击性评分",
        "评分中位数",
        "评分标准差",
    ]].sort_values(["condition", "版本", "template_id"]).reset_index(drop=True)


def add_within_group_rank(dataframe: pd.DataFrame, grouping_columns: Sequence[str]) -> pd.DataFrame:
    ranked_frames: list[pd.DataFrame] = []
    for _, group in dataframe.groupby(list(grouping_columns), sort=False):
        ranked_group = group.copy().reset_index(drop=True)
        ranked_group["主题排名"] = np.arange(1, len(ranked_group) + 1)
        ranked_frames.append(ranked_group)
    if not ranked_frames:
        return dataframe.copy()
    return pd.concat(ranked_frames, ignore_index=True)


def build_topic_ranking_frame(gender_score_long_frame: pd.DataFrame) -> pd.DataFrame:
    ranking_frames: list[pd.DataFrame] = []
    for level_name, source_column, output_column in DIMENSION_LEVEL_MAP:
        ranking_frame = (
            gender_score_long_frame.groupby(["condition", "条件标签", "版本", source_column], dropna=False)
            .agg(
                句子数量=("template_id", "size"),
                攻击指数均值=("攻击性评分", "mean"),
                攻击指数中位数=("攻击性评分", "median"),
                攻击指数标准差=("攻击性评分", "std"),
                平均评分人数=("评分人数", "mean"),
            )
            .reset_index()
            .rename(columns={source_column: "主题"})
        )
        ranking_frame["维度层级"] = level_name
        ranking_frame["攻击指数标准差"] = ranking_frame["攻击指数标准差"].fillna(0.0)
        ranking_frame = ranking_frame.sort_values(
            ["condition", "版本", "攻击指数均值", "句子数量", "主题"],
            ascending=[True, True, False, False, True],
        )
        ranking_frames.append(add_within_group_rank(ranking_frame, ["condition", "版本", "维度层级"]))

    return pd.concat(ranking_frames, ignore_index=True)


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    raw = np.asarray(list(p_values), dtype=float)
    if raw.size == 0:
        return raw
    order = np.argsort(raw)
    ranks = np.arange(1, raw.size + 1)
    adjusted = np.empty_like(raw)
    adjusted_sorted = raw[order] * raw.size / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted[order] = adjusted_sorted
    return adjusted


def bootstrap_mean_ci(values: np.ndarray, iterations: int) -> tuple[float, float]:
    if values.size == 0:
        return (math.nan, math.nan)
    bootstrap_means = [float(np.mean(RNG.choice(values, size=values.size, replace=True))) for _ in range(iterations)]
    return tuple(np.percentile(bootstrap_means, [2.5, 97.5]))


def cohens_dz(values: np.ndarray) -> float:
    if values.size < 2:
        return math.nan
    standard_deviation = np.std(values, ddof=1)
    if np.isclose(standard_deviation, 0.0):
        return 0.0
    return float(np.mean(values) / standard_deviation)


def compute_overall_statistics(difference_frame: pd.DataFrame, bootstrap_iterations: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition_name, group in difference_frame.groupby("condition", sort=False):
        diff_values = group["女减男分差"].to_numpy(dtype=float)
        t_result = stats.ttest_1samp(diff_values, popmean=0.0)
        wilcoxon_result = stats.wilcoxon(diff_values, zero_method="wilcox", alternative="two-sided")
        nonzero_values = diff_values[diff_values != 0]
        sign_test_p = (
            stats.binomtest(int(np.sum(nonzero_values > 0)), n=len(nonzero_values), p=0.5).pvalue
            if len(nonzero_values)
            else math.nan
        )
        ci_low, ci_high = bootstrap_mean_ci(diff_values, bootstrap_iterations)
        rows.append(
            {
                "condition": condition_name,
                "条件标签": condition_label(condition_name),
                "n_templates": len(group),
                "mean_male": group["男人版攻击性评分"].mean(),
                "mean_female": group["女人版攻击性评分"].mean(),
                "mean_diff_female_minus_male": float(np.mean(diff_values)),
                "median_diff_female_minus_male": float(np.median(diff_values)),
                "mean_abs_diff": float(np.mean(np.abs(diff_values))),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "t_stat": float(t_result.statistic),
                "t_p": float(t_result.pvalue),
                "wilcoxon_stat": float(wilcoxon_result.statistic),
                "wilcoxon_p": float(wilcoxon_result.pvalue),
                "cohens_dz": cohens_dz(diff_values),
                "female_higher": int(np.sum(diff_values > 0)),
                "male_higher": int(np.sum(diff_values < 0)),
                "ties": int(np.sum(diff_values == 0)),
                "sign_test_p": sign_test_p,
            }
        )
    overall_frame = pd.DataFrame(rows)
    overall_frame["t_p_fdr"] = benjamini_hochberg(overall_frame["t_p"])
    overall_frame["wilcoxon_p_fdr"] = benjamini_hochberg(overall_frame["wilcoxon_p"])
    overall_frame["sign_test_p_fdr"] = benjamini_hochberg(overall_frame["sign_test_p"].fillna(1.0))
    return overall_frame.sort_values("condition", key=lambda series: series.map({name: index for index, name in enumerate(CONDITION_ORDER)})).reset_index(drop=True)


def compute_direction_statistics(difference_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition_name, group in difference_frame.groupby("condition", sort=False):
        counts = group["分差方向"].value_counts()
        female_higher = int(counts.get("女人版更高", 0))
        male_higher = int(counts.get("男人版更高", 0))
        ties = int(counts.get("两者相同", 0))
        discordant = female_higher + male_higher
        sign_test_p = stats.binomtest(female_higher, n=discordant, p=0.5).pvalue if discordant else math.nan
        rows.append(
            {
                "condition": condition_name,
                "条件标签": condition_label(condition_name),
                "female_higher": female_higher,
                "ties": ties,
                "male_higher": male_higher,
                "discordant": discordant,
                "female_higher_pct": female_higher / len(group),
                "ties_pct": ties / len(group),
                "male_higher_pct": male_higher / len(group),
                "sign_test_p": sign_test_p,
            }
        )
    direction_frame = pd.DataFrame(rows)
    direction_frame["sign_test_p_fdr"] = benjamini_hochberg(direction_frame["sign_test_p"].fillna(1.0))
    return direction_frame.sort_values("condition", key=lambda series: series.map({name: index for index, name in enumerate(CONDITION_ORDER)})).reset_index(drop=True)


def compute_level1_statistics(difference_frame: pd.DataFrame, bootstrap_iterations: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (condition_name, level1_name), group in difference_frame.groupby(["condition", "dimension_1"], sort=False):
        diff_values = group["女减男分差"].to_numpy(dtype=float)
        if len(diff_values) < 2:
            continue
        try:
            t_result = stats.ttest_1samp(diff_values, popmean=0.0)
            wilcoxon_result = stats.wilcoxon(diff_values, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            continue
        ci_low, ci_high = bootstrap_mean_ci(diff_values, bootstrap_iterations)
        rows.append(
            {
                "condition": condition_name,
                "条件标签": condition_label(condition_name),
                "一级-攻击领域": level1_name,
                "n": len(group),
                "mean_diff_female_minus_male": float(np.mean(diff_values)),
                "median_diff_female_minus_male": float(np.median(diff_values)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "t_p": float(t_result.pvalue),
                "wilcoxon_p": float(wilcoxon_result.pvalue),
                "cohens_dz": cohens_dz(diff_values),
                "female_higher": int(np.sum(diff_values > 0)),
                "male_higher": int(np.sum(diff_values < 0)),
                "ties": int(np.sum(diff_values == 0)),
                "abs_mean_diff": float(np.mean(np.abs(diff_values))),
            }
        )
    level1_frame = pd.DataFrame(rows)
    if level1_frame.empty:
        return level1_frame
    level1_frame["t_p_fdr"] = level1_frame.groupby("condition")["t_p"].transform(benjamini_hochberg)
    level1_frame["wilcoxon_p_fdr"] = level1_frame.groupby("condition")["wilcoxon_p"].transform(benjamini_hochberg)
    condition_order_map = {name: index for index, name in enumerate(CONDITION_ORDER)}
    level1_frame["condition_order"] = level1_frame["condition"].map(condition_order_map)
    level1_frame = level1_frame.sort_values(
        ["condition_order", "abs_mean_diff", "一级-攻击领域"],
        ascending=[True, False, True],
    ).drop(columns=["condition_order"])
    return level1_frame.reset_index(drop=True)


def build_h2a4_item_cell_frame(long_frame: pd.DataFrame) -> pd.DataFrame:
    """Build the item-level 2x3 participant-gender x scale matrix for H2a.4."""
    analysis_frame = long_frame.loc[long_frame["gender"].isin(PARTICIPANT_GENDER_CHOICES)].copy()
    grouped = (
        analysis_frame.groupby(["condition", "gender", "template_id", "shown_version"], dropna=False)
        .agg(
            n_ratings=("response_value_num", "size"),
            mean_score=("response_value_num", "mean"),
            dimension_1=("dimension_1", lambda values: next((value for value in values if pd.notna(value)), None)),
            dimension_2=("dimension_2", lambda values: next((value for value in values if pd.notna(value)), None)),
            dimension_3=("dimension_3", lambda values: next((value for value in values if pd.notna(value)), None)),
        )
        .reset_index()
    )

    male_version_frame = (
        grouped.loc[grouped["shown_version"] == "男人版"]
        .rename(columns={"mean_score": "mean_male_version", "n_ratings": "n_male_version_ratings"})
        .drop(columns=["shown_version"])
    )
    female_version_frame = (
        grouped.loc[grouped["shown_version"] == "女人版"]
        .rename(columns={"mean_score": "mean_female_version", "n_ratings": "n_female_version_ratings"})
        .drop(columns=["shown_version"])
    )
    cell_frame = male_version_frame.merge(
        female_version_frame,
        on=["condition", "gender", "template_id", "dimension_1", "dimension_2", "dimension_3"],
        how="inner",
        validate="one_to_one",
    )
    cell_frame["participant_gender"] = cell_frame["gender"]
    cell_frame["participant_gender_label"] = cell_frame["participant_gender"].map(participant_gender_label)
    cell_frame["条件标签"] = cell_frame["condition"].map(condition_label)
    cell_frame["scale_max"] = cell_frame["condition"].map(lambda condition_name: CONDITION_SCALE_LIMITS[condition_name][1])
    cell_frame["delta_female_minus_male"] = (
        cell_frame["mean_female_version"] - cell_frame["mean_male_version"]
    )
    cell_frame["normalized_delta_female_minus_male"] = (
        cell_frame["delta_female_minus_male"] / cell_frame["scale_max"]
    )
    return cell_frame[
        [
            "condition",
            "条件标签",
            "participant_gender",
            "participant_gender_label",
            "template_id",
            "dimension_1",
            "dimension_2",
            "dimension_3",
            "n_male_version_ratings",
            "n_female_version_ratings",
            "mean_male_version",
            "mean_female_version",
            "delta_female_minus_male",
            "scale_max",
            "normalized_delta_female_minus_male",
        ]
    ].sort_values(["participant_gender", "condition", "template_id"]).reset_index(drop=True)


def compute_h2a4_cell_descriptives(h2a4_cell_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (participant_gender, condition_name), group in h2a4_cell_frame.groupby(["participant_gender", "condition"], sort=False):
        normalized_values = group["normalized_delta_female_minus_male"].to_numpy(dtype=float)
        rows.append(
            {
                "participant_gender": participant_gender,
                "participant_gender_label": participant_gender_label(participant_gender),
                "condition": condition_name,
                "条件标签": condition_label(condition_name),
                "n_templates": len(group),
                "mean_normalized_delta": float(np.mean(normalized_values)),
                "median_normalized_delta": float(np.median(normalized_values)),
                "sd_normalized_delta": float(np.std(normalized_values, ddof=1)),
                "cohens_dz": cohens_dz(normalized_values),
                "female_version_higher": int(np.sum(normalized_values > 0)),
                "male_version_higher": int(np.sum(normalized_values < 0)),
                "ties": int(np.sum(normalized_values == 0)),
            }
        )
    order_map = {condition_name: index for index, condition_name in enumerate(CONDITION_ORDER)}
    descriptive_frame = pd.DataFrame(rows)
    descriptive_frame["condition_order"] = descriptive_frame["condition"].map(order_map)
    descriptive_frame["participant_gender_order"] = descriptive_frame["participant_gender"].map({"男": 0, "女": 1})
    return descriptive_frame.sort_values(["participant_gender_order", "condition_order"]).drop(
        columns=["participant_gender_order", "condition_order"]
    ).reset_index(drop=True)


def build_complete_h2a4_wide_frame(h2a4_cell_frame: pd.DataFrame) -> pd.DataFrame:
    wide_frame = h2a4_cell_frame.pivot_table(
        index="template_id",
        columns=["participant_gender", "condition"],
        values="normalized_delta_female_minus_male",
        aggfunc="mean",
    )
    expected_columns = pd.MultiIndex.from_product([PARTICIPANT_GENDER_CHOICES, CONDITION_ORDER])
    wide_frame = wide_frame.reindex(columns=expected_columns).dropna()
    return wide_frame


def paired_posthoc_row(
    wide_frame: pd.DataFrame,
    family: str,
    effect: str,
    level_a: str,
    level_b: str,
    values_a: pd.Series,
    values_b: pd.Series,
) -> dict[str, object]:
    paired_frame = pd.concat([values_a.rename("a"), values_b.rename("b")], axis=1).dropna()
    difference_values = paired_frame["b"].to_numpy(dtype=float) - paired_frame["a"].to_numpy(dtype=float)
    t_result = stats.ttest_rel(paired_frame["b"], paired_frame["a"], nan_policy="omit")
    try:
        wilcoxon_result = stats.wilcoxon(difference_values, zero_method="wilcox", alternative="two-sided")
        wilcoxon_statistic = float(wilcoxon_result.statistic)
        wilcoxon_p = float(wilcoxon_result.pvalue)
    except ValueError:
        wilcoxon_statistic = math.nan
        wilcoxon_p = math.nan
    ci_low, ci_high = bootstrap_mean_ci(difference_values, iterations=3000)
    return {
        "family": family,
        "effect": effect,
        "level_a": level_a,
        "level_b": level_b,
        "n_templates": len(paired_frame),
        "mean_a": float(paired_frame["a"].mean()),
        "mean_b": float(paired_frame["b"].mean()),
        "mean_difference_b_minus_a": float(np.mean(difference_values)),
        "median_difference_b_minus_a": float(np.median(difference_values)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohens_dz": cohens_dz(difference_values),
        "t_stat": float(t_result.statistic),
        "t_p": float(t_result.pvalue),
        "wilcoxon_stat": wilcoxon_statistic,
        "wilcoxon_p": wilcoxon_p,
        "complete_item_matrix_n": len(wide_frame),
    }


def compute_h2a4_posthoc_tests(h2a4_cell_frame: pd.DataFrame) -> pd.DataFrame:
    """Paired post-hoc comparisons after the H2a.4 2x3 repeated-measures ANOVA."""
    wide_frame = build_complete_h2a4_wide_frame(h2a4_cell_frame)
    rows: list[dict[str, object]] = []

    male_average = wide_frame["男"].mean(axis=1)
    female_average = wide_frame["女"].mean(axis=1)
    rows.append(
        paired_posthoc_row(
            wide_frame,
            family="gender_main_effect",
            effect="被试性别主效应",
            level_a="男性被试（跨三种量表均值）",
            level_b="女性被试（跨三种量表均值）",
            values_a=male_average,
            values_b=female_average,
        )
    )

    for condition_name in CONDITION_ORDER:
        rows.append(
            paired_posthoc_row(
                wide_frame,
                family="gender_simple_effect_within_scale",
                effect=f"被试性别简单效应 | {condition_label(condition_name)}",
                level_a="男性被试",
                level_b="女性被试",
                values_a=wide_frame[("男", condition_name)],
                values_b=wide_frame[("女", condition_name)],
            )
        )

    for first_index, first_condition in enumerate(CONDITION_ORDER):
        for second_condition in CONDITION_ORDER[first_index + 1:]:
            first_average = wide_frame.loc[:, (slice(None), first_condition)].mean(axis=1)
            second_average = wide_frame.loc[:, (slice(None), second_condition)].mean(axis=1)
            rows.append(
                paired_posthoc_row(
                    wide_frame,
                    family="scale_main_effect",
                    effect="量表主效应",
                    level_a=condition_label(first_condition),
                    level_b=condition_label(second_condition),
                    values_a=first_average,
                    values_b=second_average,
                )
            )

    for participant_gender in PARTICIPANT_GENDER_CHOICES:
        gender_label = participant_gender_label(participant_gender)
        for first_index, first_condition in enumerate(CONDITION_ORDER):
            for second_condition in CONDITION_ORDER[first_index + 1:]:
                rows.append(
                    paired_posthoc_row(
                        wide_frame,
                        family="scale_simple_effect_within_gender",
                        effect=f"量表简单效应 | {gender_label}",
                        level_a=condition_label(first_condition),
                        level_b=condition_label(second_condition),
                        values_a=wide_frame[(participant_gender, first_condition)],
                        values_b=wide_frame[(participant_gender, second_condition)],
                    )
                )

    posthoc_frame = pd.DataFrame(rows)
    posthoc_frame["t_p_fdr"] = posthoc_frame.groupby("family")["t_p"].transform(benjamini_hochberg)
    posthoc_frame["wilcoxon_p_fdr"] = posthoc_frame.groupby("family")["wilcoxon_p"].transform(
        lambda values: benjamini_hochberg(values.fillna(1.0))
    )
    return posthoc_frame


def configure_matplotlib_fonts() -> None:
    preferred_fonts = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Serif CJK SC",
        "Noto Serif CJK JP",
        "Source Han Sans SC",
        "Source Han Serif SC",
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "Songti SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available_font_names = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next((font_name for font_name in preferred_fonts if font_name in available_font_names), None)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"] if selected_font else ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = JOURNAL_COLORS["background"]
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = JOURNAL_COLORS["spine"]
    plt.rcParams["axes.labelcolor"] = JOURNAL_COLORS["text"]
    plt.rcParams["axes.titlecolor"] = JOURNAL_COLORS["text"]
    plt.rcParams["text.color"] = JOURNAL_COLORS["text"]
    plt.rcParams["xtick.color"] = JOURNAL_COLORS["text"]
    plt.rcParams["ytick.color"] = JOURNAL_COLORS["text"]
    plt.rcParams["grid.color"] = JOURNAL_COLORS["grid"]
    plt.rcParams["grid.alpha"] = 0.45
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.titleweight"] = "semibold"
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10


def apply_journal_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(JOURNAL_COLORS["background"])
    axis.grid(axis="y", linestyle="-", linewidth=0.7, alpha=0.4)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(JOURNAL_COLORS["spine"])
    axis.spines["bottom"].set_color(JOURNAL_COLORS["spine"])


def add_panel_letter(axis: plt.Axes, letter: str) -> None:
    axis.text(
        -0.08,
        1.05,
        letter,
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
        color=JOURNAL_COLORS["text"],
    )


def format_q_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "q = NA"
    if p_value < 0.001:
        return "q < .001"
    return f"q = {p_value:.3f}".replace("0.", ".")


def significance_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


def save_axis_as_figure(axis: plt.Axes, output_path: Path) -> None:
    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    tight_bbox = axis.get_tightbbox(renderer).expanded(1.03, 1.06)
    tight_bbox_inches = tight_bbox.transformed(figure.dpi_scale_trans.inverted())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches=tight_bbox_inches)
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches=tight_bbox_inches)


def plot_figure1_paired_scores(
    difference_frame: pd.DataFrame,
    overall_frame: pd.DataFrame,
    output_path: Path,
    analysis_group_label: str,
    panel_output_dir: Path | None = None,
) -> None:
    apply_journal_style()
    configure_matplotlib_fonts()
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.2), sharey=False)

    for panel_index, (axis, condition_name) in enumerate(zip(axes.flat, CONDITION_ORDER)):
        style_axis(axis)
        panel_frame = difference_frame.loc[difference_frame["condition"] == condition_name].copy()
        stats_row = overall_frame.loc[overall_frame["condition"] == condition_name].iloc[0]
        male_scores = panel_frame["男人版攻击性评分"].to_numpy(dtype=float)
        female_scores = panel_frame["女人版攻击性评分"].to_numpy(dtype=float)
        jitter_male = RNG.normal(0.0, 0.022, size=len(panel_frame))
        jitter_female = RNG.normal(0.0, 0.022, size=len(panel_frame))

        for index in range(len(panel_frame)):
            axis.plot(
                [0 + jitter_male[index], 1 + jitter_female[index]],
                [male_scores[index], female_scores[index]],
                color=JOURNAL_COLORS["connector"],
                alpha=0.10,
                linewidth=0.75,
                zorder=1,
            )

        axis.scatter(
            np.full(len(panel_frame), 0) + jitter_male,
            male_scores,
            s=13,
            alpha=0.20,
            color=VERSION_COLOR_MAP["男人版"],
            zorder=2,
            linewidths=0,
        )
        axis.scatter(
            np.full(len(panel_frame), 1) + jitter_female,
            female_scores,
            s=13,
            alpha=0.20,
            color=VERSION_COLOR_MAP["女人版"],
            zorder=2,
            linewidths=0,
        )

        means = [stats_row["mean_male"], stats_row["mean_female"]]
        lower_errors = [
            means[0] - np.quantile(male_scores, 0.025),
            means[1] - np.quantile(female_scores, 0.025),
        ]
        upper_errors = [
            np.quantile(male_scores, 0.975) - means[0],
            np.quantile(female_scores, 0.975) - means[1],
        ]
        axis.plot([0, 1], means, color=JOURNAL_COLORS["text"], linewidth=1.4, alpha=0.9, zorder=3)
        axis.errorbar(
            [0, 1],
            means,
            yerr=[lower_errors, upper_errors],
            fmt="o",
            color=JOURNAL_COLORS["text"],
            ecolor=JOURNAL_COLORS["text"],
            elinewidth=1.7,
            capsize=4,
            markersize=7.5,
            zorder=4,
        )

        annotation = (
            f"Delta = {stats_row['mean_diff_female_minus_male']:.3f}\n"
            f"95% CI [{stats_row['ci_low']:.3f}, {stats_row['ci_high']:.3f}]\n"
            f"{format_q_value(stats_row['wilcoxon_p_fdr'])}\n"
            f"dz = {stats_row['cohens_dz']:.3f}"
        )
        axis.text(
            0.98,
            0.97,
            annotation,
            transform=axis.transAxes,
            va="top",
            ha="right",
            fontsize=9.5,
            color=JOURNAL_COLORS["muted_text"],
            bbox=dict(
                boxstyle="round,pad=0.32,rounding_size=0.14",
                facecolor=JOURNAL_COLORS["soft_white"],
                alpha=0.96,
                edgecolor="#E7E1D8",
                linewidth=0.7,
            ),
        )
        scale_min, scale_max = CONDITION_SCALE_LIMITS.get(condition_name, (float(min(male_scores.min(), female_scores.min())), float(max(male_scores.max(), female_scores.max()))))
        axis.set_xticks([0, 1], [version_label(name) for name in VERSION_ORDER])
        axis.set_ylim(scale_min - 0.05 * (scale_max - scale_min + 1), scale_max + 0.08 * (scale_max - scale_min + 1))
        axis.set_title(condition_label(condition_name), loc="left", pad=10)
        axis.set_xlabel("")
        axis.set_ylabel("攻击性评分")
        add_panel_letter(axis, chr(ord("A") + panel_index))

    fig.suptitle(f"图1 | 同一句话的男女版配对攻击性评分（{analysis_group_label}，按评分条件分面）", y=0.995, fontsize=17, fontweight="semibold")
    fig.text(
        0.5,
        0.01,
        "每条浅线连接同一模板下的男版与女版平均评分；黑色均值点与误差线用于强调总体趋势。",
        ha="center",
        va="bottom",
        fontsize=10,
        color=JOURNAL_COLORS["muted_text"],
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), w_pad=2.4)
    save_figure(fig, output_path)
    if panel_output_dir is not None:
        for panel_index, axis in enumerate(axes.flat, start=1):
            save_axis_as_figure(axis, panel_output_dir / f"fig1_paired_scores_panel_{panel_index}.png")
    plt.close(fig)


def plot_figure2_difference_distribution(
    difference_frame: pd.DataFrame,
    overall_frame: pd.DataFrame,
    output_path: Path,
    analysis_group_label: str,
    panel_output_dir: Path | None = None,
) -> None:
    apply_journal_style()
    configure_matplotlib_fonts()
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.9), sharey=False)

    for panel_index, (axis, condition_name) in enumerate(zip(axes.flat, CONDITION_ORDER)):
        style_axis(axis)
        panel_frame = difference_frame.loc[difference_frame["condition"] == condition_name].copy()
        stats_row = overall_frame.loc[overall_frame["condition"] == condition_name].iloc[0]
        values = panel_frame["女减男分差"].to_numpy(dtype=float)
        if np.allclose(values.min(), values.max()):
            bins = 15
        else:
            bins = min(30, max(12, int(np.sqrt(len(values)))))
        axis.hist(
            values,
            bins=bins,
            density=True,
            color=JOURNAL_COLORS["accent_gold"],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.8,
        )
        if len(np.unique(values)) > 1:
            sns.kdeplot(x=values, ax=axis, color=JOURNAL_COLORS["black"], linewidth=1.8, fill=False, bw_adjust=0.9)
        axis.axvline(0.0, color=JOURNAL_COLORS["neutral"], linestyle="--", linewidth=1.15)
        axis.axvline(stats_row["mean_diff_female_minus_male"], color=VERSION_COLOR_MAP["女人版"], linestyle="-", linewidth=1.9)
        axis.set_title(condition_label(condition_name), loc="left", pad=10)
        axis.set_xlabel("女版评分 - 男版评分")
        axis.set_ylabel("密度")
        annotation = (
            f"均值 = {stats_row['mean_diff_female_minus_male']:.3f}\n"
            f"中位数 = {stats_row['median_diff_female_minus_male']:.3f}\n"
            f"平均|分差| = {stats_row['mean_abs_diff']:.3f}\n"
            f"{format_q_value(stats_row['wilcoxon_p_fdr'])}"
        )
        axis.text(
            0.98,
            0.97,
            annotation,
            transform=axis.transAxes,
            va="top",
            ha="right",
            fontsize=9.5,
            color=JOURNAL_COLORS["muted_text"],
            bbox=dict(
                boxstyle="round,pad=0.32,rounding_size=0.14",
                facecolor=JOURNAL_COLORS["soft_white"],
                alpha=0.96,
                edgecolor="#E7E1D8",
                linewidth=0.7,
            ),
        )
        add_panel_letter(axis, chr(ord("A") + panel_index))

    fig.suptitle(f"图2 | 同一句话的配对分差分布（{analysis_group_label}，女版 - 男版）", y=0.995, fontsize=17, fontweight="semibold")
    fig.text(
        0.5,
        0.01,
        "半透明直方图显示经验分布，黑色密度曲线辅助观察整体形状；虚线为零差异，红棕线为平均分差。",
        ha="center",
        va="bottom",
        fontsize=10,
        color=JOURNAL_COLORS["muted_text"],
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), w_pad=2.4)
    save_figure(fig, output_path)
    if panel_output_dir is not None:
        for panel_index, axis in enumerate(axes.flat, start=1):
            save_axis_as_figure(axis, panel_output_dir / f"fig2_difference_distribution_panel_{panel_index}.png")
    plt.close(fig)


def plot_figure3_directionality(direction_frame: pd.DataFrame, output_path: Path, analysis_group_label: str) -> None:
    apply_journal_style()
    configure_matplotlib_fonts()
    ordered_frame = direction_frame.copy()
    ordered_frame["panel_label"] = ordered_frame["条件标签"]

    fig, axis = plt.subplots(figsize=(12.6, 8.6))
    style_axis(axis)
    bottoms = np.zeros(len(ordered_frame))
    x_positions = np.arange(len(ordered_frame))
    direction_column_map = {"女人版更高": "female_higher", "两者相同": "ties", "男人版更高": "male_higher"}

    for direction_name in DIRECTION_ORDER:
        values = ordered_frame[f"{direction_column_map[direction_name]}_pct"].to_numpy(dtype=float)
        axis.bar(
            x_positions,
            values,
            bottom=bottoms,
            color=DIRECTION_COLOR_MAP[direction_name],
            label=direction_label(direction_name),
            width=0.66,
            edgecolor="white",
            linewidth=0.8,
        )
        bottoms += values

    for index, row in ordered_frame.iterrows():
        annotation = (
            f"女>男 {row['female_higher']} · 相同 {row['ties']} · 男>女 {row['male_higher']}\n"
            f"{format_q_value(row['sign_test_p_fdr'])} {significance_stars(row['sign_test_p_fdr'])}"
        )
        axis.text(index, 1.03, annotation, ha="center", va="bottom", fontsize=9.2, color=JOURNAL_COLORS["muted_text"])

    axis.set_xticks(x_positions, ordered_frame["panel_label"].tolist())
    axis.set_ylabel("模板占比")
    axis.set_ylim(0, 1.2)
    axis.set_title(f"图3 | 三种评分条件下的分差方向（{analysis_group_label}）", loc="left", pad=12, fontsize=16, fontweight="semibold")
    legend = axis.legend(
        title="分差方向",
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        borderpad=0.8,
        labelspacing=0.6,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#E3DDD4")
    legend.get_frame().set_alpha(0.96)
    fig.text(
        0.5,
        0.01,
        "堆叠柱表示女版更高、两者相同、男版更高的比例构成；柱顶文字给出原始计数与符号检验校正 q 值。",
        ha="center",
        va="bottom",
        fontsize=10,
        color=JOURNAL_COLORS["muted_text"],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    save_figure(fig, output_path)
    plt.close(fig)


def plot_figure4_level1_forest(
    level1_frame: pd.DataFrame,
    output_path: Path,
    top_level1: int,
    analysis_group_label: str,
    panel_output_dir: Path | None = None,
) -> None:
    apply_journal_style()
    configure_matplotlib_fonts()
    fig, axes = plt.subplots(1, 3, figsize=(20.5, 8.8), sharex=False)

    for panel_index, (axis, condition_name) in enumerate(zip(axes.flat, CONDITION_ORDER)):
        style_axis(axis)
        panel_frame = level1_frame.loc[level1_frame["condition"] == condition_name].copy()
        panel_frame = panel_frame.nlargest(top_level1, columns="abs_mean_diff").sort_values("mean_diff_female_minus_male")
        y_positions = np.arange(len(panel_frame))
        for y_position, (_, row) in zip(y_positions, panel_frame.iterrows()):
            point_color = VERSION_COLOR_MAP["女人版"] if row["mean_diff_female_minus_male"] >= 0 else VERSION_COLOR_MAP["男人版"]
            point_alpha = 1.0 if row["wilcoxon_p_fdr"] < 0.05 else 0.42
            axis.plot([row["ci_low"], row["ci_high"]], [y_position, y_position], color=JOURNAL_COLORS["neutral"], linewidth=1.9, alpha=0.85)
            axis.scatter(
                row["mean_diff_female_minus_male"],
                y_position,
                s=28 + row["n"] * 4,
                color=point_color,
                alpha=point_alpha,
                edgecolor=JOURNAL_COLORS["text"],
                linewidth=0.45,
                zorder=3,
            )
        labels = [f"{name} ({n})" for name, n in zip(panel_frame["一级-攻击领域"], panel_frame["n"])]
        axis.set_yticks(y_positions, labels)
        axis.axvline(0.0, color=JOURNAL_COLORS["neutral"], linestyle="--", linewidth=1.2)
        axis.set_title(condition_label(condition_name), loc="left", pad=10)
        axis.set_xlabel("平均分差（女版 - 男版）")
        axis.set_ylabel("一级维度")
        add_panel_letter(axis, chr(ord("A") + panel_index))

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="红棕点：女版评分更高", markerfacecolor=VERSION_COLOR_MAP["女人版"], markeredgecolor=JOURNAL_COLORS["text"], markersize=10),
        Line2D([0], [0], marker="o", color="w", label="蓝灰点：男版评分更高", markerfacecolor=VERSION_COLOR_MAP["男人版"], markeredgecolor=JOURNAL_COLORS["text"], markersize=10),
        Line2D([0], [0], marker="o", color="w", label="深色：FDR < 0.05", markerfacecolor=JOURNAL_COLORS["neutral"], alpha=1.0, markersize=10),
        Line2D([0], [0], marker="o", color="w", label="浅色：FDR ≥ 0.05", markerfacecolor=JOURNAL_COLORS["neutral"], alpha=0.42, markersize=10),
        Line2D([0], [0], marker="o", color="w", label="点大小 ∝ 该维度模板数", markerfacecolor="#999999", markeredgecolor=JOURNAL_COLORS["text"], markersize=14),
        Line2D([0, 1], [0, 0], color=JOURNAL_COLORS["neutral"], linewidth=2.0, label="横线：95% 置信区间"),
    ]
    fig.suptitle(f"图4 | 一级维度效应及其 95% 置信区间（{analysis_group_label}，按绝对效应排序）", y=0.995, fontsize=17, fontweight="semibold")
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.01), fontsize=10.3)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95), w_pad=2.6)
    save_figure(fig, output_path)
    if panel_output_dir is not None:
        for panel_index, axis in enumerate(axes.flat, start=1):
            save_axis_as_figure(axis, panel_output_dir / f"fig4_level1_forest_panel_{panel_index}.png")
    plt.close(fig)


def save_statistics_tables(
    output_dir: Path,
    overall_frame: pd.DataFrame,
    direction_frame: pd.DataFrame,
    level1_frame: pd.DataFrame,
    h2a4_cell_descriptive_frame: pd.DataFrame | None = None,
    h2a4_posthoc_frame: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_frame.to_csv(output_dir / "stats_overall.csv", index=False, encoding="utf-8-sig")
    direction_frame.to_csv(output_dir / "stats_directionality.csv", index=False, encoding="utf-8-sig")
    level1_frame.to_csv(output_dir / "stats_level1.csv", index=False, encoding="utf-8-sig")
    if h2a4_cell_descriptive_frame is not None:
        h2a4_cell_descriptive_frame.to_csv(output_dir / "h2a4_cell_descriptives.csv", index=False, encoding="utf-8-sig")
    if h2a4_posthoc_frame is not None:
        h2a4_posthoc_frame.to_csv(output_dir / "h2a4_posthoc_pairwise.csv", index=False, encoding="utf-8-sig")


def write_excel_workbook(
    workbook_path: Path,
    template_score_frame: pd.DataFrame,
    difference_frame: pd.DataFrame,
    topic_ranking_frame: pd.DataFrame,
    overall_frame: pd.DataFrame,
    direction_frame: pd.DataFrame,
    level1_frame: pd.DataFrame,
    h2a4_cell_frame: pd.DataFrame | None = None,
    h2a4_cell_descriptive_frame: pd.DataFrame | None = None,
    h2a4_posthoc_frame: pd.DataFrame | None = None,
) -> None:
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(workbook_path, engine="xlsxwriter") as writer:
        template_score_frame.to_excel(writer, sheet_name="template_scores", index=False)
        difference_frame.to_excel(writer, sheet_name="same_sentence_diff", index=False)
        overall_frame.to_excel(writer, sheet_name="stats_overall", index=False)
        direction_frame.to_excel(writer, sheet_name="stats_direction", index=False)
        level1_frame.to_excel(writer, sheet_name="stats_level1", index=False)
        if h2a4_cell_frame is not None:
            h2a4_cell_frame.to_excel(writer, sheet_name="h2a4_item_cells", index=False)
        if h2a4_cell_descriptive_frame is not None:
            h2a4_cell_descriptive_frame.to_excel(writer, sheet_name="h2a4_cell_desc", index=False)
        if h2a4_posthoc_frame is not None:
            h2a4_posthoc_frame.to_excel(writer, sheet_name="h2a4_posthoc", index=False)
        for sheet_name, level_name in (("topic_rank_l1", "一级维度"), ("topic_rank_l2", "二级维度"), ("topic_rank_l3", "三级维度")):
            topic_ranking_frame.loc[topic_ranking_frame["维度层级"] == level_name].to_excel(writer, sheet_name=sheet_name, index=False)


def write_markdown_summary(
    output_dir: Path,
    input_long_path: Path,
    include_repeat_trials: bool,
    participant_gender: str | None,
    difference_frame: pd.DataFrame,
    overall_frame: pd.DataFrame,
    h2a4_posthoc_frame: pd.DataFrame | None = None,
) -> None:
    summary_lines = [
        "# Human gender-difference analysis summary",
        "",
        f"- Input long file: `{input_long_path}`",
        f"- Repeat trials included: `{include_repeat_trials}`",
        f"- Participant subgroup: `{participant_gender_label(participant_gender)}`",
        f"- Same-sentence paired rows: `{len(difference_frame)}`",
        "- Pairing assumption: within each condition, `template_id` identifies the same underlying sentence template, while `shown_version` distinguishes the male vs female surface form.",
        "",
        "## Overall condition-level summary",
        overall_frame.to_string(index=False),
        "",
    ]
    if h2a4_posthoc_frame is not None:
        summary_lines.extend(
            [
                "## H2a.4 post-hoc pairwise comparisons",
                h2a4_posthoc_frame[
                    [
                        "family",
                        "effect",
                        "level_a",
                        "level_b",
                        "mean_difference_b_minus_a",
                        "cohens_dz",
                        "t_p_fdr",
                        "wilcoxon_p_fdr",
                    ]
                ].to_string(index=False),
                "",
            ]
        )
    (output_dir / "analysis_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def write_csv_outputs(
    output_dir: Path,
    template_score_frame: pd.DataFrame,
    difference_frame: pd.DataFrame,
    gender_score_long_frame: pd.DataFrame,
    topic_ranking_frame: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    template_score_frame.to_csv(output_dir / "template_version_scores.csv", index=False, encoding="utf-8-sig")
    difference_frame.to_csv(output_dir / "same_sentence_gender_differences.csv", index=False, encoding="utf-8-sig")
    gender_score_long_frame.to_csv(output_dir / "gender_score_long.csv", index=False, encoding="utf-8-sig")
    topic_ranking_frame.to_csv(output_dir / "topic_rankings.csv", index=False, encoding="utf-8-sig")


def print_summary(
    output_dir: Path,
    participant_gender: str | None,
    difference_frame: pd.DataFrame,
    overall_frame: pd.DataFrame,
    level1_frame: pd.DataFrame,
) -> None:
    print("Human gender-difference analysis finished.")
    print(f"Output directory: {output_dir}")
    print(f"Participant subgroup: {participant_gender_label(participant_gender)}")
    print(f"Same-sentence difference rows: {len(difference_frame)}")
    print("Overall condition summary:")
    print(overall_frame[["condition", "n_templates", "mean_diff_female_minus_male", "wilcoxon_p_fdr"]].to_string(index=False))
    print("Level-1 rows by condition:")
    if level1_frame.empty:
        print("  (no level-1 rows)")
    else:
        print(level1_frame.groupby("condition").size().to_string())


def main() -> None:
    arguments = parse_arguments()
    analysis_group_label = participant_gender_label(arguments.gender)
    output_dir = arguments.output_dir or build_default_output_dir(arguments.input_long_path, arguments.gender)
    figure_output_dir = output_dir / "figures"
    panel_output_dir = figure_output_dir / "panels"

    long_frame = load_human_long_frame(arguments.input_long_path, arguments.include_repeat_trials, arguments.gender)
    template_score_frame = aggregate_template_scores(long_frame)
    difference_frame = build_same_sentence_difference_frame(template_score_frame)
    gender_score_long_frame = build_gender_score_long_frame(difference_frame)
    topic_ranking_frame = build_topic_ranking_frame(gender_score_long_frame)
    overall_frame = compute_overall_statistics(difference_frame, arguments.bootstrap_iterations)
    direction_frame = compute_direction_statistics(difference_frame)
    level1_frame = compute_level1_statistics(difference_frame, arguments.bootstrap_iterations)
    h2a4_cell_frame = None
    h2a4_cell_descriptive_frame = None
    h2a4_posthoc_frame = None
    if arguments.gender is None:
        h2a4_cell_frame = build_h2a4_item_cell_frame(long_frame)
        h2a4_cell_descriptive_frame = compute_h2a4_cell_descriptives(h2a4_cell_frame)
        h2a4_posthoc_frame = compute_h2a4_posthoc_tests(h2a4_cell_frame)

    write_csv_outputs(output_dir, template_score_frame, difference_frame, gender_score_long_frame, topic_ranking_frame)
    save_statistics_tables(
        output_dir,
        overall_frame,
        direction_frame,
        level1_frame,
        h2a4_cell_descriptive_frame,
        h2a4_posthoc_frame,
    )
    if h2a4_cell_frame is not None:
        h2a4_cell_frame.to_csv(output_dir / "h2a4_item_cells.csv", index=False, encoding="utf-8-sig")
    write_excel_workbook(
        output_dir / ANALYSIS_WORKBOOK_NAME,
        template_score_frame,
        difference_frame,
        topic_ranking_frame,
        overall_frame,
        direction_frame,
        level1_frame,
        h2a4_cell_frame,
        h2a4_cell_descriptive_frame,
        h2a4_posthoc_frame,
    )
    write_markdown_summary(
        output_dir,
        arguments.input_long_path,
        arguments.include_repeat_trials,
        arguments.gender,
        difference_frame,
        overall_frame,
        h2a4_posthoc_frame,
    )

    plot_figure1_paired_scores(
        difference_frame,
        overall_frame,
        figure_output_dir / "fig1_paired_scores.png",
        analysis_group_label,
        panel_output_dir=panel_output_dir,
    )
    plot_figure2_difference_distribution(
        difference_frame,
        overall_frame,
        figure_output_dir / "fig2_difference_distribution.png",
        analysis_group_label,
        panel_output_dir=panel_output_dir,
    )
    plot_figure3_directionality(direction_frame, figure_output_dir / "fig3_directionality.png", analysis_group_label)
    plot_figure4_level1_forest(
        level1_frame,
        figure_output_dir / "fig4_level1_forest.png",
        arguments.top_level1,
        analysis_group_label,
        panel_output_dir=panel_output_dir,
    )

    print_summary(output_dir, arguments.gender, difference_frame, overall_frame, level1_frame)


if __name__ == "__main__":
    main()
