"""Descriptive dz comparison between Study 1b LLMs and Study 2 humans.

This companion analysis fills the descriptive gap left by the ICC scripts:
it compares the female-minus-male gender-swap effect across 12 rater sources
(all humans, male human raters, female human raters, and 9 LLMs), then checks
human-model agreement at the same item and same dimension/type levels.

Outputs are written to ``artifacts/human_model_dz_descriptive_comparison/``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from human.analyze_gender_difference_pipeline import (
    CONDITION_LABEL_MAP,
    CONDITION_ORDER,
    CONDITION_SCALE_LIMITS,
    PARTICIPANT_GENDER_LABEL_MAP,
    aggregate_template_scores,
    benjamini_hochberg,
    build_same_sentence_difference_frame,
    configure_matplotlib_fonts,
    load_human_long_frame,
)
from human_model_delta_similarity import MODEL_FILE_LABELS


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HUMAN_LONG_PATH = PROJECT_ROOT / "human" / "outputs" / "final_clean_long.csv"
DEFAULT_MODEL_DIFFERENCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "group_swap_1b"
    / "1b_groupswap_demensionsentence"
    / "analysis_1b"
    / "same_sentence_gender_differences.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "human_model_dz_descriptive_comparison"

HUMAN_GROUPS: tuple[tuple[str, str | None, str], ...] = (
    ("human_all", None, "全部人类"),
    ("human_male", "男", "男性被试"),
    ("human_female", "女", "女性被试"),
)
DIMENSION_COLUMNS = ("dimension_1", "dimension_2", "dimension_3")
MODEL_PREFIX_LABELS = {
    filename.removesuffix("_results.csv"): label
    for filename, label in MODEL_FILE_LABELS.items()
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Study 1b LLM and Study 2 human female-minus-male dz "
            "descriptives at overall, item, and dimension/type levels."
        )
    )
    parser.add_argument(
        "--human-long-path",
        type=Path,
        default=DEFAULT_HUMAN_LONG_PATH,
        help="Cleaned long-format human data.",
    )
    parser.add_argument(
        "--model-difference-path",
        type=Path,
        default=DEFAULT_MODEL_DIFFERENCE_PATH,
        help="Study 1b same-sentence gender-difference table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for tables and figures.",
    )
    parser.add_argument(
        "--include-repeat-trials",
        action="store_true",
        help="Include repeat trials in the human aggregation.",
    )
    return parser.parse_args()


def cohens_dz(values: Iterable[float]) -> float:
    value_array = np.asarray(list(values), dtype=float)
    value_array = value_array[~np.isnan(value_array)]
    if value_array.size < 2:
        return math.nan
    standard_deviation = np.std(value_array, ddof=1)
    if np.isclose(standard_deviation, 0.0):
        return 0.0
    return float(np.mean(value_array) / standard_deviation)


def scale_maximum(condition_name: str) -> float:
    _, upper_bound = CONDITION_SCALE_LIMITS[condition_name]
    return float(upper_bound)


def direction_sign(values: pd.Series) -> pd.Series:
    return np.sign(values.astype(float)).astype(int)


def build_human_item_frame(
    human_long_path: Path,
    include_repeat_trials: bool,
) -> pd.DataFrame:
    item_frames: list[pd.DataFrame] = []
    for rater_id, participant_gender, rater_label in HUMAN_GROUPS:
        long_frame = load_human_long_frame(
            input_path=human_long_path,
            include_repeat_trials=include_repeat_trials,
            participant_gender=participant_gender,
        )
        participant_counts = (
            long_frame.groupby("condition")["session_id"].nunique().to_dict()
            if "session_id" in long_frame.columns
            else {}
        )
        template_score_frame = aggregate_template_scores(long_frame)
        difference_frame = build_same_sentence_difference_frame(template_score_frame)
        current_frame = difference_frame.rename(
            columns={
                "template_id": "item_id",
                "男人版攻击性评分": "male_score",
                "女人版攻击性评分": "female_score",
                "女减男分差": "delta_female_minus_male",
            }
        ).copy()
        current_frame["rater_kind"] = "human"
        current_frame["rater_id"] = rater_id
        current_frame["rater_label"] = rater_label
        current_frame["participant_gender"] = participant_gender or "all"
        current_frame["participant_gender_label"] = (
            PARTICIPANT_GENDER_LABEL_MAP.get(participant_gender, "全部被试")
            if participant_gender is not None
            else "全部被试"
        )
        current_frame["n_participants"] = current_frame["condition"].map(participant_counts)
        item_frames.append(
            current_frame[
                [
                    "rater_kind",
                    "rater_id",
                    "rater_label",
                    "participant_gender",
                    "participant_gender_label",
                    "condition",
                    "item_id",
                    *DIMENSION_COLUMNS,
                    "male_score",
                    "female_score",
                    "delta_female_minus_male",
                    "n_participants",
                ]
            ]
        )
    return pd.concat(item_frames, ignore_index=True)


def build_model_item_frame(model_difference_path: Path) -> pd.DataFrame:
    model_frame = pd.read_csv(model_difference_path, encoding="utf-8-sig")
    model_frame = model_frame.rename(
        columns={
            "一级-攻击领域": "dimension_1",
            "二级-攻击属性": "dimension_2",
            "三级-攻击表达策略": "dimension_3",
            "setting": "condition",
            "男人版攻击性评分": "male_score",
            "女人版攻击性评分": "female_score",
        }
    ).copy()
    model_frame["item_id"] = model_frame["句对ID"].map(lambda value: f"T{int(value):04d}")
    model_frame["delta_female_minus_male"] = (
        pd.to_numeric(model_frame["female_score"], errors="coerce")
        - pd.to_numeric(model_frame["male_score"], errors="coerce")
    )
    model_frame["rater_kind"] = "model"
    model_frame["rater_id"] = model_frame["model_prefix"]
    model_frame["rater_label"] = model_frame["model_prefix"].map(MODEL_PREFIX_LABELS).fillna(
        model_frame["model_prefix"]
    )
    model_frame["participant_gender"] = "model"
    model_frame["participant_gender_label"] = "LLM"
    model_frame["n_participants"] = np.nan
    return model_frame[
        [
            "rater_kind",
            "rater_id",
            "rater_label",
            "participant_gender",
            "participant_gender_label",
            "condition",
            "item_id",
            *DIMENSION_COLUMNS,
            "male_score",
            "female_score",
            "delta_female_minus_male",
            "n_participants",
        ]
    ]


def add_normalized_columns(item_frame: pd.DataFrame) -> pd.DataFrame:
    item_frame = item_frame.copy()
    item_frame["scale_label"] = item_frame["condition"].map(CONDITION_LABEL_MAP)
    item_frame["scale_max"] = item_frame["condition"].map(scale_maximum)
    for score_column in ("male_score", "female_score", "delta_female_minus_male"):
        item_frame[score_column] = pd.to_numeric(item_frame[score_column], errors="coerce")
        item_frame[f"{score_column}_norm"] = item_frame[score_column] / item_frame["scale_max"]
    item_frame["delta_sign"] = direction_sign(item_frame["delta_female_minus_male"])
    return item_frame


def compute_descriptive_statistics(item_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    order_map = {condition_name: index for index, condition_name in enumerate(CONDITION_ORDER)}
    for group_key, group in item_frame.groupby(
        ["rater_kind", "rater_id", "rater_label", "condition", "scale_label"],
        sort=False,
    ):
        rater_kind, rater_id, rater_label, condition_name, scale_label = group_key
        delta_values = group["delta_female_minus_male"].to_numpy(dtype=float)
        normalized_delta_values = group["delta_female_minus_male_norm"].to_numpy(dtype=float)
        rows.append(
            {
                "rater_kind": rater_kind,
                "rater_id": rater_id,
                "rater_label": rater_label,
                "condition": condition_name,
                "scale_label": scale_label,
                "scale_order": order_map[condition_name],
                "n_items": int(group["item_id"].nunique()),
                "n_participants": group["n_participants"].dropna().iloc[0]
                if group["n_participants"].notna().any()
                else np.nan,
                "mean_male": float(group["male_score"].mean()),
                "mean_female": float(group["female_score"].mean()),
                "mean_delta": float(np.mean(delta_values)),
                "median_delta": float(np.median(delta_values)),
                "sd_delta": float(np.std(delta_values, ddof=1)),
                "mean_abs_delta": float(np.mean(np.abs(delta_values))),
                "mean_male_norm": float(group["male_score_norm"].mean()),
                "mean_female_norm": float(group["female_score_norm"].mean()),
                "mean_delta_norm": float(np.mean(normalized_delta_values)),
                "median_delta_norm": float(np.median(normalized_delta_values)),
                "sd_delta_norm": float(np.std(normalized_delta_values, ddof=1)),
                "mean_abs_delta_norm": float(np.mean(np.abs(normalized_delta_values))),
                "cohens_dz": cohens_dz(delta_values),
                "female_higher": int(np.sum(delta_values > 0)),
                "male_higher": int(np.sum(delta_values < 0)),
                "ties": int(np.sum(delta_values == 0)),
            }
        )
    descriptive_frame = pd.DataFrame(rows)
    descriptive_frame["female_higher_pct"] = descriptive_frame["female_higher"] / descriptive_frame["n_items"]
    descriptive_frame["male_higher_pct"] = descriptive_frame["male_higher"] / descriptive_frame["n_items"]
    descriptive_frame["ties_pct"] = descriptive_frame["ties"] / descriptive_frame["n_items"]
    rater_order = {
        **{rater_id: index for index, (rater_id, _, _) in enumerate(HUMAN_GROUPS)},
        **{
            model_prefix: index + len(HUMAN_GROUPS)
            for index, model_prefix in enumerate(MODEL_PREFIX_LABELS)
        },
    }
    descriptive_frame["rater_order"] = descriptive_frame["rater_id"].map(rater_order).fillna(999)
    return descriptive_frame.sort_values(["rater_order", "scale_order"]).reset_index(drop=True)


def compute_pairwise_item_comparison(item_frame: pd.DataFrame) -> pd.DataFrame:
    human_frame = item_frame.loc[item_frame["rater_kind"] == "human"].copy()
    model_frame = item_frame.loc[item_frame["rater_kind"] == "model"].copy()
    comparison_frames: list[pd.DataFrame] = []
    for human_rater_id, human_group in human_frame.groupby("rater_id", sort=False):
        for model_rater_id, model_group in model_frame.groupby("rater_id", sort=False):
            merged = human_group.merge(
                model_group,
                on=["condition", "scale_label", "item_id", *DIMENSION_COLUMNS],
                how="inner",
                suffixes=("_human", "_model"),
            )
            merged["human_group_id"] = human_rater_id
            merged["human_group_label"] = merged["rater_label_human"]
            merged["model_id"] = model_rater_id
            merged["model_label"] = merged["rater_label_model"]
            merged["delta_gap_model_minus_human"] = (
                merged["delta_female_minus_male_model"]
                - merged["delta_female_minus_male_human"]
            )
            merged["delta_gap_model_minus_human_norm"] = (
                merged["delta_female_minus_male_norm_model"]
                - merged["delta_female_minus_male_norm_human"]
            )
            merged["abs_delta_gap_norm"] = merged["delta_gap_model_minus_human_norm"].abs()
            merged["same_direction"] = (
                merged["delta_sign_human"] == merged["delta_sign_model"]
            )
            comparison_frames.append(
                merged[
                    [
                        "human_group_id",
                        "human_group_label",
                        "model_id",
                        "model_label",
                        "condition",
                        "scale_label",
                        "item_id",
                        *DIMENSION_COLUMNS,
                        "delta_female_minus_male_human",
                        "delta_female_minus_male_model",
                        "delta_female_minus_male_norm_human",
                        "delta_female_minus_male_norm_model",
                        "delta_gap_model_minus_human",
                        "delta_gap_model_minus_human_norm",
                        "abs_delta_gap_norm",
                        "delta_sign_human",
                        "delta_sign_model",
                        "same_direction",
                    ]
                ]
            )
    return pd.concat(comparison_frames, ignore_index=True)


def paired_test_row(values_a: np.ndarray, values_b: np.ndarray) -> dict[str, float]:
    difference_values = values_b - values_a
    t_result = stats.ttest_rel(values_b, values_a, nan_policy="omit")
    try:
        wilcoxon_result = stats.wilcoxon(difference_values, zero_method="wilcox")
        wilcoxon_statistic = float(wilcoxon_result.statistic)
        wilcoxon_p_value = float(wilcoxon_result.pvalue)
    except ValueError:
        wilcoxon_statistic = math.nan
        wilcoxon_p_value = math.nan
    return {
        "mean_a": float(np.mean(values_a)),
        "mean_b": float(np.mean(values_b)),
        "mean_difference_b_minus_a": float(np.mean(difference_values)),
        "median_difference_b_minus_a": float(np.median(difference_values)),
        "cohens_dz_difference": cohens_dz(difference_values),
        "t_stat": float(t_result.statistic),
        "t_p": float(t_result.pvalue),
        "wilcoxon_stat": wilcoxon_statistic,
        "wilcoxon_p": wilcoxon_p_value,
    }


def compute_pairwise_summary(item_comparison_frame: pd.DataFrame, descriptive_frame: pd.DataFrame) -> pd.DataFrame:
    dz_lookup = descriptive_frame.set_index(["rater_id", "condition"])["cohens_dz"].to_dict()
    rows: list[dict[str, object]] = []
    for group_key, group in item_comparison_frame.groupby(
        ["human_group_id", "human_group_label", "model_id", "model_label", "condition", "scale_label"],
        sort=False,
    ):
        human_group_id, human_group_label, model_id, model_label, condition_name, scale_label = group_key
        human_values = group["delta_female_minus_male_norm_human"].to_numpy(dtype=float)
        model_values = group["delta_female_minus_male_norm_model"].to_numpy(dtype=float)
        test_values = paired_test_row(human_values, model_values)
        rows.append(
            {
                "human_group_id": human_group_id,
                "human_group_label": human_group_label,
                "model_id": model_id,
                "model_label": model_label,
                "condition": condition_name,
                "scale_label": scale_label,
                "n_items": len(group),
                "human_dz": dz_lookup.get((human_group_id, condition_name), math.nan),
                "model_dz": dz_lookup.get((model_id, condition_name), math.nan),
                "dz_gap_model_minus_human": dz_lookup.get((model_id, condition_name), math.nan)
                - dz_lookup.get((human_group_id, condition_name), math.nan),
                "pearson_r": float(pd.Series(human_values).corr(pd.Series(model_values), method="pearson")),
                "spearman_rho": float(pd.Series(human_values).corr(pd.Series(model_values), method="spearman")),
                "rmse_norm": float(np.sqrt(np.mean((model_values - human_values) ** 2))),
                "mae_norm": float(np.mean(np.abs(model_values - human_values))),
                "same_direction_rate": float(group["same_direction"].mean()),
                **test_values,
            }
        )
    summary_frame = pd.DataFrame(rows)
    summary_frame["t_p_fdr"] = summary_frame.groupby(["human_group_id", "condition"])["t_p"].transform(
        benjamini_hochberg
    )
    summary_frame["wilcoxon_p_fdr"] = summary_frame.groupby(["human_group_id", "condition"])[
        "wilcoxon_p"
    ].transform(lambda values: benjamini_hochberg(values.fillna(1.0)))
    return summary_frame.sort_values(
        ["human_group_id", "condition", "mae_norm", "model_label"]
    ).reset_index(drop=True)


def compute_dimension_comparison(item_comparison_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dimension_level, dimension_column in enumerate(DIMENSION_COLUMNS, start=1):
        group_columns = [
            "human_group_id",
            "human_group_label",
            "model_id",
            "model_label",
            "condition",
            "scale_label",
            dimension_column,
        ]
        for group_key, group in item_comparison_frame.groupby(group_columns, sort=False):
            (
                human_group_id,
                human_group_label,
                model_id,
                model_label,
                condition_name,
                scale_label,
                dimension_name,
            ) = group_key
            human_values = group["delta_female_minus_male_norm_human"].to_numpy(dtype=float)
            model_values = group["delta_female_minus_male_norm_model"].to_numpy(dtype=float)
            rows.append(
                {
                    "dimension_level": dimension_level,
                    "dimension_column": dimension_column,
                    "dimension_name": dimension_name,
                    "human_group_id": human_group_id,
                    "human_group_label": human_group_label,
                    "model_id": model_id,
                    "model_label": model_label,
                    "condition": condition_name,
                    "scale_label": scale_label,
                    "n_items": len(group),
                    "human_mean_delta_norm": float(np.mean(human_values)),
                    "model_mean_delta_norm": float(np.mean(model_values)),
                    "mean_gap_model_minus_human_norm": float(np.mean(model_values - human_values)),
                    "abs_mean_gap_norm": float(abs(np.mean(model_values) - np.mean(human_values))),
                    "human_dz": cohens_dz(human_values),
                    "model_dz": cohens_dz(model_values),
                    "dz_gap_model_minus_human": cohens_dz(model_values) - cohens_dz(human_values),
                    "same_direction_rate": float(group["same_direction"].mean()),
                    "aggregate_direction_match": int(np.sign(np.mean(human_values)) == np.sign(np.mean(model_values))),
                }
            )
    return pd.DataFrame(rows).sort_values(
        [
            "dimension_level",
            "human_group_id",
            "condition",
            "abs_mean_gap_norm",
            "model_label",
            "dimension_name",
        ],
        ascending=[True, True, True, False, True, True],
    ).reset_index(drop=True)


def save_descriptive_heatmap(descriptive_frame: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    configure_matplotlib_fonts()
    heatmap_frame = descriptive_frame.pivot(
        index="rater_label", columns="scale_label", values="cohens_dz"
    )
    heatmap_frame = heatmap_frame[[CONDITION_LABEL_MAP[name] for name in CONDITION_ORDER]]
    figure_height = max(5.8, 0.42 * len(heatmap_frame) + 1.6)
    fig, axis = plt.subplots(figsize=(8.2, figure_height))
    sns.heatmap(
        heatmap_frame,
        ax=axis,
        annot=True,
        fmt=".3f",
        cmap="vlag",
        center=0.0,
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Cohen's dz (女版 - 男版)"},
    )
    axis.set_title("12 个评分来源在三种量表上的性别反转效应量", loc="left", pad=12)
    axis.set_xlabel("评分量表")
    axis.set_ylabel("评分来源")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_dz_gap_heatmap(pairwise_summary_frame: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    configure_matplotlib_fonts()
    human_groups = pairwise_summary_frame[["human_group_id", "human_group_label"]].drop_duplicates()
    fig, axes = plt.subplots(1, len(human_groups), figsize=(17.2, 7.0), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, (_, human_row) in zip(axes, human_groups.iterrows()):
        panel_frame = pairwise_summary_frame.loc[
            pairwise_summary_frame["human_group_id"] == human_row["human_group_id"]
        ].copy()
        heatmap_frame = panel_frame.pivot(
            index="model_label", columns="scale_label", values="dz_gap_model_minus_human"
        )
        heatmap_frame = heatmap_frame[[CONDITION_LABEL_MAP[name] for name in CONDITION_ORDER]]
        sns.heatmap(
            heatmap_frame,
            ax=axis,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0.0,
            linewidths=0.7,
            linecolor="white",
            cbar=axis is axes[-1],
            cbar_kws={"label": "模型 dz - 人类 dz"} if axis is axes[-1] else None,
        )
        axis.set_title(human_row["human_group_label"], loc="left", pad=10)
        axis.set_xlabel("评分量表")
        axis.set_ylabel("模型" if axis is axes[0] else "")
    fig.suptitle("模型相对人类的 dz 差异：正值表示模型女性方向偏差更强", y=1.02, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    item_frame: pd.DataFrame,
    descriptive_frame: pd.DataFrame,
    item_comparison_frame: pd.DataFrame,
    pairwise_summary_frame: pd.DataFrame,
    dimension_comparison_frame: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    item_frame.to_csv(output_dir / "all_rater_item_deltas.csv", index=False, encoding="utf-8-sig")
    descriptive_frame.to_csv(output_dir / "dz_descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    item_comparison_frame.to_csv(output_dir / "same_item_human_model_comparison.csv", index=False, encoding="utf-8-sig")
    pairwise_summary_frame.to_csv(output_dir / "same_item_human_model_summary.csv", index=False, encoding="utf-8-sig")
    dimension_comparison_frame.to_csv(output_dir / "same_type_dimension_comparison.csv", index=False, encoding="utf-8-sig")
    save_descriptive_heatmap(descriptive_frame, output_dir / "fig1_dz_descriptive_heatmap.png")
    save_dz_gap_heatmap(pairwise_summary_frame, output_dir / "fig2_model_minus_human_dz_gap_heatmap.png")


def print_summary(descriptive_frame: pd.DataFrame, pairwise_summary_frame: pd.DataFrame) -> None:
    print("Human-model dz descriptive comparison finished.")
    print("Top descriptive dz rows:")
    print(
        descriptive_frame[
            ["rater_label", "scale_label", "n_items", "mean_delta", "mean_delta_norm", "cohens_dz"]
        ]
        .sort_values("cohens_dz", ascending=False)
        .head(12)
        .to_string(index=False)
    )
    print("\nClosest human-model pairs by normalized MAE:")
    print(
        pairwise_summary_frame[
            ["human_group_label", "model_label", "scale_label", "mae_norm", "same_direction_rate", "dz_gap_model_minus_human"]
        ]
        .sort_values("mae_norm")
        .head(12)
        .to_string(index=False)
    )


def main() -> None:
    arguments = parse_arguments()
    human_item_frame = build_human_item_frame(arguments.human_long_path, arguments.include_repeat_trials)
    model_item_frame = build_model_item_frame(arguments.model_difference_path)
    item_frame = add_normalized_columns(pd.concat([human_item_frame, model_item_frame], ignore_index=True))
    descriptive_frame = compute_descriptive_statistics(item_frame)
    item_comparison_frame = compute_pairwise_item_comparison(item_frame)
    pairwise_summary_frame = compute_pairwise_summary(item_comparison_frame, descriptive_frame)
    dimension_comparison_frame = compute_dimension_comparison(item_comparison_frame)
    write_outputs(
        arguments.output_dir,
        item_frame,
        descriptive_frame,
        item_comparison_frame,
        pairwise_summary_frame,
        dimension_comparison_frame,
    )
    print_summary(descriptive_frame, pairwise_summary_frame)
    print(f"\nOutput directory: {arguments.output_dir}")


if __name__ == "__main__":
    main()
