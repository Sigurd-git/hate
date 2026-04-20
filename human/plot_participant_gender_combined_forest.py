from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    import analyze_gender_difference_pipeline as base_analysis
except ModuleNotFoundError:
    sibling_module_path = Path(__file__).resolve().parent / "analyze_gender_difference_pipeline.py"
    sibling_module_spec = importlib.util.spec_from_file_location("analyze_gender_difference_pipeline", sibling_module_path)
    if sibling_module_spec is None or sibling_module_spec.loader is None:
        raise
    base_analysis = importlib.util.module_from_spec(sibling_module_spec)
    sibling_module_spec.loader.exec_module(base_analysis)


PARTICIPANT_GENDER_ORDER: list[str] = ["男", "女"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one six-panel forest plot that compares male-vs-female sentence-version effects "
            "across participant gender subgroups and rating conditions."
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
            "Directory for the combined participant-gender forest plot and summary tables. "
            "Defaults to a sibling folder named <input_stem>_participant_gender_combined_analysis."
        ),
    )
    parser.add_argument(
        "--include-repeat-trials",
        action="store_true",
        help="Include repeat trials in the subgroup analyses. By default, repeat trials are excluded.",
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
        help="Maximum number of level-1 dimensions to show in each subplot.",
    )
    return parser.parse_args()


def build_default_output_dir(input_long_path: Path) -> Path:
    return input_long_path.parent / f"{input_long_path.stem}_participant_gender_combined_analysis"


def run_subgroup_analysis(
    input_long_path: Path,
    include_repeat_trials: bool,
    participant_gender: str,
    bootstrap_iterations: int,
) -> dict[str, pd.DataFrame]:
    """Run the existing subgroup pipeline and attach participant-gender metadata to the outputs."""
    long_frame = base_analysis.load_human_long_frame(
        input_path=input_long_path,
        include_repeat_trials=include_repeat_trials,
        participant_gender=participant_gender,
    )
    template_score_frame = base_analysis.aggregate_template_scores(long_frame)
    difference_frame = base_analysis.build_same_sentence_difference_frame(template_score_frame)
    overall_frame = base_analysis.compute_overall_statistics(difference_frame, bootstrap_iterations)
    direction_frame = base_analysis.compute_direction_statistics(difference_frame)
    level1_frame = base_analysis.compute_level1_statistics(difference_frame, bootstrap_iterations)

    participant_gender_text = base_analysis.participant_gender_label(participant_gender)
    output_frames = {
        "template_score_frame": template_score_frame.copy(),
        "difference_frame": difference_frame.copy(),
        "overall_frame": overall_frame.copy(),
        "direction_frame": direction_frame.copy(),
        "level1_frame": level1_frame.copy(),
    }
    for frame_name, dataframe in output_frames.items():
        dataframe["participant_gender"] = participant_gender
        dataframe["participant_gender_label"] = participant_gender_text
        output_frames[frame_name] = dataframe
    return output_frames


def panel_title(participant_gender: str, condition_name: str) -> str:
    return f"{base_analysis.participant_gender_label(participant_gender)} | {base_analysis.condition_label(condition_name)}"


def get_panel_letter(row_index: int, column_index: int) -> str:
    return chr(ord("A") + row_index * len(base_analysis.CONDITION_ORDER) + column_index)


def plot_combined_paired_scores(
    difference_frame: pd.DataFrame,
    overall_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot six paired-score panels across participant gender and rating condition."""
    base_analysis.apply_journal_style()
    base_analysis.configure_matplotlib_fonts()
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 11.5), sharey=False)

    for row_index, participant_gender in enumerate(PARTICIPANT_GENDER_ORDER):
        for column_index, condition_name in enumerate(base_analysis.CONDITION_ORDER):
            axis = axes[row_index, column_index]
            base_analysis.style_axis(axis)
            panel_frame = difference_frame.loc[
                (difference_frame["participant_gender"] == participant_gender)
                & (difference_frame["condition"] == condition_name)
            ].copy()
            stats_row = overall_frame.loc[
                (overall_frame["participant_gender"] == participant_gender)
                & (overall_frame["condition"] == condition_name)
            ].iloc[0]
            male_scores = panel_frame["男人版攻击性评分"].to_numpy(dtype=float)
            female_scores = panel_frame["女人版攻击性评分"].to_numpy(dtype=float)
            jitter_male = base_analysis.RNG.normal(0.0, 0.022, size=len(panel_frame))
            jitter_female = base_analysis.RNG.normal(0.0, 0.022, size=len(panel_frame))

            for index in range(len(panel_frame)):
                axis.plot(
                    [0 + jitter_male[index], 1 + jitter_female[index]],
                    [male_scores[index], female_scores[index]],
                    color=base_analysis.JOURNAL_COLORS["connector"],
                    alpha=0.10,
                    linewidth=0.75,
                    zorder=1,
                )

            axis.scatter(
                np.full(len(panel_frame), 0) + jitter_male,
                male_scores,
                s=13,
                alpha=0.20,
                color=base_analysis.VERSION_COLOR_MAP["男人版"],
                zorder=2,
                linewidths=0,
            )
            axis.scatter(
                np.full(len(panel_frame), 1) + jitter_female,
                female_scores,
                s=13,
                alpha=0.20,
                color=base_analysis.VERSION_COLOR_MAP["女人版"],
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
            axis.plot([0, 1], means, color=base_analysis.JOURNAL_COLORS["text"], linewidth=1.4, alpha=0.9, zorder=3)
            axis.errorbar(
                [0, 1],
                means,
                yerr=[lower_errors, upper_errors],
                fmt="o",
                color=base_analysis.JOURNAL_COLORS["text"],
                ecolor=base_analysis.JOURNAL_COLORS["text"],
                elinewidth=1.7,
                capsize=4,
                markersize=7.5,
                zorder=4,
            )

            annotation = (
                f"Delta = {stats_row['mean_diff_female_minus_male']:.3f}\n"
                f"95% CI [{stats_row['ci_low']:.3f}, {stats_row['ci_high']:.3f}]\n"
                f"{base_analysis.format_q_value(stats_row['wilcoxon_p_fdr'])}\n"
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
                color=base_analysis.JOURNAL_COLORS["muted_text"],
                bbox=dict(
                    boxstyle="round,pad=0.32,rounding_size=0.14",
                    facecolor=base_analysis.JOURNAL_COLORS["soft_white"],
                    alpha=0.96,
                    edgecolor="#E7E1D8",
                    linewidth=0.7,
                ),
            )
            scale_min, scale_max = base_analysis.CONDITION_SCALE_LIMITS[condition_name]
            axis.set_xticks([0, 1], [base_analysis.version_label(name) for name in base_analysis.VERSION_ORDER])
            axis.set_ylim(scale_min - 0.05 * (scale_max - scale_min + 1), scale_max + 0.08 * (scale_max - scale_min + 1))
            axis.set_title(panel_title(participant_gender, condition_name), loc="left", pad=10)
            axis.set_xlabel("")
            axis.set_ylabel("攻击性评分")
            base_analysis.add_panel_letter(axis, get_panel_letter(row_index, column_index))

    fig.suptitle("图1 | 同一句话的男女版配对攻击性评分（男女被试 × 三种评分条件）", y=0.995, fontsize=17, fontweight="semibold")
    fig.text(
        0.5,
        0.01,
        "每条浅线连接同一模板下的男版与女版平均评分；黑色均值点与误差线用于强调各子组内部趋势。",
        ha="center",
        va="bottom",
        fontsize=10,
        color=base_analysis.JOURNAL_COLORS["muted_text"],
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), w_pad=2.3, h_pad=2.3)
    base_analysis.save_figure(fig, output_path)
    plt.close(fig)


def plot_combined_difference_distribution(
    difference_frame: pd.DataFrame,
    overall_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot six difference-distribution panels across participant gender and rating condition."""
    base_analysis.apply_journal_style()
    base_analysis.configure_matplotlib_fonts()
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 11.2), sharey=False)

    for row_index, participant_gender in enumerate(PARTICIPANT_GENDER_ORDER):
        for column_index, condition_name in enumerate(base_analysis.CONDITION_ORDER):
            axis = axes[row_index, column_index]
            base_analysis.style_axis(axis)
            panel_frame = difference_frame.loc[
                (difference_frame["participant_gender"] == participant_gender)
                & (difference_frame["condition"] == condition_name)
            ].copy()
            stats_row = overall_frame.loc[
                (overall_frame["participant_gender"] == participant_gender)
                & (overall_frame["condition"] == condition_name)
            ].iloc[0]
            values = panel_frame["女减男分差"].to_numpy(dtype=float)
            if np.allclose(values.min(), values.max()):
                bins = 15
            else:
                bins = min(30, max(12, int(np.sqrt(len(values)))))
            axis.hist(
                values,
                bins=bins,
                density=True,
                color=base_analysis.JOURNAL_COLORS["accent_gold"],
                alpha=0.55,
                edgecolor="white",
                linewidth=0.8,
            )
            if len(np.unique(values)) > 1:
                base_analysis.sns.kdeplot(x=values, ax=axis, color=base_analysis.JOURNAL_COLORS["black"], linewidth=1.8, fill=False, bw_adjust=0.9)
            axis.axvline(0.0, color=base_analysis.JOURNAL_COLORS["neutral"], linestyle="--", linewidth=1.15)
            axis.axvline(stats_row["mean_diff_female_minus_male"], color=base_analysis.VERSION_COLOR_MAP["女人版"], linestyle="-", linewidth=1.9)
            axis.set_title(panel_title(participant_gender, condition_name), loc="left", pad=10)
            axis.set_xlabel("女版评分 - 男版评分")
            axis.set_ylabel("密度")
            annotation = (
                f"均值 = {stats_row['mean_diff_female_minus_male']:.3f}\n"
                f"中位数 = {stats_row['median_diff_female_minus_male']:.3f}\n"
                f"平均|分差| = {stats_row['mean_abs_diff']:.3f}\n"
                f"{base_analysis.format_q_value(stats_row['wilcoxon_p_fdr'])}"
            )
            axis.text(
                0.98,
                0.97,
                annotation,
                transform=axis.transAxes,
                va="top",
                ha="right",
                fontsize=9.5,
                color=base_analysis.JOURNAL_COLORS["muted_text"],
                bbox=dict(
                    boxstyle="round,pad=0.32,rounding_size=0.14",
                    facecolor=base_analysis.JOURNAL_COLORS["soft_white"],
                    alpha=0.96,
                    edgecolor="#E7E1D8",
                    linewidth=0.7,
                ),
            )
            base_analysis.add_panel_letter(axis, get_panel_letter(row_index, column_index))

    fig.suptitle("图2 | 同一句话的配对分差分布（男女被试 × 三种评分条件）", y=0.995, fontsize=17, fontweight="semibold")
    fig.text(
        0.5,
        0.01,
        "半透明直方图显示经验分布，黑色密度曲线辅助观察整体形状；虚线为零差异，红棕线为平均分差。",
        ha="center",
        va="bottom",
        fontsize=10,
        color=base_analysis.JOURNAL_COLORS["muted_text"],
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), w_pad=2.3, h_pad=2.3)
    base_analysis.save_figure(fig, output_path)
    plt.close(fig)


def plot_combined_directionality(
    direction_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot six directionality panels across participant gender and rating condition."""
    base_analysis.apply_journal_style()
    base_analysis.configure_matplotlib_fonts()
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.8), sharey=True)

    direction_column_map = {"女人版更高": "female_higher", "两者相同": "ties", "男人版更高": "male_higher"}
    for row_index, participant_gender in enumerate(PARTICIPANT_GENDER_ORDER):
        for column_index, condition_name in enumerate(base_analysis.CONDITION_ORDER):
            axis = axes[row_index, column_index]
            base_analysis.style_axis(axis)
            row = direction_frame.loc[
                (direction_frame["participant_gender"] == participant_gender)
                & (direction_frame["condition"] == condition_name)
            ].iloc[0]
            bottoms = 0.0
            for direction_name in base_analysis.DIRECTION_ORDER:
                value = float(row[f"{direction_column_map[direction_name]}_pct"])
                axis.bar(
                    [0],
                    [value],
                    bottom=bottoms,
                    color=base_analysis.DIRECTION_COLOR_MAP[direction_name],
                    width=0.62,
                    edgecolor="white",
                    linewidth=0.8,
                )
                bottoms += value

            annotation = (
                f"女>男 {row['female_higher']} · 相同 {row['ties']} · 男>女 {row['male_higher']}\n"
                f"{base_analysis.format_q_value(row['sign_test_p_fdr'])} {base_analysis.significance_stars(row['sign_test_p_fdr'])}"
            )
            axis.text(0, 1.03, annotation, ha="center", va="bottom", fontsize=9.2, color=base_analysis.JOURNAL_COLORS["muted_text"])
            axis.set_xticks([0], ["分差方向"])
            axis.set_ylim(0, 1.2)
            axis.set_ylabel("模板占比")
            axis.set_title(panel_title(participant_gender, condition_name), loc="left", pad=10)
            base_analysis.add_panel_letter(axis, get_panel_letter(row_index, column_index))

    legend_handles = [
        Line2D([0], [0], color=base_analysis.DIRECTION_COLOR_MAP[direction_name], linewidth=10, label=base_analysis.direction_label(direction_name))
        for direction_name in base_analysis.DIRECTION_ORDER
    ]
    fig.suptitle("图3 | 分差方向构成（男女被试 × 三种评分条件）", y=0.995, fontsize=17, fontweight="semibold")
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.04), fontsize=10.3)
    fig.text(
        0.5,
        0.01,
        "每个子图对应一个被试子组与评分条件；柱体展示女版更高、两者相同、男版更高的比例构成。",
        ha="center",
        va="bottom",
        fontsize=10,
        color=base_analysis.JOURNAL_COLORS["muted_text"],
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95), w_pad=2.3, h_pad=2.4)
    base_analysis.save_figure(fig, output_path)
    plt.close(fig)


def plot_combined_level1_forest(
    level1_frame: pd.DataFrame,
    output_path: Path,
    top_level1: int,
) -> None:
    """Plot the top level-1 dimensions for each participant-gender subgroup and condition."""
    base_analysis.apply_journal_style()
    base_analysis.configure_matplotlib_fonts()
    fig, axes = plt.subplots(2, 3, figsize=(20.5, 13.5), sharex=False)

    for row_index, participant_gender in enumerate(PARTICIPANT_GENDER_ORDER):
        for column_index, condition_name in enumerate(base_analysis.CONDITION_ORDER):
            axis = axes[row_index, column_index]
            base_analysis.style_axis(axis)
            panel_frame = level1_frame.loc[
                (level1_frame["participant_gender"] == participant_gender)
                & (level1_frame["condition"] == condition_name)
            ].copy()

            if panel_frame.empty:
                axis.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    color=base_analysis.JOURNAL_COLORS["muted_text"],
                )
                axis.set_title(panel_title(participant_gender, condition_name), loc="left", pad=10)
                axis.set_xlabel("平均分差（女版 - 男版）")
                axis.set_ylabel("一级维度" if column_index == 0 else "")
                continue

            panel_frame = panel_frame.nlargest(top_level1, columns="abs_mean_diff").sort_values("mean_diff_female_minus_male")
            y_positions = np.arange(len(panel_frame))

            for y_position, (_, row) in zip(y_positions, panel_frame.iterrows()):
                point_color = (
                    base_analysis.VERSION_COLOR_MAP["女人版"]
                    if row["mean_diff_female_minus_male"] >= 0
                    else base_analysis.VERSION_COLOR_MAP["男人版"]
                )
                point_alpha = 1.0 if row["wilcoxon_p_fdr"] < 0.05 else 0.42
                axis.plot(
                    [row["ci_low"], row["ci_high"]],
                    [y_position, y_position],
                    color=base_analysis.JOURNAL_COLORS["neutral"],
                    linewidth=1.9,
                    alpha=0.85,
                )
                axis.scatter(
                    row["mean_diff_female_minus_male"],
                    y_position,
                    s=28 + row["n"] * 4,
                    color=point_color,
                    alpha=point_alpha,
                    edgecolor=base_analysis.JOURNAL_COLORS["text"],
                    linewidth=0.45,
                    zorder=3,
                )

            y_labels = [f"{dimension_name} ({count})" for dimension_name, count in zip(panel_frame["一级-攻击领域"], panel_frame["n"])]
            axis.set_yticks(y_positions, y_labels)
            axis.axvline(0.0, color=base_analysis.JOURNAL_COLORS["neutral"], linestyle="--", linewidth=1.2)
            axis.set_title(panel_title(participant_gender, condition_name), loc="left", pad=10)
            axis.set_xlabel("平均分差（女版 - 男版）")
            axis.set_ylabel("一级维度" if column_index == 0 else "")
            base_analysis.add_panel_letter(axis, get_panel_letter(row_index, column_index))

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="红棕点：女版评分更高",
            markerfacecolor=base_analysis.VERSION_COLOR_MAP["女人版"],
            markeredgecolor=base_analysis.JOURNAL_COLORS["text"],
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="蓝灰点：男版评分更高",
            markerfacecolor=base_analysis.VERSION_COLOR_MAP["男人版"],
            markeredgecolor=base_analysis.JOURNAL_COLORS["text"],
            markersize=10,
        ),
        Line2D([0], [0], marker="o", color="w", label="深色：FDR < 0.05", markerfacecolor=base_analysis.JOURNAL_COLORS["neutral"], alpha=1.0, markersize=10),
        Line2D([0], [0], marker="o", color="w", label="浅色：FDR ≥ 0.05", markerfacecolor=base_analysis.JOURNAL_COLORS["neutral"], alpha=0.42, markersize=10),
        Line2D([0], [0], marker="o", color="w", label="点大小 ∝ 该维度模板数", markerfacecolor="#999999", markeredgecolor=base_analysis.JOURNAL_COLORS["text"], markersize=14),
        Line2D([0, 1], [0, 0], color=base_analysis.JOURNAL_COLORS["neutral"], linewidth=2.0, label="横线：95% 置信区间"),
    ]
    fig.suptitle("图4 | 一级维度森林图（男女被试 × 三种评分条件）", y=0.995, fontsize=17, fontweight="semibold")
    fig.text(
        0.5,
        0.01,
        "每个子图展示同一句模板中女版减男版的平均分差；本图只展示分组结果，不进行男女被试之间的差异检验。",
        ha="center",
        va="bottom",
        fontsize=10,
        color=base_analysis.JOURNAL_COLORS["muted_text"],
    )
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.045), fontsize=10.3)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95), w_pad=2.6, h_pad=2.2)
    base_analysis.save_figure(fig, output_path)
    plt.close(fig)


def write_markdown_summary(
    output_dir: Path,
    input_long_path: Path,
    include_repeat_trials: bool,
    overall_frame: pd.DataFrame,
) -> None:
    """Write a short summary that documents the combined subgroup figure inputs."""
    summary_lines = [
        "# Participant-gender combined forest summary",
        "",
        f"- Input long file: `{input_long_path}`",
        f"- Repeat trials included: `{include_repeat_trials}`",
        "- Figure layout: four figures, each with six subplots (male/female raters x three rating conditions).",
        "- Between-rater-gender significance testing is intentionally omitted in this output.",
        "",
        "## Overall subgroup summary",
        overall_frame.to_string(index=False),
        "",
    ]
    (output_dir / "analysis_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    arguments = parse_arguments()
    output_dir = arguments.output_dir or build_default_output_dir(arguments.input_long_path)
    figure_output_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_output_dir.mkdir(parents=True, exist_ok=True)

    difference_frames: list[pd.DataFrame] = []
    overall_frames: list[pd.DataFrame] = []
    direction_frames: list[pd.DataFrame] = []
    level1_frames: list[pd.DataFrame] = []
    for participant_gender in PARTICIPANT_GENDER_ORDER:
        subgroup_results = run_subgroup_analysis(
            input_long_path=arguments.input_long_path,
            include_repeat_trials=arguments.include_repeat_trials,
            participant_gender=participant_gender,
            bootstrap_iterations=arguments.bootstrap_iterations,
        )
        difference_frames.append(subgroup_results["difference_frame"])
        overall_frames.append(subgroup_results["overall_frame"])
        direction_frames.append(subgroup_results["direction_frame"])
        level1_frames.append(subgroup_results["level1_frame"])

    difference_frame = pd.concat(difference_frames, ignore_index=True)
    overall_frame = pd.concat(overall_frames, ignore_index=True)
    direction_frame = pd.concat(direction_frames, ignore_index=True)
    level1_frame = pd.concat(level1_frames, ignore_index=True)
    difference_frame.to_csv(output_dir / "same_sentence_differences_by_participant_gender.csv", index=False, encoding="utf-8-sig")
    overall_frame.to_csv(output_dir / "stats_overall_by_participant_gender.csv", index=False, encoding="utf-8-sig")
    direction_frame.to_csv(output_dir / "stats_directionality_by_participant_gender.csv", index=False, encoding="utf-8-sig")
    level1_frame.to_csv(output_dir / "stats_level1_by_participant_gender.csv", index=False, encoding="utf-8-sig")
    write_markdown_summary(output_dir, arguments.input_long_path, arguments.include_repeat_trials, overall_frame)
    plot_combined_paired_scores(difference_frame, overall_frame, figure_output_dir / "fig1_participant_gender_paired_scores.png")
    plot_combined_difference_distribution(difference_frame, overall_frame, figure_output_dir / "fig2_participant_gender_difference_distribution.png")
    plot_combined_directionality(direction_frame, figure_output_dir / "fig3_participant_gender_directionality.png")
    plot_combined_level1_forest(level1_frame, figure_output_dir / "fig4_participant_gender_level1_forest.png", arguments.top_level1)

    print("Participant-gender combined figures finished.")
    print(f"Output directory: {output_dir}")
    print("Overall summary:")
    print(
        overall_frame[
            ["participant_gender_label", "condition", "n_templates", "mean_diff_female_minus_male", "wilcoxon_p_fdr"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
