from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


DEFAULT_ANALYSIS_DIR = Path("outputs/group_swap_1b/1b_groupswap_demensionsentence/analysis_1b")
DEFAULT_OUTPUT_DIR = DEFAULT_ANALYSIS_DIR / "figures"
RNG = np.random.default_rng(20260318)
PANEL_ORDER = [
    ("meta-llama_llama-4-maverick", "zeroshot"),
    ("meta-llama_llama-4-maverick", "cot"),
    ("qwen_qwen-2.5-72b-instruct", "zeroshot"),
    ("qwen_qwen-2.5-72b-instruct", "cot"),
]
PANEL_LETTERS = ["A", "B", "C", "D"]
MODEL_LABEL_MAP = {
    "meta-llama_llama-4-maverick": "Llama Maverick",
    "qwen_qwen-2.5-72b-instruct": "Qwen 2.5 72B",
}
SETTING_LABEL_MAP = {
    "zeroshot": "Zero-shot",
    "cot": "CoT",
}
VERSION_ORDER = ["男人版", "女人版"]
VERSION_LABEL_MAP = {
    "男人版": "男版",
    "女人版": "女版",
}
VERSION_COLOR_MAP = {
    "男人版": "#4C607A",
    "女人版": "#B26A6A",
}

JOURNAL_COLORS = {
    "text": "#222222",
    "muted_text": "#6B655E",
    "grid": "#DDD8CF",
    "spine": "#CFC8BE",
    "background": "#FCFBF9",
    "panel_fill": "#F8F6F2",
    "connector": "#B9B1A8",
    "neutral": "#7E7A75",
    "accent_gold": "#C8A96B",
    "black": "#1E1E1E",
    "soft_white": "#FEFDFC",
}
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
DIRECTION_ORDER = ["女人版更高", "两者相同", "男人版更高"]
LEVEL1_LABEL_MAP = {
    "外貌形象攻击": "外貌形象攻击",
    "能力才干攻击": "能力才干攻击",
    "智力理性攻击": "智力理性攻击",
    "情绪稳定攻击": "情绪稳定攻击",
    "道德品行攻击": "道德品行攻击",
    "人际关系攻击": "人际关系攻击",
    "经济资源攻击": "经济资源攻击",
    "社会地位攻击": "社会地位攻击",
    "性化攻击（性羞辱）": "性化攻击（性羞辱）",
    "性别角色/性别表达攻击": "性别角色/性别表达攻击",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-style figures and statistics for 1b group-swap analysis."
    )
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-iterations", type=int, default=3000)
    parser.add_argument("--top-level1", type=int, default=10)
    return parser.parse_args()


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


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def bootstrap_mean_ci(values: np.ndarray, iterations: int) -> tuple[float, float]:
    if values.size == 0:
        return (math.nan, math.nan)
    bootstrap_means = []
    for _ in range(iterations):
        sample = RNG.choice(values, size=values.size, replace=True)
        bootstrap_means.append(float(np.mean(sample)))
    return tuple(np.percentile(bootstrap_means, [2.5, 97.5]))


def cohens_dz(values: np.ndarray) -> float:
    if values.size < 2:
        return math.nan
    standard_deviation = np.std(values, ddof=1)
    if np.isclose(standard_deviation, 0.0):
        return 0.0
    return float(np.mean(values) / standard_deviation)


def model_label(model_prefix: str) -> str:
    return MODEL_LABEL_MAP.get(model_prefix, model_prefix)


def setting_label(setting: str) -> str:
    return SETTING_LABEL_MAP.get(setting, setting)


def version_label(version_name: str) -> str:
    return VERSION_LABEL_MAP.get(version_name, version_name)


def direction_label(direction_name: str) -> str:
    return DIRECTION_LABEL_MAP.get(direction_name, direction_name)


def level1_label(level1_name: str) -> str:
    return LEVEL1_LABEL_MAP.get(level1_name, level1_name)


def configure_matplotlib_fonts() -> None:
    preferred_fonts = [
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
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

    if selected_font is not None:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        print("Warning: No preferred CJK font detected by matplotlib; Chinese text may render as tofu boxes.")

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
    print(f"Matplotlib selected font: {selected_font or 'DejaVu Sans'}")


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
        -0.14,
        1.06,
        letter,
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="center",
        color=JOURNAL_COLORS["text"],
    )


def format_q_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "q = NA"
    if p_value < 0.001:
        return "q < .001"
    return f"q = {p_value:.3f}".replace("0.", ".")


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


def save_axis_as_figure(axis: plt.Axes, output_path: Path) -> None:
    """Save a single matplotlib axis as its own standalone figure file."""
    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    tight_bbox = axis.get_tightbbox(renderer).expanded(1.03, 1.06)
    tight_bbox_inches = tight_bbox.transformed(figure.dpi_scale_trans.inverted())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches=tight_bbox_inches)
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches=tight_bbox_inches)


def prepare_frames(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged_df = pd.read_csv(analysis_dir / "merged_scores_with_dimensions.csv")
    difference_df = pd.read_csv(analysis_dir / "same_sentence_gender_differences.csv")
    topic_df = pd.read_csv(analysis_dir / "topic_rankings.csv")

    difference_df["女减男分差"] = difference_df["女人版攻击性评分"] - difference_df["男人版攻击性评分"]
    difference_df["model_label"] = difference_df["model_prefix"].map(model_label)
    difference_df["setting_label"] = difference_df["setting"].map(setting_label)

    merged_df["model_label"] = merged_df["model_prefix"].map(model_label)
    topic_df["model_label"] = topic_df["model_prefix"].map(model_label)
    topic_df["setting_label"] = topic_df["setting"].map(setting_label)
    return merged_df, difference_df, topic_df


def compute_overall_statistics(difference_df: pd.DataFrame, bootstrap_iterations: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_prefix, setting), group in difference_df.groupby(["model_prefix", "setting"], sort=True):
        diff_values = group["女减男分差"].to_numpy(dtype=float)
        t_result = stats.ttest_1samp(diff_values, popmean=0.0)
        wilcoxon_result = stats.wilcoxon(diff_values, zero_method="wilcox", alternative="two-sided")
        nonzero_values = diff_values[diff_values != 0]
        sign_p = stats.binomtest(int(np.sum(nonzero_values > 0)), n=len(nonzero_values), p=0.5).pvalue if len(nonzero_values) else math.nan
        ci_low, ci_high = bootstrap_mean_ci(diff_values, bootstrap_iterations)
        rows.append(
            {
                "model_prefix": model_prefix,
                "setting": setting,
                "model_label": model_label(model_prefix),
                "setting_label": setting_label(setting),
                "n": len(group),
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
                "sign_test_p": sign_p,
            }
        )
    overall_df = pd.DataFrame(rows).sort_values(["model_label", "setting_label"]).reset_index(drop=True)
    overall_df["t_p_fdr"] = benjamini_hochberg(overall_df["t_p"])
    overall_df["wilcoxon_p_fdr"] = benjamini_hochberg(overall_df["wilcoxon_p"])
    overall_df["sign_test_p_fdr"] = benjamini_hochberg(overall_df["sign_test_p"].fillna(1.0))
    return overall_df


def compute_direction_statistics(difference_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_prefix, setting), group in difference_df.groupby(["model_prefix", "setting"], sort=True):
        counts = group["分差方向"].value_counts()
        female_higher = int(counts.get("女人版更高", 0))
        male_higher = int(counts.get("男人版更高", 0))
        ties = int(counts.get("两者相同", 0))
        discordant = female_higher + male_higher
        sign_p = stats.binomtest(female_higher, n=discordant, p=0.5).pvalue if discordant else math.nan
        rows.append(
            {
                "model_prefix": model_prefix,
                "setting": setting,
                "model_label": model_label(model_prefix),
                "setting_label": setting_label(setting),
                "female_higher": female_higher,
                "ties": ties,
                "male_higher": male_higher,
                "discordant": discordant,
                "female_higher_pct": female_higher / len(group),
                "ties_pct": ties / len(group),
                "male_higher_pct": male_higher / len(group),
                "sign_test_p": sign_p,
            }
        )
    direction_df = pd.DataFrame(rows).sort_values(["model_label", "setting_label"]).reset_index(drop=True)
    direction_df["sign_test_p_fdr"] = benjamini_hochberg(direction_df["sign_test_p"].fillna(1.0))
    return direction_df


def compute_level1_statistics(difference_df: pd.DataFrame, bootstrap_iterations: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_prefix, setting, level1_name), group in difference_df.groupby(["model_prefix", "setting", "一级-攻击领域"], sort=True):
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
                "model_prefix": model_prefix,
                "setting": setting,
                "model_label": model_label(model_prefix),
                "setting_label": setting_label(setting),
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
    level1_df = pd.DataFrame(rows)
    level1_df["t_p_fdr"] = level1_df.groupby(["model_prefix", "setting"])["t_p"].transform(benjamini_hochberg)
    level1_df["wilcoxon_p_fdr"] = level1_df.groupby(["model_prefix", "setting"])["wilcoxon_p"].transform(benjamini_hochberg)
    level1_df = level1_df.sort_values(
        ["model_label", "setting_label", "abs_mean_diff", "一级-攻击领域"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    return level1_df


def plot_figure1_paired_scores(
    difference_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    output_path: Path,
    panel_output_dir: Path | None = None,
) -> None:
    apply_journal_style()
    configure_matplotlib_fonts()
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 12.2), sharey=True)

    for panel_index, (axis, (model_prefix, setting)) in enumerate(zip(axes.flat, PANEL_ORDER)):
        style_axis(axis)
        panel_df = difference_df[
            (difference_df["model_prefix"] == model_prefix) & (difference_df["setting"] == setting)
        ].copy()
        stats_row = overall_df[
            (overall_df["model_prefix"] == model_prefix) & (overall_df["setting"] == setting)
        ].iloc[0]

        male_scores = panel_df["男人版攻击性评分"].to_numpy(dtype=float)
        female_scores = panel_df["女人版攻击性评分"].to_numpy(dtype=float)
        jitter_male = RNG.normal(0.0, 0.022, size=len(panel_df))
        jitter_female = RNG.normal(0.0, 0.022, size=len(panel_df))

        for index in range(len(panel_df)):
            axis.plot(
                [0 + jitter_male[index], 1 + jitter_female[index]],
                [male_scores[index], female_scores[index]],
                color=JOURNAL_COLORS["connector"],
                alpha=0.10,
                linewidth=0.75,
                zorder=1,
            )

        axis.scatter(
            np.full(len(panel_df), 0) + jitter_male,
            male_scores,
            s=13,
            alpha=0.20,
            color=VERSION_COLOR_MAP["男人版"],
            zorder=2,
            linewidths=0,
        )
        axis.scatter(
            np.full(len(panel_df), 1) + jitter_female,
            female_scores,
            s=13,
            alpha=0.20,
            color=VERSION_COLOR_MAP["女人版"],
            zorder=2,
            linewidths=0,
        )

        means = [stats_row["mean_male"], stats_row["mean_female"]]
        lower_errors = [means[0] - np.quantile(male_scores, 0.025), means[1] - np.quantile(female_scores, 0.025)]
        upper_errors = [np.quantile(male_scores, 0.975) - means[0], np.quantile(female_scores, 0.975) - means[1]]
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
            fontsize=9.9,
            color=JOURNAL_COLORS["muted_text"],
            bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.14", facecolor=JOURNAL_COLORS["soft_white"], alpha=0.96, edgecolor="#E7E1D8", linewidth=0.7),
        )
        axis.set_xticks([0, 1], [version_label(name) for name in VERSION_ORDER])
        axis.set_ylim(-0.2, 6.2)
        axis.set_title(f"{model_label(model_prefix)} · {setting_label(setting)}", loc="left", pad=10)
        axis.set_xlabel("")
        axis.set_ylabel("攻击性评分")
        add_panel_letter(axis, PANEL_LETTERS[panel_index])

    fig.suptitle("图1|性别替换句子版本的配对攻击性评分", y=0.992, fontsize=17, fontweight="semibold")
    fig.text(
        0.5,
        0.012,
        "每条浅线连接同一句子的男版与女版；黑色均值点与误差线用于强调总体趋势。",
        ha="center",
        va="bottom",
        fontsize=10,
        color=JOURNAL_COLORS["muted_text"],
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.968), w_pad=2.3, h_pad=2.5)
    save_figure(fig, output_path)
    if panel_output_dir is not None:
        for panel_index, axis in enumerate(axes.flat, start=1):
            save_axis_as_figure(axis, panel_output_dir / f"fig1_paired_scores_panel_{panel_index}.png")
    plt.close(fig)


def plot_figure2_difference_distribution(
    difference_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    output_path: Path,
    panel_output_dir: Path | None = None,
) -> None:
    apply_journal_style()
    configure_matplotlib_fonts()
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 11.8), sharex=True, sharey=True)
    for panel_index, (axis, (model_prefix, setting)) in enumerate(zip(axes.flat, PANEL_ORDER)):
        style_axis(axis)
        panel_df = difference_df[
            (difference_df["model_prefix"] == model_prefix) & (difference_df["setting"] == setting)
        ]
        stats_row = overall_df[
            (overall_df["model_prefix"] == model_prefix) & (overall_df["setting"] == setting)
        ].iloc[0]
        values = panel_df["女减男分差"].to_numpy(dtype=float)
        bins = np.arange(values.min() - 0.25, values.max() + 0.75, 0.5)
        axis.hist(values, bins=bins, density=True, color=JOURNAL_COLORS["accent_gold"], alpha=0.55, edgecolor="white", linewidth=0.8)
        if len(np.unique(values)) > 1:
            sns.kdeplot(x=values, ax=axis, color=JOURNAL_COLORS["black"], linewidth=1.8, fill=False, bw_adjust=0.9, clip=(values.min(), values.max()))
        axis.axvline(0.0, color=JOURNAL_COLORS["neutral"], linestyle="--", linewidth=1.15)
        axis.axvline(stats_row["mean_diff_female_minus_male"], color=VERSION_COLOR_MAP["女人版"], linestyle="-", linewidth=1.9)
        axis.set_title(f"{model_label(model_prefix)} · {setting_label(setting)}", loc="left", pad=10)
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
            fontsize=9.9,
            color=JOURNAL_COLORS["muted_text"],
            bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.14", facecolor=JOURNAL_COLORS["soft_white"], alpha=0.96, edgecolor="#E7E1D8", linewidth=0.7),
        )
        add_panel_letter(axis, PANEL_LETTERS[panel_index])
    fig.suptitle("图2|配对分差分布（女版 - 男版）", y=0.992, fontsize=17, fontweight="semibold")
    fig.text(
        0.5,
        0.012,
        "半透明直方图显示经验分布，黑色密度曲线用于增强整体形状感；虚线为零差异，红棕线为平均分差。",
        ha="center",
        va="bottom",
        fontsize=10,
        color=JOURNAL_COLORS["muted_text"],
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.968), w_pad=2.3, h_pad=2.5)
    save_figure(fig, output_path)
    if panel_output_dir is not None:
        for panel_index, axis in enumerate(axes.flat, start=1):
            save_axis_as_figure(axis, panel_output_dir / f"fig2_difference_distribution_panel_{panel_index}.png")
    plt.close(fig)


def plot_figure3_directionality(direction_df: pd.DataFrame, output_path: Path) -> None:
    apply_journal_style()
    configure_matplotlib_fonts()
    ordered_df = direction_df.copy()
    ordered_df["panel_label"] = ordered_df["model_label"] + "\n" + ordered_df["setting_label"]

    fig, ax = plt.subplots(figsize=(12.6, 8.6))
    style_axis(ax)
    bottoms = np.zeros(len(ordered_df))
    x_positions = np.arange(len(ordered_df))

    direction_column_map = {"女人版更高": "female_higher", "两者相同": "ties", "男人版更高": "male_higher"}
    for direction_name in DIRECTION_ORDER:
        values = ordered_df[f"{direction_column_map[direction_name]}_pct"].to_numpy(dtype=float)
        ax.bar(
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

    for idx, row in ordered_df.iterrows():
        annotation = (
            f"女>男 {row['female_higher']} · 相同 {row['ties']} · 男>女 {row['male_higher']}\n"
            f"{format_q_value(row['sign_test_p_fdr'])} {significance_stars(row['sign_test_p_fdr'])}"
        )
        ax.text(idx, 1.03, annotation, ha="center", va="bottom", fontsize=9.2, color=JOURNAL_COLORS["muted_text"])

    ax.set_xticks(x_positions, ordered_df["panel_label"].tolist())
    ax.set_ylabel("句子对占比")
    ax.set_ylim(0, 1.2)
    ax.set_title("图3|不同模型与提示设置下的分差方向", loc="left", pad=12, fontsize=16, fontweight="semibold")
    legend = ax.legend(title="分差方向", frameon=True, loc="upper right", ncol=1, borderpad=0.8, labelspacing=0.6)
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
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    save_figure(fig, output_path)
    plt.close(fig)


def plot_figure4_level1_forest(
    level1_df: pd.DataFrame,
    output_path: Path,
    top_level1: int,
    panel_output_dir: Path | None = None,
) -> None:
    apply_journal_style()
    configure_matplotlib_fonts()
    fig, axes = plt.subplots(2, 2, figsize=(18.4, 14.2), sharex=True)

    for panel_index, (axis, (model_prefix, setting)) in enumerate(zip(axes.flat, PANEL_ORDER)):
        style_axis(axis)
        panel_df = level1_df[
            (level1_df["model_prefix"] == model_prefix) & (level1_df["setting"] == setting)
        ].copy()
        panel_df = panel_df.nlargest(top_level1, columns="abs_mean_diff").sort_values("mean_diff_female_minus_male")
        y_positions = np.arange(len(panel_df))
        for y_pos, (_, row) in zip(y_positions, panel_df.iterrows()):
            point_color = VERSION_COLOR_MAP["女人版"] if row["mean_diff_female_minus_male"] >= 0 else VERSION_COLOR_MAP["男人版"]
            point_alpha = 1.0 if row["wilcoxon_p_fdr"] < 0.05 else 0.42
            axis.plot([row["ci_low"], row["ci_high"]], [y_pos, y_pos], color=JOURNAL_COLORS["neutral"], linewidth=1.9, alpha=0.85)
            axis.scatter(
                row["mean_diff_female_minus_male"],
                y_pos,
                s=28 + row["n"] * 4,
                color=point_color,
                alpha=point_alpha,
                edgecolor=JOURNAL_COLORS["text"],
                linewidth=0.45,
                zorder=3,
            )
        labels = [f"{level1_label(name)} ({n})" for name, n in zip(panel_df["一级-攻击领域"], panel_df["n"])]
        axis.set_yticks(y_positions, labels)
        axis.axvline(0.0, color=JOURNAL_COLORS["neutral"], linestyle="--", linewidth=1.2)
        axis.set_title(f"{model_label(model_prefix)} · {setting_label(setting)}", loc="left", pad=10)
        axis.set_xlabel("平均分差（女版 - 男版）")
        axis.set_ylabel("一级维度")
        add_panel_letter(axis, PANEL_LETTERS[panel_index])

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="红棕点：女版评分更高", markerfacecolor=VERSION_COLOR_MAP["女人版"], markeredgecolor=JOURNAL_COLORS["text"], markersize=10),
        Line2D([0], [0], marker="o", color="w", label="蓝灰点：男版评分更高", markerfacecolor=VERSION_COLOR_MAP["男人版"], markeredgecolor=JOURNAL_COLORS["text"], markersize=10),
        Line2D([0], [0], marker="o", color="w", label="深色：FDR < 0.05", markerfacecolor=JOURNAL_COLORS["neutral"], alpha=1.0, markersize=10),
        Line2D([0], [0], marker="o", color="w", label="浅色：FDR ≥ 0.05", markerfacecolor=JOURNAL_COLORS["neutral"], alpha=0.42, markersize=10),
        Line2D([0], [0], marker="o", color="w", label="点大小 ∝ 该维度样本数", markerfacecolor="#999999", markeredgecolor=JOURNAL_COLORS["text"], markersize=14),
        Line2D([0, 1], [0, 0], color=JOURNAL_COLORS["neutral"], linewidth=2.0, label="横线：95% 置信区间"),
    ]

    fig.suptitle("图4|一级维度效应及其 95% 置信区间（按绝对效应排序）", y=0.992, fontsize=17, fontweight="semibold")
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.012), fontsize=10.3)
    fig.tight_layout(rect=(0, 0.075, 1, 0.968), w_pad=2.4, h_pad=2.4)
    save_figure(fig, output_path)
    if panel_output_dir is not None:
        for panel_index, axis in enumerate(axes.flat, start=1):
            save_axis_as_figure(axis, panel_output_dir / f"fig4_level1_forest_panel_{panel_index}.png")
    plt.close(fig)


def save_statistics_tables(output_dir: Path, overall_df: pd.DataFrame, direction_df: pd.DataFrame, level1_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_df.to_csv(output_dir / "stats_overall.csv", index=False)
    direction_df.to_csv(output_dir / "stats_directionality.csv", index=False)
    level1_df.to_csv(output_dir / "stats_level1.csv", index=False)


def main() -> None:
    configure_matplotlib_fonts()
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    panel_output_dir = arguments.output_dir / "panels"

    _, difference_df, _ = prepare_frames(arguments.analysis_dir)
    overall_df = compute_overall_statistics(difference_df, arguments.bootstrap_iterations)
    direction_df = compute_direction_statistics(difference_df)
    level1_df = compute_level1_statistics(difference_df, arguments.bootstrap_iterations)
    save_statistics_tables(arguments.output_dir, overall_df, direction_df, level1_df)

    plot_figure1_paired_scores(
        difference_df,
        overall_df,
        arguments.output_dir / "fig1_paired_scores.png",
        panel_output_dir=panel_output_dir,
    )
    plot_figure2_difference_distribution(
        difference_df,
        overall_df,
        arguments.output_dir / "fig2_difference_distribution.png",
        panel_output_dir=panel_output_dir,
    )
    plot_figure3_directionality(direction_df, arguments.output_dir / "fig3_directionality.png")
    plot_figure4_level1_forest(
        level1_df,
        arguments.output_dir / "fig4_level1_forest.png",
        arguments.top_level1,
        panel_output_dir=panel_output_dir,
    )

    print("Plotting finished.")
    print(f"Output directory: {arguments.output_dir}")
    print("Generated files:")
    for output_path in sorted(arguments.output_dir.iterdir()):
        print(f"  - {output_path.name}")


if __name__ == "__main__":
    main()
