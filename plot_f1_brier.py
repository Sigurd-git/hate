from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = PROJECT_ROOT / "9_model_f1_brier_skip_missing.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_SCATTER_FIGURE = DEFAULT_OUTPUT_DIR / "hate_f1_brier_2x2_scatter.png"
DEFAULT_SCATTER_VECTOR = DEFAULT_OUTPUT_DIR / "hate_f1_brier_2x2_scatter.pdf"
DEFAULT_F1_BAR_FIGURE = DEFAULT_OUTPUT_DIR / "hate_f1_2x2_bars.png"
DEFAULT_F1_BAR_VECTOR = DEFAULT_OUTPUT_DIR / "hate_f1_2x2_bars.pdf"
DEFAULT_BRIER_BAR_FIGURE = DEFAULT_OUTPUT_DIR / "hate_brier_2x2_bars.png"
DEFAULT_BRIER_BAR_VECTOR = DEFAULT_OUTPUT_DIR / "hate_brier_2x2_bars.pdf"
DEFAULT_F1_DUMBBELL_FIGURE = DEFAULT_OUTPUT_DIR / "hate_f1_2x2_dumbbell.png"
DEFAULT_F1_DUMBBELL_VECTOR = DEFAULT_OUTPUT_DIR / "hate_f1_2x2_dumbbell.pdf"
DEFAULT_BRIER_DUMBBELL_FIGURE = DEFAULT_OUTPUT_DIR / "hate_brier_2x2_dumbbell.png"
DEFAULT_BRIER_DUMBBELL_VECTOR = DEFAULT_OUTPUT_DIR / "hate_brier_2x2_dumbbell.pdf"
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "hate_f1_brier_best_models.csv"

PANEL_ORDER = [
    ("zh", "zeroshot", "Chinese · Zero-shot"),
    ("zh", "cot", "Chinese · CoT"),
    ("en", "zeroshot", "English · Zero-shot"),
    ("en", "cot", "English · CoT"),
]

MODEL_LABELS = {
    "chatgpt5.1": "ChatGPT-5.1",
    "claude4.5": "Claude-4.5",
    "deepseek_deepseek-r1-0528": "DeepSeek-R1",
    "deepseek_deepseek-v3.2-exp": "DeepSeek-V3.2",
    "google_gemma-4-31b-it": "Gemma-4-31B",
    "meta-llama_llama-4-maverick": "Llama-4-Maverick",
    "moonshotai_kimi-k2-thinking": "Kimi-K2",
    "qwen_qwen-2.5-72b-instruct": "Qwen-2.5-72B",
    "z-ai_glm-4.6": "GLM-4.6",
}

MUTED_MODEL_COLORS = {
    "ChatGPT-5.1": "#5B7FA3",
    "Claude-4.5": "#C9736B",
    "DeepSeek-R1": "#74AFA7",
    "DeepSeek-V3.2": "#739A63",
    "Gemma-4-31B": "#8F6FB5",
    "GLM-4.6": "#9C7C67",
    "Kimi-K2": "#C9AD58",
    "Llama-4-Maverick": "#D28B47",
    "Qwen-2.5-72B": "#7B8794",
}

EXCLUDED_MODEL_NAMES = {
    "baidu_ernie-4.5-21b-a3b",
}

BASE_FONT_FAMILY = "DejaVu Sans"
GRID_COLOR = "#E8EDF3"
SPINE_COLOR = "#CCD4DD"
TEXT_COLOR = "#22303C"
SUBTLE_TEXT_COLOR = "#607080"
BACKGROUND_COLOR = "#FFFFFF"
BEST_HIGHLIGHT_EDGE = "#18222C"
BEST_LABEL_FACE = "#F6F8FB"
CONNECTOR_COLOR = "#C8D0D9"
ZERO_SHOT_MARKER = "o"
COT_MARKER = "s"
ZH_MARKER = "o"
EN_MARKER = "D"

DUMBBELL_LAYOUTS = [
    {
        "panel_key": "zh_prompting",
        "title": "Chinese · Zero-shot vs CoT",
        "fixed_column": "language",
        "fixed_value": "zh",
        "compare_column": "setting",
        "left_value": "zeroshot",
        "right_value": "cot",
        "left_label": "Zero-shot",
        "right_label": "CoT",
        "marker_map": {"zeroshot": ZERO_SHOT_MARKER, "cot": COT_MARKER},
    },
    {
        "panel_key": "en_prompting",
        "title": "English · Zero-shot vs CoT",
        "fixed_column": "language",
        "fixed_value": "en",
        "compare_column": "setting",
        "left_value": "zeroshot",
        "right_value": "cot",
        "left_label": "Zero-shot",
        "right_label": "CoT",
        "marker_map": {"zeroshot": ZERO_SHOT_MARKER, "cot": COT_MARKER},
    },
    {
        "panel_key": "zeroshot_language",
        "title": "Zero-shot · Chinese vs English",
        "fixed_column": "setting",
        "fixed_value": "zeroshot",
        "compare_column": "language",
        "left_value": "zh",
        "right_value": "en",
        "left_label": "Chinese",
        "right_label": "English",
        "marker_map": {"zh": ZH_MARKER, "en": EN_MARKER},
    },
    {
        "panel_key": "cot_language",
        "title": "CoT · Chinese vs English",
        "fixed_column": "setting",
        "fixed_value": "cot",
        "compare_column": "language",
        "left_value": "zh",
        "right_value": "en",
        "left_label": "Chinese",
        "right_label": "English",
        "marker_map": {"zh": ZH_MARKER, "en": EN_MARKER},
    },
]


def configure_plot_theme() -> None:
    """Apply one restrained, publication-style visual theme to all figures."""
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "font.family": BASE_FONT_FAMILY,
            "axes.facecolor": BACKGROUND_COLOR,
            "figure.facecolor": BACKGROUND_COLOR,
            "axes.edgecolor": SPINE_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": SUBTLE_TEXT_COLOR,
            "ytick.color": SUBTLE_TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.65,
            "axes.linewidth": 0.8,
            "axes.titleweight": "semibold",
            "axes.titlesize": 13.0,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.6,
            "legend.fontsize": 9.2,
            "legend.title_fontsize": 10.0,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def load_metrics_table(csv_path: Path) -> pd.DataFrame:
    """Load and validate the metrics table used by the plots."""
    dataframe = pd.read_csv(csv_path)

    required_columns = {"language", "model", "setting", "n_scored", "f1", "brier"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    dataframe = dataframe.copy()
    dataframe = dataframe.loc[
        ~dataframe["model"].astype(str).isin(EXCLUDED_MODEL_NAMES)
    ].copy()
    dataframe["model_label"] = dataframe["model"].map(MODEL_LABELS).fillna(dataframe["model"])
    dataframe["language"] = dataframe["language"].str.lower()
    dataframe["setting"] = dataframe["setting"].str.lower()
    dataframe = dataframe.sort_values(["language", "setting", "model_label"]).reset_index(drop=True)
    return dataframe


def build_color_mapping(model_names: Iterable[str]) -> dict[str, str]:
    """Assign one stable muted color per model."""
    model_name_list = sorted(model_names)
    color_mapping: dict[str, str] = {}
    fallback_palette = sns.color_palette("crest", n_colors=max(len(model_name_list), 3)).as_hex()
    for index, model_name in enumerate(model_name_list):
        color_mapping[model_name] = MUTED_MODEL_COLORS.get(model_name, fallback_palette[index])
    return color_mapping


def build_legend_handles(color_mapping: dict[str, str]) -> list[Line2D]:
    """Create a consistent legend shared by figure types that need one."""
    handles: list[Line2D] = []
    for model_label in sorted(color_mapping):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color_mapping[model_label],
                markeredgecolor="white",
                markeredgewidth=0.75,
                markersize=7.2,
                label=model_label,
            )
        )
    return handles


def build_condition_legend_handles(left_label: str, right_label: str, left_marker: str, right_marker: str) -> list[Line2D]:
    """Create a compact legend explaining the two compared conditions."""
    return [
        Line2D(
            [0],
            [0],
            marker=left_marker,
            color=CONNECTOR_COLOR,
            markerfacecolor="white",
            markeredgecolor=BEST_HIGHLIGHT_EDGE,
            markeredgewidth=0.9,
            markersize=6.8,
            linewidth=1.1,
            label=left_label,
        ),
        Line2D(
            [0],
            [0],
            marker=right_marker,
            color=CONNECTOR_COLOR,
            markerfacecolor="white",
            markeredgecolor=BEST_HIGHLIGHT_EDGE,
            markeredgewidth=0.9,
            markersize=6.8,
            linewidth=1.1,
            label=right_label,
        ),
    ]


def compute_best_models(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return the best F1 and best Brier model per language/setting group."""
    summary_rows: list[dict[str, object]] = []

    for language_code, setting_name, panel_title in PANEL_ORDER:
        panel_dataframe = dataframe[
            (dataframe["language"] == language_code) & (dataframe["setting"] == setting_name)
        ].copy()
        if panel_dataframe.empty:
            continue

        best_f1_row = panel_dataframe.loc[panel_dataframe["f1"].idxmax()]
        best_brier_row = panel_dataframe.loc[panel_dataframe["brier"].idxmin()]
        summary_rows.append(
            {
                "language": language_code,
                "setting": setting_name,
                "panel_title": panel_title,
                "n_models": int(len(panel_dataframe)),
                "best_f1_model": best_f1_row["model_label"],
                "best_f1": float(best_f1_row["f1"]),
                "best_brier_model": best_brier_row["model_label"],
                "best_brier": float(best_brier_row["brier"]),
                "same_model_best_both": bool(best_f1_row["model"] == best_brier_row["model"]),
            }
        )

    return pd.DataFrame(summary_rows)


def style_axis(axis: plt.Axes) -> None:
    """Apply restrained spine and grid styling to one axis."""
    axis.grid(True, axis="both", color=GRID_COLOR, linewidth=0.65, alpha=0.95)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE_COLOR)
    axis.spines["bottom"].set_color(SPINE_COLOR)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)


def add_best_direction_hint(axis: plt.Axes) -> None:
    """Add a subtle corner cue indicating the preferred direction."""
    arrow = FancyArrowPatch(
        (0.16, 0.20),
        (0.07, 0.78),
        arrowstyle="-|>",
        mutation_scale=9,
        lw=0.85,
        color="#93A0AD",
        alpha=0.55,
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.add_patch(arrow)
    axis.text(
        0.073,
        0.80,
        "better",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#7C8995",
        alpha=0.95,
    )


def format_numeric_axis(axis: plt.Axes, decimals: int = 2) -> None:
    """Use a compact and publication-friendly numeric tick format."""
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.xaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))


def save_figure_pair(figure: plt.Figure, png_path: Path, pdf_path: Path) -> None:
    """Save both raster and vector versions for paper and slides."""
    figure.savefig(png_path, dpi=320, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
    figure.savefig(pdf_path, bbox_inches="tight", facecolor=BACKGROUND_COLOR)


def create_scatter_figure(
    dataframe: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
    color_mapping: dict[str, str],
) -> tuple[Path, dict[str, int]]:
    """Create the 2x2 scatter figure with a compact right-side legend."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    configure_plot_theme()

    figure, axes = plt.subplots(2, 2, figsize=(15.2, 12.2), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    x_min = max(0.0, dataframe["brier"].min() - 0.012)
    x_max = dataframe["brier"].max() + 0.014
    y_min = max(0.0, dataframe["f1"].min() - 0.028)
    y_max = min(1.0, dataframe["f1"].max() + 0.028)

    panel_counts: dict[str, int] = {}

    for axis, (language_code, setting_name, panel_title) in zip(axes_flat, PANEL_ORDER):
        panel_dataframe = dataframe[
            (dataframe["language"] == language_code) & (dataframe["setting"] == setting_name)
        ].copy()
        panel_counts[f"{language_code}_{setting_name}"] = int(len(panel_dataframe))

        for _, row in panel_dataframe.iterrows():
            axis.scatter(
                row["brier"],
                row["f1"],
                s=110,
                color="#F7F9FB",
                edgecolors="none",
                zorder=2,
            )
            axis.scatter(
                row["brier"],
                row["f1"],
                s=82,
                color=color_mapping[row["model_label"]],
                edgecolors="white",
                linewidths=0.9,
                alpha=0.98,
                zorder=3,
            )

        axis.set_title(panel_title, pad=8, color=TEXT_COLOR)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_xlabel("Brier score ↓")
        axis.set_ylabel("F1 ↑")
        axis.set_box_aspect(1)
        format_numeric_axis(axis, decimals=2)
        style_axis(axis)
        add_best_direction_hint(axis)

    legend_handles = build_legend_handles(color_mapping)
    figure.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.855, 0.52),
        frameon=False,
        title="Models",
        borderaxespad=0.0,
        labelspacing=0.72,
        handletextpad=0.55,
        handlelength=0.9,
    )
    figure.suptitle(
        "Model performance across language and prompting settings",
        fontsize=16.6,
        fontweight="semibold",
        y=0.972,
        color=TEXT_COLOR,
    )
    figure.text(
        0.425,
        0.038,
        "Performance improves toward the upper-left: lower Brier and higher F1.",
        ha="center",
        fontsize=9.6,
        color=SUBTLE_TEXT_COLOR,
    )
    figure.tight_layout(rect=(0.03, 0.065, 0.83, 0.94))
    save_figure_pair(figure, png_path, pdf_path)
    plt.close(figure)
    return png_path, panel_counts


def annotate_best_bar(
    axis: plt.Axes,
    best_value: float,
    best_y_position: int,
    x_limits: tuple[float, float],
) -> None:
    """Annotate only the best bar to keep the chart clean and informative."""
    x_min, x_max = x_limits
    offset = (x_max - x_min) * 0.016
    text_x = min(best_value + offset, x_max - offset * 0.4)
    axis.text(
        text_x,
        best_y_position,
        f"{best_value:.3f}",
        va="center",
        ha="left",
        fontsize=9.1,
        color=BEST_HIGHLIGHT_EDGE,
        fontweight="semibold",
        bbox={
            "boxstyle": "round,pad=0.18,rounding_size=0.12",
            "facecolor": BEST_LABEL_FACE,
            "edgecolor": "none",
            "alpha": 0.95,
        },
    )


def create_single_metric_bar_figure(
    dataframe: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
    color_mapping: dict[str, str],
    metric_column: str,
    metric_title: str,
    direction_label: str,
    larger_is_better: bool,
) -> tuple[Path, dict[str, int]]:
    """Create a polished 2x2 bar figure for one metric only."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    configure_plot_theme()

    figure, axes = plt.subplots(2, 2, figsize=(15.2, 11.0), sharex=True, sharey=False)
    axes_flat = axes.flatten()

    pad = 0.028 if metric_column == "f1" else 0.012
    upper_pad = 0.024 if metric_column == "f1" else 0.014
    x_limits = (max(0.0, dataframe[metric_column].min() - pad), min(1.0, dataframe[metric_column].max() + upper_pad))
    panel_counts: dict[str, int] = {}

    for axis, (language_code, setting_name, panel_title) in zip(axes_flat, PANEL_ORDER):
        panel_dataframe = dataframe[
            (dataframe["language"] == language_code) & (dataframe["setting"] == setting_name)
        ].copy()
        panel_counts[f"{language_code}_{setting_name}"] = int(len(panel_dataframe))
        panel_dataframe = panel_dataframe.sort_values(metric_column, ascending=not larger_is_better).reset_index(drop=True)

        model_labels = panel_dataframe["model_label"].tolist()
        y_positions = list(range(len(panel_dataframe)))
        best_model_label = model_labels[0]

        bar_colors = []
        bar_alphas = []
        for model_label in model_labels:
            bar_colors.append(color_mapping[model_label])
            bar_alphas.append(1.0 if model_label == best_model_label else 0.78)

        bars = axis.barh(
            y_positions,
            panel_dataframe[metric_column],
            color=bar_colors,
            edgecolor="white",
            linewidth=0.8,
            height=0.56,
        )

        for bar, alpha_value in zip(bars, bar_alphas):
            bar.set_alpha(alpha_value)

        best_bar = bars[0]
        best_bar.set_edgecolor(BEST_HIGHLIGHT_EDGE)
        best_bar.set_linewidth(1.15)

        axis.set_title(panel_title, pad=8, color=TEXT_COLOR)
        axis.set_xlim(*x_limits)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(model_labels, fontsize=9.4)
        axis.invert_yaxis()
        axis.set_xlabel(direction_label)
        format_numeric_axis(axis, decimals=2)
        style_axis(axis)
        axis.grid(False, axis="y")
        annotate_best_bar(
            axis=axis,
            best_value=float(panel_dataframe.iloc[0][metric_column]),
            best_y_position=0,
            x_limits=x_limits,
        )

    figure.suptitle(
        metric_title,
        fontsize=16.6,
        fontweight="semibold",
        y=0.972,
        color=TEXT_COLOR,
    )
    figure.text(
        0.5,
        0.035,
        "Best-performing model in each panel is outlined and numerically annotated.",
        ha="center",
        fontsize=9.4,
        color=SUBTLE_TEXT_COLOR,
    )
    figure.tight_layout(rect=(0.03, 0.065, 0.995, 0.94))
    save_figure_pair(figure, png_path, pdf_path)
    plt.close(figure)
    return png_path, panel_counts


def build_dumbbell_panel_dataframe(
    dataframe: pd.DataFrame,
    layout: dict[str, object],
    metric_column: str,
    larger_is_better: bool,
) -> pd.DataFrame:
    """Construct one panel table where each row is a model with two matched conditions."""
    fixed_column = str(layout["fixed_column"])
    fixed_value = str(layout["fixed_value"])
    compare_column = str(layout["compare_column"])
    left_value = str(layout["left_value"])
    right_value = str(layout["right_value"])

    left_dataframe = dataframe[
        (dataframe[fixed_column] == fixed_value) & (dataframe[compare_column] == left_value)
    ][["model", "model_label", metric_column]].rename(columns={metric_column: "left_metric"})
    right_dataframe = dataframe[
        (dataframe[fixed_column] == fixed_value) & (dataframe[compare_column] == right_value)
    ][["model", metric_column]].rename(columns={metric_column: "right_metric"})

    panel_dataframe = left_dataframe.merge(right_dataframe, on="model", how="inner")
    panel_dataframe["change"] = panel_dataframe["right_metric"] - panel_dataframe["left_metric"]
    panel_dataframe = panel_dataframe.sort_values(
        ["right_metric", "left_metric"],
        ascending=not larger_is_better,
    ).reset_index(drop=True)
    return panel_dataframe


def annotate_dumbbell_delta(
    axis: plt.Axes,
    x_position: float,
    y_position: int,
    delta_value: float,
    metric_column: str,
    x_limits: tuple[float, float],
) -> None:
    """Annotate the largest absolute change in each panel with a compact delta label."""
    x_min, x_max = x_limits
    offset = (x_max - x_min) * 0.018
    horizontal_alignment = "left"
    text_x = min(x_position + offset, x_max - offset * 0.4)
    if x_position > x_max - 2.5 * offset:
        text_x = x_position - offset
        horizontal_alignment = "right"

    delta_prefix = "+" if delta_value >= 0 else "−"
    delta_text = f"Δ{metric_column.upper()} {delta_prefix}{abs(delta_value):.3f}"
    axis.text(
        text_x,
        y_position,
        delta_text,
        va="center",
        ha=horizontal_alignment,
        fontsize=8.8,
        color=BEST_HIGHLIGHT_EDGE,
        fontweight="semibold",
        bbox={
            "boxstyle": "round,pad=0.16,rounding_size=0.12",
            "facecolor": BEST_LABEL_FACE,
            "edgecolor": "none",
            "alpha": 0.96,
        },
    )


def create_dumbbell_figure(
    dataframe: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
    color_mapping: dict[str, str],
    metric_column: str,
    metric_title: str,
    larger_is_better: bool,
) -> tuple[Path, dict[str, int]]:
    """Create a 2x2 dumbbell figure for within-model comparisons across paired conditions."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    configure_plot_theme()

    figure, axes = plt.subplots(2, 2, figsize=(15.2, 11.2), sharex=True, sharey=False)
    axes_flat = axes.flatten()

    pad = 0.028 if metric_column == "f1" else 0.012
    upper_pad = 0.024 if metric_column == "f1" else 0.014
    x_limits = (max(0.0, dataframe[metric_column].min() - pad), min(1.0, dataframe[metric_column].max() + upper_pad))
    panel_counts: dict[str, int] = {}

    for axis, layout in zip(axes_flat, DUMBBELL_LAYOUTS):
        panel_dataframe = build_dumbbell_panel_dataframe(
            dataframe=dataframe,
            layout=layout,
            metric_column=metric_column,
            larger_is_better=larger_is_better,
        )
        panel_counts[str(layout["panel_key"])] = int(len(panel_dataframe))
        y_positions = list(range(len(panel_dataframe)))

        for y_position, (_, row) in zip(y_positions, panel_dataframe.iterrows()):
            axis.plot(
                [row["left_metric"], row["right_metric"]],
                [y_position, y_position],
                color=CONNECTOR_COLOR,
                linewidth=1.4,
                solid_capstyle="round",
                zorder=1,
            )

        for y_position, (_, row) in zip(y_positions, panel_dataframe.iterrows()):
            left_marker = layout["marker_map"][str(layout["left_value"])]
            right_marker = layout["marker_map"][str(layout["right_value"])]
            axis.scatter(
                row["left_metric"],
                y_position,
                s=64,
                marker=left_marker,
                color=color_mapping[row["model_label"]],
                edgecolors="white",
                linewidths=0.9,
                zorder=3,
            )
            axis.scatter(
                row["right_metric"],
                y_position,
                s=64,
                marker=right_marker,
                color=color_mapping[row["model_label"]],
                edgecolors="white",
                linewidths=0.9,
                zorder=4,
            )

        largest_change_index = panel_dataframe["change"].abs().idxmax()
        largest_change_row = panel_dataframe.loc[largest_change_index]
        annotate_dumbbell_delta(
            axis=axis,
            x_position=max(largest_change_row["left_metric"], largest_change_row["right_metric"]),
            y_position=int(panel_dataframe.index.get_loc(largest_change_index)),
            delta_value=float(largest_change_row["change"]),
            metric_column=metric_column,
            x_limits=x_limits,
        )

        axis.set_title(str(layout["title"]), pad=8, color=TEXT_COLOR)
        axis.set_xlim(*x_limits)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(panel_dataframe["model_label"].tolist(), fontsize=9.4)
        axis.invert_yaxis()
        axis.set_xlabel("F1" if metric_column == "f1" else "Brier score")
        format_numeric_axis(axis, decimals=2)
        style_axis(axis)
        axis.grid(False, axis="y")

        condition_handles = build_condition_legend_handles(
            left_label=str(layout["left_label"]),
            right_label=str(layout["right_label"]),
            left_marker=layout["marker_map"][str(layout["left_value"])],
            right_marker=layout["marker_map"][str(layout["right_value"])],
        )
        axis.legend(
            handles=condition_handles,
            loc="lower right",
            frameon=False,
            ncol=2,
            handletextpad=0.4,
            columnspacing=1.0,
            borderaxespad=0.25,
        )

    figure.suptitle(
        metric_title,
        fontsize=16.6,
        fontweight="semibold",
        y=0.972,
        color=TEXT_COLOR,
    )
    figure.text(
        0.5,
        0.035,
        "Each row is one model; the connector shows within-model change between two matched conditions.",
        ha="center",
        fontsize=9.4,
        color=SUBTLE_TEXT_COLOR,
    )
    figure.tight_layout(rect=(0.03, 0.065, 0.995, 0.94))
    save_figure_pair(figure, png_path, pdf_path)
    plt.close(figure)
    return png_path, panel_counts


def main() -> None:
    dataframe = load_metrics_table(DEFAULT_INPUT_CSV)
    color_mapping = build_color_mapping(dataframe["model_label"].unique())
    summary_dataframe = compute_best_models(dataframe)

    scatter_path, scatter_panel_counts = create_scatter_figure(
        dataframe=dataframe,
        png_path=DEFAULT_SCATTER_FIGURE,
        pdf_path=DEFAULT_SCATTER_VECTOR,
        color_mapping=color_mapping,
    )
    f1_bar_path, f1_bar_panel_counts = create_single_metric_bar_figure(
        dataframe=dataframe,
        png_path=DEFAULT_F1_BAR_FIGURE,
        pdf_path=DEFAULT_F1_BAR_VECTOR,
        color_mapping=color_mapping,
        metric_column="f1",
        metric_title="F1 across language and prompting settings",
        direction_label="F1 ↑",
        larger_is_better=True,
    )
    brier_bar_path, brier_bar_panel_counts = create_single_metric_bar_figure(
        dataframe=dataframe,
        png_path=DEFAULT_BRIER_BAR_FIGURE,
        pdf_path=DEFAULT_BRIER_BAR_VECTOR,
        color_mapping=color_mapping,
        metric_column="brier",
        metric_title="Brier score across language and prompting settings",
        direction_label="Brier score ↓",
        larger_is_better=False,
    )
    f1_dumbbell_path, f1_dumbbell_panel_counts = create_dumbbell_figure(
        dataframe=dataframe,
        png_path=DEFAULT_F1_DUMBBELL_FIGURE,
        pdf_path=DEFAULT_F1_DUMBBELL_VECTOR,
        color_mapping=color_mapping,
        metric_column="f1",
        metric_title="Within-model F1 comparisons across prompting and language conditions",
        larger_is_better=True,
    )
    brier_dumbbell_path, brier_dumbbell_panel_counts = create_dumbbell_figure(
        dataframe=dataframe,
        png_path=DEFAULT_BRIER_DUMBBELL_FIGURE,
        pdf_path=DEFAULT_BRIER_DUMBBELL_VECTOR,
        color_mapping=color_mapping,
        metric_column="brier",
        metric_title="Within-model Brier comparisons across prompting and language conditions",
        larger_is_better=False,
    )
    summary_dataframe.to_csv(DEFAULT_SUMMARY_CSV, index=False)

    print(f"Saved scatter figure: {scatter_path}")
    print(f"Saved scatter vector: {DEFAULT_SCATTER_VECTOR}")
    print(f"Saved F1 bar figure: {f1_bar_path}")
    print(f"Saved F1 bar vector: {DEFAULT_F1_BAR_VECTOR}")
    print(f"Saved Brier bar figure: {brier_bar_path}")
    print(f"Saved Brier bar vector: {DEFAULT_BRIER_BAR_VECTOR}")
    print(f"Saved F1 dumbbell figure: {f1_dumbbell_path}")
    print(f"Saved F1 dumbbell vector: {DEFAULT_F1_DUMBBELL_VECTOR}")
    print(f"Saved Brier dumbbell figure: {brier_dumbbell_path}")
    print(f"Saved Brier dumbbell vector: {DEFAULT_BRIER_DUMBBELL_VECTOR}")
    print(f"Saved summary: {DEFAULT_SUMMARY_CSV}")
    print("Scatter panel counts:")
    for panel_name, row_count in scatter_panel_counts.items():
        print(f"  - {panel_name}: {row_count}")
    print("F1 bar panel counts:")
    for panel_name, row_count in f1_bar_panel_counts.items():
        print(f"  - {panel_name}: {row_count}")
    print("Brier bar panel counts:")
    for panel_name, row_count in brier_bar_panel_counts.items():
        print(f"  - {panel_name}: {row_count}")
    print("F1 dumbbell panel counts:")
    for panel_name, row_count in f1_dumbbell_panel_counts.items():
        print(f"  - {panel_name}: {row_count}")
    print("Brier dumbbell panel counts:")
    for panel_name, row_count in brier_dumbbell_panel_counts.items():
        print(f"  - {panel_name}: {row_count}")
    print("Best models table:")
    print(summary_dataframe.to_string(index=False))


if __name__ == "__main__":
    main()
