# Generate three alternative renderings of the scale-response panel for the
# ICSC poster's (c) subfigure. Each option emphasizes the core message
# "human curves flatten/decline, LLM curves rise" in a different visual idiom.
# All three read from the per-rater d_z values across the three response
# formats (3-point, 7-point, slider).
#
# Outputs into artifacts/paper_revision/:
#   fig_c_option_A_group_band.{pdf,png}
#   fig_c_option_C_recolored_traces.{pdf,png}
#   fig_c_option_D_slopeplot.{pdf,png}
#   fig_c_option_E_mean_with_traces.{pdf,png}

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "paper_revision"


# Per-rater Cohen's d_z values across the three rating formats. Numbers come
# directly from Table 1 / Section 3.5 of slides/paper.tex. We hand-encode them
# here rather than re-reading the upstream csv so this script is small and
# self-contained for the poster panel-design step.
SCALE_LABELS: list[str] = ["3-point", "7-point", "Slider"]

HUMAN_DZ: dict[str, list[float]] = {
    "Overall humans":      [0.621, 0.588, 0.512],
    "Female participants": [0.655, 0.704, 0.590],
    "Male participants":   [0.219, 0.198, 0.167],
}

LLM_DZ: dict[str, list[float]] = {
    "Claude-Opus-4.5":  [0.498, 0.693, 0.748],
    "DeepSeek-R1":      [0.289, 0.422, 0.510],
    "DeepSeek-V3.2":    [0.289, 0.371, 0.460],
    "GLM-4.6":          [0.297, 0.353, 0.423],
    "Gemma-4-31B":      [0.408, 0.773, 1.016],
    "GPT-5.1":          [0.653, 0.840, 0.714],
    "Kimi-K2-Thinking": [0.572, 0.505, 0.593],
    "Llama-4-Maverick": [0.379, 0.477, 0.409],
    "Qwen-2.5-72B":     [0.553, 0.578, 0.686],
}


HUMAN_COLOR = "#2b6cb0"
LLM_COLOR = "#c23a3a"


def common_axis_setup(axis: plt.Axes) -> None:
    """Shared formatting for all three options."""
    axis.set_xticks(range(len(SCALE_LABELS)))
    axis.set_xticklabels(SCALE_LABELS, fontsize=18)
    axis.set_ylabel(r"Cohen's $d_z$ of $\Delta_{F-M}$", fontsize=19)
    axis.set_xlabel("Rating format", fontsize=19)
    axis.tick_params(axis="y", labelsize=16)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#e5e7eb", linewidth=0.6)


# ---------------------------------------------------------------------------
# Option A: two group-mean curves with shaded +/- SD bands.
# ---------------------------------------------------------------------------


def render_option_A_group_band() -> Path:
    """Render Option A: one solid line per group (humans, LLMs), shaded band
    showing one standard deviation across the raters in that group at each
    scale. Maximally minimal -- no per-rater traces are drawn at all."""
    sns.set_theme(context="paper", style="white")
    figure, axis = plt.subplots(figsize=(4.6, 3.2))
    x_positions = np.arange(len(SCALE_LABELS))

    human_matrix = np.array([HUMAN_DZ[label] for label in HUMAN_DZ])
    llm_matrix = np.array([LLM_DZ[label] for label in LLM_DZ])

    human_mean = human_matrix.mean(axis=0)
    human_sd = human_matrix.std(axis=0, ddof=1)
    llm_mean = llm_matrix.mean(axis=0)
    llm_sd = llm_matrix.std(axis=0, ddof=1)

    axis.fill_between(
        x_positions, human_mean - human_sd, human_mean + human_sd,
        color=HUMAN_COLOR, alpha=0.18, zorder=1,
    )
    axis.fill_between(
        x_positions, llm_mean - llm_sd, llm_mean + llm_sd,
        color=LLM_COLOR, alpha=0.18, zorder=1,
    )
    axis.plot(
        x_positions, human_mean,
        color=HUMAN_COLOR, linewidth=3.0, marker="o", markersize=8,
        markerfacecolor="white", markeredgewidth=2.0, zorder=3,
        label="Humans",
    )
    axis.plot(
        x_positions, llm_mean,
        color=LLM_COLOR, linewidth=3.0, marker="o", markersize=8,
        markerfacecolor="white", markeredgewidth=2.0, zorder=3,
        label="LLMs",
    )

    # Annotate slope direction at the right end of each line.
    axis.annotate(
        f"  Humans: {human_mean[0]:.2f} → {human_mean[-1]:.2f}",
        xy=(x_positions[-1], human_mean[-1]), xytext=(8, 0),
        textcoords="offset points", va="center", fontsize=9, color=HUMAN_COLOR,
        fontweight="bold",
    )
    axis.annotate(
        f"  LLMs: {llm_mean[0]:.2f} → {llm_mean[-1]:.2f}",
        xy=(x_positions[-1], llm_mean[-1]), xytext=(8, 0),
        textcoords="offset points", va="center", fontsize=9, color=LLM_COLOR,
        fontweight="bold",
    )

    common_axis_setup(axis)
    axis.set_xlim(-0.2, len(SCALE_LABELS) - 0.1 + 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_title(
        "Humans flatten, LLMs rise as the scale becomes finer",
        fontsize=9.5, pad=4,
    )

    output_pdf = ARTIFACTS_DIR / "fig_c_option_A_group_band.pdf"
    output_png = ARTIFACTS_DIR / "fig_c_option_A_group_band.png"
    figure.tight_layout()
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_png


# ---------------------------------------------------------------------------
# Option C: keep per-rater traces, but recolor by group.
# ---------------------------------------------------------------------------


def render_option_C_recolored_traces() -> Path:
    """Render Option C: each rater is drawn, with humans in blue (bold) and
    every LLM as a semi-transparent red line. Two big text labels on the
    right anchor the human/LLM grouping. No model is highlighted by name."""
    sns.set_theme(context="paper", style="white")
    figure, axis = plt.subplots(figsize=(4.6, 3.2))
    x_positions = np.arange(len(SCALE_LABELS))

    for label, values in LLM_DZ.items():
        axis.plot(
            x_positions, values,
            color=LLM_COLOR, linewidth=1.2, alpha=0.45, zorder=2,
            marker="o", markersize=3.5,
        )
    for label, values in HUMAN_DZ.items():
        axis.plot(
            x_positions, values,
            color=HUMAN_COLOR, linewidth=2.4, zorder=4,
            marker="o", markersize=6,
            markerfacecolor="white", markeredgewidth=1.5,
        )

    # Group labels on the right side instead of a legend box.
    human_right_y = float(np.mean([values[-1] for values in HUMAN_DZ.values()]))
    llm_right_y = float(np.mean([values[-1] for values in LLM_DZ.values()]))
    axis.annotate(
        "Humans", xy=(x_positions[-1], human_right_y), xytext=(10, 0),
        textcoords="offset points", va="center", fontsize=10,
        color=HUMAN_COLOR, fontweight="bold",
    )
    axis.annotate(
        "LLMs", xy=(x_positions[-1], llm_right_y), xytext=(10, 0),
        textcoords="offset points", va="center", fontsize=10,
        color=LLM_COLOR, fontweight="bold",
    )

    common_axis_setup(axis)
    axis.set_xlim(-0.2, len(SCALE_LABELS) - 0.1 + 0.7)
    axis.set_ylim(0.0, 1.10)
    axis.set_title(
        "Humans flatten, LLMs rise as the scale becomes finer",
        fontsize=9.5, pad=4,
    )

    output_pdf = ARTIFACTS_DIR / "fig_c_option_C_recolored_traces.pdf"
    output_png = ARTIFACTS_DIR / "fig_c_option_C_recolored_traces.png"
    figure.tight_layout()
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_png


# ---------------------------------------------------------------------------
# Option D: minimalist slope plot of the two group means only.
# ---------------------------------------------------------------------------


def render_option_D_slopeplot() -> Path:
    """Render Option D: only the two group-mean lines. Each endpoint carries
    its d_z value, and the connecting slope is drawn thick so the visual
    rhetoric reduces to 'one slope goes down, the other goes up.' No
    per-rater detail. Aspect ratio is roughly 1 : 1.3 (wide:tall) so this
    panel visually aligns with the (a) magnitude figure (862 : 1120) and the
    (b) domain forest (311 : 410) when the three subfigures are placed
    side-by-side in icsc_poster.tex."""
    sns.set_theme(context="paper", style="white")
    figure, axis = plt.subplots(figsize=(5.0, 5.8))
    x_positions = np.arange(len(SCALE_LABELS))

    human_matrix = np.array([HUMAN_DZ[label] for label in HUMAN_DZ])
    llm_matrix = np.array([LLM_DZ[label] for label in LLM_DZ])
    human_mean = human_matrix.mean(axis=0)
    llm_mean = llm_matrix.mean(axis=0)

    axis.plot(
        x_positions, human_mean,
        color=HUMAN_COLOR, linewidth=3.5, zorder=4,
        marker="o", markersize=10, markerfacecolor="white",
        markeredgewidth=2.5,
    )
    axis.plot(
        x_positions, llm_mean,
        color=LLM_COLOR, linewidth=3.5, zorder=4,
        marker="o", markersize=10, markerfacecolor="white",
        markeredgewidth=2.5,
    )

    # Symmetric handling: humans default below the marker; flip above when
    # the LLM mean is below (so the LLM line owns the area below and humans
    # stay clear).
    for index, (x_value, y_value) in enumerate(zip(x_positions, human_mean)):
        llm_is_below = llm_mean[index] < human_mean[index]
        y_offset = 18 if llm_is_below else -22
        axis.annotate(
            f"{y_value:.2f}", xy=(x_value, y_value), xytext=(0, y_offset),
            textcoords="offset points", ha="center", fontsize=15,
            color=HUMAN_COLOR, fontweight="bold",
        )
    # By default each LLM number is placed above its marker. For the 3-point
    # column the LLM mean (0.44) is below the human mean (0.50), so placing
    # 0.44 above the marker collides with the blue 0.50. We special-case the
    # first column (3-point) to drop the LLM number below its marker.
    for index, (x_value, y_value) in enumerate(zip(x_positions, llm_mean)):
        is_below_human = llm_mean[index] < human_mean[index]
        y_offset = -22 if is_below_human else 18
        axis.annotate(
            f"{y_value:.2f}", xy=(x_value, y_value), xytext=(0, y_offset),
            textcoords="offset points", ha="center", fontsize=15,
            color=LLM_COLOR, fontweight="bold",
        )

    axis.annotate(
        "Humans ↓", xy=(x_positions[-1], human_mean[-1]),
        xytext=(10, 0), textcoords="offset points", va="center", fontsize=18,
        color=HUMAN_COLOR, fontweight="bold",
    )
    axis.annotate(
        "LLMs ↑", xy=(x_positions[-1], llm_mean[-1]),
        xytext=(10, 0), textcoords="offset points", va="center", fontsize=18,
        color=LLM_COLOR, fontweight="bold",
    )

    common_axis_setup(axis)
    axis.set_xlim(-0.2, len(SCALE_LABELS) - 0.1 + 0.8)
    axis.set_ylim(0.30, 0.75)
    # No in-figure title -- the title text now lives in the LaTeX caption of
    # the (c) subfigure in icsc_poster.tex.

    output_pdf = ARTIFACTS_DIR / "fig_c_option_D_slopeplot.pdf"
    output_png = ARTIFACTS_DIR / "fig_c_option_D_slopeplot.png"
    # right=0.85 lets "Humans ↓" / "LLMs ↑" labels (anchored 10pt past
    # the rightmost slope endpoint) reach close to the figure's right
    # edge without trailing whitespace.
    figure.subplots_adjust(top=0.96, bottom=0.13, left=0.20, right=0.85)
    figure.savefig(output_pdf)
    figure.savefig(output_png, dpi=200)
    plt.close(figure)
    return output_png


def render_option_E_mean_with_traces() -> Path:
    """Render Option E: a paper-oriented compromise between A and C.

    Thin transparent lines show the individual human references and LLMs;
    thick lines show the group means. This keeps the main scale-response
    contrast legible while preserving the within-group heterogeneity that a
    two-line slope plot hides.
    """
    sns.set_theme(context="paper", style="white")
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    x_positions = np.arange(len(SCALE_LABELS))

    human_matrix = np.array([HUMAN_DZ[label] for label in HUMAN_DZ])
    llm_matrix = np.array([LLM_DZ[label] for label in LLM_DZ])
    human_mean = human_matrix.mean(axis=0)
    llm_mean = llm_matrix.mean(axis=0)

    for label, values in LLM_DZ.items():
        axis.plot(
            x_positions,
            values,
            color=LLM_COLOR,
            linewidth=1.0,
            alpha=0.22,
            marker="o",
            markersize=3.0,
            zorder=1,
        )
    for label, values in HUMAN_DZ.items():
        axis.plot(
            x_positions,
            values,
            color=HUMAN_COLOR,
            linewidth=1.4,
            alpha=0.35,
            marker="o",
            markersize=4.0,
            zorder=2,
        )

    axis.plot(
        x_positions,
        human_mean,
        color=HUMAN_COLOR,
        linewidth=3.2,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2.0,
        zorder=5,
        label="Human mean",
    )
    axis.plot(
        x_positions,
        llm_mean,
        color=LLM_COLOR,
        linewidth=3.2,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2.0,
        zorder=5,
        label="LLM mean",
    )

    axis.annotate(
        f"Human mean: {human_mean[0]:.2f} -> {human_mean[-1]:.2f}",
        xy=(x_positions[-1], human_mean[-1]),
        xytext=(10, -4),
        textcoords="offset points",
        va="center",
        fontsize=10.5,
        color=HUMAN_COLOR,
        fontweight="bold",
    )
    axis.annotate(
        f"LLM mean: {llm_mean[0]:.2f} -> {llm_mean[-1]:.2f}",
        xy=(x_positions[-1], llm_mean[-1]),
        xytext=(10, 4),
        textcoords="offset points",
        va="center",
        fontsize=10.5,
        color=LLM_COLOR,
        fontweight="bold",
    )

    axis.set_xticks(range(len(SCALE_LABELS)))
    axis.set_xticklabels(SCALE_LABELS, fontsize=12)
    axis.set_ylabel(r"Cohen's $d_z$ of $\Delta_{F-M}$", fontsize=12)
    axis.set_xlabel("Rating format", fontsize=12)
    axis.tick_params(axis="y", labelsize=11)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#e5e7eb", linewidth=0.6)
    axis.set_xlim(-0.15, len(SCALE_LABELS) - 0.1 + 0.9)
    axis.set_ylim(0.0, 1.08)

    axis.text(
        0.02,
        0.97,
        "Thin lines: individual human references and LLMs",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#4b5563",
    )

    output_pdf = ARTIFACTS_DIR / "fig_c_option_E_mean_with_traces.pdf"
    output_png = ARTIFACTS_DIR / "fig_c_option_E_mean_with_traces.png"
    figure.tight_layout()
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_png


def main() -> None:
    # Options A and C were exploratory and have been retired; the poster
    # uses Option D (slopeplot) as the (c) panel. The two render_option_A_
    # and render_option_C_ functions are kept for reference but not called,
    # so re-running this script does not regenerate the retired PNG/PDF
    # intermediates after they have been cleaned up.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    d_path = render_option_D_slopeplot()
    e_path = render_option_E_mean_with_traces()
    print("Option D:", d_path.relative_to(PROJECT_ROOT))
    print("Option E:", e_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
