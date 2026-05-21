# Wide horizontal version of the per-domain errorbar forest, restricted to
# the 3-point response format, for the (b) panel of the ICSC poster's
# combined figure. Aspect ratio is roughly 2:1 (wide:tall) so the panel fits
# neatly as the centerpiece between the user-supplied magnitude figure on
# the left and the existing scale-trajectory figure on the right.
#
# Output:
#   artifacts/paper_revision/fig_r4_poster_3pt_errorbar_forest.{pdf,png}
#   icsc/figures/fig_r4_poster_3pt_errorbar_forest.pdf

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "paper_revision"
POSTER_FIG_DIR = PROJECT_ROOT / "icsc" / "figures"

SUMMARY_CSV = ARTIFACTS_DIR / "domain_alignment_errorbar_summary.csv"
DOM_OUT_PDF = ARTIFACTS_DIR / "fig_r4_poster_3pt_errorbar_forest.pdf"
DOM_OUT_PNG = ARTIFACTS_DIR / "fig_r4_poster_3pt_errorbar_forest.png"
DOM_OUT_PDF_COPY = POSTER_FIG_DIR / "fig_r4_poster_3pt_errorbar_forest.pdf"

DOM_RAW_OUT_PDF = ARTIFACTS_DIR / "fig_r4_poster_3pt_errorbar_forest_raw.pdf"
DOM_RAW_OUT_PNG = ARTIFACTS_DIR / "fig_r4_poster_3pt_errorbar_forest_raw.png"
DOM_RAW_OUT_PDF_COPY = POSTER_FIG_DIR / "fig_r4_poster_3pt_errorbar_forest_raw.pdf"


# Domain order in display order (top-to-bottom). Sexualization and appearance
# first to highlight the documented hot zone.
DOMAIN_ORDER_ZH: list[str] = [
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

DOMAIN_SHORT_LABELS_EN: dict[str, str] = {
    "性化攻击（性羞辱）":     "Sexualization",
    "外貌形象攻击":           "Appearance",
    "性别角色/性别表达攻击":  "Gender role",
    "人际关系攻击":           "Interpersonal",
    "道德品行攻击":           "Moral character",
    "经济资源攻击":           "Economic",
    "社会地位攻击":           "Social status",
    "情绪稳定攻击":           "Emotional",
    "能力才干攻击":           "Capability",
    "智力理性攻击":           "Intellectual",
}

RATER_GROUP_ORDER: list[str] = [
    "Overall humans",
    "Female participants",
    "Male participants",
    "LLM median",
]
RATER_PALETTE: dict[str, str] = {
    "Overall humans":      "#1f77b4",
    "Female participants": "#d62728",
    "Male participants":   "#2ca02c",
    "LLM median":          "#4c4c4c",
}
RATER_MARKER: dict[str, str] = {
    "Overall humans":      "o",
    "Female participants": "D",
    "Male participants":   "s",
    "LLM median":          "^",
}
# Compact vertical offsets for the vertical (tall) layout: rows are spread
# further apart by the tall aspect ratio, so each rater dot only needs a
# small offset from the row center to stay separable.
RATER_VERTICAL_OFFSET: dict[str, float] = {
    "Overall humans":      0.22,
    "Female participants": 0.07,
    "Male participants":  -0.07,
    "LLM median":         -0.22,
}


def load_domain_3pt_summary() -> pl.DataFrame:
    """Per-domain bootstrap summary for the 3-point scale."""
    full = pl.read_csv(SUMMARY_CSV)
    return full.filter(pl.col("condition") == "attack_3pt")


def render_domain_errorbar_forest() -> None:
    """Render the wide (2:1) horizontal version of the per-domain forest.

    Visual grammar matches the original three-panel errorbar forest -- one
    row per attack domain, four markers per row, 95% bootstrap CIs -- so the
    poster reader who has seen the main paper figure recognizes this
    immediately. Only difference: 3-point only, in a 2:1 (wide:tall) frame."""
    sns.set_theme(context="paper", style="white")

    summary = load_domain_3pt_summary()
    y_positions = {
        domain: len(DOMAIN_ORDER_ZH) - 1 - index
        for index, domain in enumerate(DOMAIN_ORDER_ZH)
    }

    # (b) is rendered on a WIDER canvas than (a)/(c) so the in-axes legend
    # ("Overall humans" at fontsize=16) can fit inside the empty x>0.13
    # band without clipping. The LaTeX subfigure widths are set so that
    # the rendered PDF heights stay visually equal across the three panels:
    # 0.42 * 5.8/6.7 ~= 0.295 * 5.8/5.0 ~= 0.34*linewidth.
    figure, axis = plt.subplots(figsize=(6.7, 5.8))

    for domain in DOMAIN_ORDER_ZH:
        axis.axhline(y_positions[domain], color="#e5e7eb", linewidth=0.6, zorder=0)
    axis.axvline(0, color="#6b7280", linestyle="--", linewidth=0.7, zorder=1)

    for rater_group in RATER_GROUP_ORDER:
        rater_frame = summary.filter(pl.col("rater_group") == rater_group)
        domains = rater_frame["dimension_1"].to_list()
        x_values = rater_frame["mean_delta_norm"].to_numpy()
        ci_low_values = rater_frame["ci_low"].to_numpy()
        ci_high_values = rater_frame["ci_high"].to_numpy()
        y_values = np.array(
            [y_positions[domain] + RATER_VERTICAL_OFFSET[rater_group] for domain in domains]
        )
        lower_error = x_values - ci_low_values
        upper_error = ci_high_values - x_values

        axis.errorbar(
            x_values,
            y_values,
            xerr=np.vstack([lower_error, upper_error]),
            fmt=RATER_MARKER[rater_group],
            markersize=5.0,
            color=RATER_PALETTE[rater_group],
            ecolor=RATER_PALETTE[rater_group],
            elinewidth=1.0,
            capsize=2.2,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=rater_group,
            zorder=3,
        )

    axis.set_yticks([y_positions[domain] for domain in DOMAIN_ORDER_ZH])
    axis.set_yticklabels(
        [DOMAIN_SHORT_LABELS_EN[domain] for domain in DOMAIN_ORDER_ZH], fontsize=18,
    )
    axis.set_ylim(-0.6, len(DOMAIN_ORDER_ZH) - 0.2)

    x_upper = max(0.45, float(summary["ci_high"].max()) * 1.08)
    axis.set_xlim(-0.06, x_upper)
    axis.set_xlabel(r"Mean $\Delta_{F-M}$ / scale max", fontsize=19)
    axis.tick_params(axis="x", labelsize=16)

    # Hot-zone domains bolded.
    for tick_label in axis.get_yticklabels()[:2]:
        tick_label.set_fontweight("bold")

    axis.grid(axis="x", color="#e5e7eb", linewidth=0.6)
    axis.grid(axis="y", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    # Single-column legend stacked vertically and pinned to the lower-right
    # corner of the axes (inside the plot box, in the unused 0.25-0.45 x
    # region near the Intellectual row). Anchoring inside the axes leaves
    # the xlim untouched and keeps the figure crop tight; ncol=1 makes the
    # four labels stack vertically rather than overlap the Intellectual
    # marker row at the bottom.
    # Legend lives INSIDE the axes, in the lower-right region of the data
    # area where no domain has data (all forest CIs stop before x = 0.22,
    # so the band x in [0.22, 0.45] across all rows is empty). Short
    # display labels keep the legend column narrow enough to fit in that
    # band, regardless of figure cropping.
    legend_display_labels = {
        "Overall humans":      "Overall humans",
        "Female participants": "Female",
        "Male participants":   "Male",
        "LLM median":          "LLM",
    }
    handles, labels = axis.get_legend_handles_labels()
    short_labels = [legend_display_labels.get(lab, lab) for lab in labels]
    # Place the legend inside the axes box in the empty x>0.17 band of the
    # lower forest rows (Emotional / Capability / Intellectual; all ci_high
    # values in that band are <= 0.17). The 16pt label column starting at
    # data x = 0.17 stays well left of the right axis edge (xlim = 0.45),
    # so "Overall humans" fits without clipping.
    axis.legend(
        handles, short_labels,
        loc="upper left",
        bbox_to_anchor=(0.13, 3.5),
        bbox_transform=axis.transData,
        fontsize=16,
        frameon=False,
        handletextpad=0.4,
        labelspacing=0.50,
        borderpad=0.2,
        ncol=1,
    )

    # (b): figure is 6.7 inch wide, so left=0.27 still gives 0.27*6.7 =
    # 1.81 inch of room for the longest y-tick label ("Moral character" at
    # fontsize=18 ~ 1.7 inch). Right=0.99 stretches the axes to the very
    # right edge of the figure (the legend is anchored at data x=0.13
    # inside the axes, so right-margin space is not needed for it).
    figure.subplots_adjust(top=0.96, bottom=0.13, left=0.27, right=0.99)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    POSTER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Do NOT use bbox_inches='tight' -- it re-crops every figure differently
    # depending on label widths, which undoes the alignment above.
    figure.savefig(DOM_OUT_PDF)
    figure.savefig(DOM_OUT_PNG, dpi=200)
    figure.savefig(DOM_OUT_PDF_COPY)
    plt.close(figure)

    print(f"Wrote {DOM_OUT_PDF.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {DOM_OUT_PNG.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {DOM_OUT_PDF_COPY.relative_to(PROJECT_ROOT)}  (for icsc_poster.tex)")


# ---------------------------------------------------------------------------
# (a) Magnitude per rater on the 3-point scale.
# ---------------------------------------------------------------------------

DIRECTION_SUMMARY_CSV = ARTIFACTS_DIR / "direction_agreement_summary.csv"
MAG_OUT_PDF = ARTIFACTS_DIR / "fig_a_magnitude_3pt_forest.pdf"
MAG_OUT_PNG = ARTIFACTS_DIR / "fig_a_magnitude_3pt_forest.png"
MAG_OUT_PDF_COPY = POSTER_FIG_DIR / "fig_a_magnitude_3pt_forest.pdf"

MAG_RAW_OUT_PDF = ARTIFACTS_DIR / "fig_a_magnitude_3pt_forest_raw.pdf"
MAG_RAW_OUT_PNG = ARTIFACTS_DIR / "fig_a_magnitude_3pt_forest_raw.png"
MAG_RAW_OUT_PDF_COPY = POSTER_FIG_DIR / "fig_a_magnitude_3pt_forest_raw.pdf"


HUMAN_DISPLAY_ORDER: list[tuple[str, str]] = [
    ("human_all",    "Overall humans"),
    ("human_female", "Female"),
    ("human_male",   "Male"),
]

LLM_DISPLAY_LABELS: dict[str, str] = {
    "deepseek_deepseek-r1-0528":     "DeepSeek-R1",
    "deepseek_deepseek-v3.2-exp":    "DeepSeek-V3.2",
    "z-ai_glm-4.6":                  "GLM-4.6",
    "meta-llama_llama-4-maverick":   "Llama-4-Maverick",
    "google_gemma-4-31b-it":         "Gemma-4-31B",
    "anthropic_claude-opus-4.5":     "Claude-Opus-4.5",
    "qwen_qwen-2.5-72b-instruct":    "Qwen-2.5-72B",
    "moonshotai_kimi-k2-thinking":   "Kimi-K2-Thinking",
    "openai_gpt-5.1":                "GPT-5.1",
}


def load_magnitude_3pt() -> pl.DataFrame:
    """Per-rater bootstrap summary on the 3-point scale (from direction
    agreement bootstrap, which contains the mean Delta/scale-max + CI we
    plot in the (a) panel)."""
    full = pl.read_csv(DIRECTION_SUMMARY_CSV)
    return full.filter(pl.col("condition") == "attack_3pt")


def render_magnitude_3pt_forest() -> None:
    """Render the 12-rater magnitude forest used as the (a) panel.

    Rows: 3 human strata on top sorted by ascending mean, 9 LLMs below sorted
    by ascending mean. Colors: humans in blue, LLMs in red. Layout matches
    the original three-panel fig_r2_direction_agreement_forest but isolates
    the 3-point sub-panel for the poster."""
    sns.set_theme(context="paper", style="white")

    summary = load_magnitude_3pt()
    by_rater = {row["rater_id"]: row for row in summary.iter_rows(named=True)}

    human_rows = []
    for rater_id, display_label in HUMAN_DISPLAY_ORDER:
        if rater_id not in by_rater:
            continue
        row = by_rater[rater_id]
        human_rows.append((display_label, row["mean_delta_norm"], row["ci_low"], row["ci_high"]))

    llm_rows_raw = []
    for rater_id, display_label in LLM_DISPLAY_LABELS.items():
        if rater_id not in by_rater:
            continue
        row = by_rater[rater_id]
        llm_rows_raw.append((display_label, row["mean_delta_norm"], row["ci_low"], row["ci_high"]))
    # Sort LLMs by mean ascending so the panel walks low -> high LLM effect.
    llm_rows_raw.sort(key=lambda triple: triple[1])

    # Vertical layout: humans on top, LLMs below; top-to-bottom = order in
    # the lists above.
    all_rows = human_rows + llm_rows_raw
    n_rows = len(all_rows)
    y_positions = np.arange(n_rows)[::-1]

    figure, axis = plt.subplots(figsize=(5.0, 5.8))
    HUMAN_COLOR = "#2b6cb0"
    LLM_COLOR = "#c23a3a"

    # The left axis spine itself anchors at x=0 (xlim below), so we do not
    # add an extra axvline at 0 -- that would render as a second vertical
    # line right next to the spine.

    for index, (label, mean_value, ci_low, ci_high) in enumerate(all_rows):
        y = y_positions[index]
        is_human = index < len(human_rows)
        color = HUMAN_COLOR if is_human else LLM_COLOR
        axis.errorbar(
            mean_value, y,
            xerr=[[mean_value - ci_low], [ci_high - mean_value]],
            fmt="o", markersize=8.5,
            color=color, ecolor=color,
            elinewidth=1.5, capsize=3.6,
            zorder=3,
        )

    axis.set_yticks(y_positions)
    axis.set_yticklabels([label for label, _, _, _ in all_rows], fontsize=18)
    axis.set_ylim(-0.6, n_rows - 0.4)

    axis.set_xlabel(r"Mean $\Delta_{F-M}$ / scale max", fontsize=19)
    axis.tick_params(axis="x", labelsize=16)
    axis.set_xlim(0.0, 0.17)
    axis.set_xticks([0.05, 0.10, 0.15])
    # No in-axes title; the subfigure caption in icsc_poster.tex names this
    # panel. Removing the title also matches (b) and (c), so the three axes
    # boxes share the same vertical extent when placed side-by-side.

    axis.grid(axis="x", color="#E6EAF0", linewidth=0.6)
    axis.grid(axis="y", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    # (a): rater labels ("Llama-4-Maverick", "Kimi-K2-Thinking", "Claude-
    # Opus-4.5") at fontsize=18 need a wider left margin than (b)/(c).
    # right=0.99 stretches the axes to the very right edge of the (5.0,
    # 5.8) figure so no whitespace remains past the rightmost x-tick.
    figure.subplots_adjust(top=0.96, bottom=0.13, left=0.46, right=0.99)
    figure.savefig(MAG_OUT_PDF)
    figure.savefig(MAG_OUT_PNG, dpi=200)
    figure.savefig(MAG_OUT_PDF_COPY)
    plt.close(figure)
    print(f"Wrote {MAG_OUT_PDF.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {MAG_OUT_PNG.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {MAG_OUT_PDF_COPY.relative_to(PROJECT_ROOT)}  (for icsc_poster.tex)")


# ---------------------------------------------------------------------------
# Raw-scale versions: same plots but with x-axis in original 3-point rating
# units (mean Delta in [0,2]) rather than the normalized [0,1] fraction.
# 3-point scale max = 2 (labels 0=not, 1=mildly, 2=strongly attacking), so
# every value is multiplied by 2. These _raw outputs are written alongside
# the normalized versions so both can be referenced.
# ---------------------------------------------------------------------------


SCALE_MAX_3PT = 2.0


def render_domain_errorbar_forest_raw() -> None:
    """Same as render_domain_errorbar_forest but in original 3-point units.
    Figure is rendered on a 5.0 x 5.8 inch canvas matching (a) and (c)
    exactly, so all three panels can use the same LaTeX subfigure width
    (0.295\linewidth) and render at the same visible height."""
    sns.set_theme(context="paper", style="white")

    summary = load_domain_3pt_summary()
    y_positions = {
        domain: len(DOMAIN_ORDER_ZH) - 1 - index
        for index, domain in enumerate(DOMAIN_ORDER_ZH)
    }

    figure, axis = plt.subplots(figsize=(5.0, 5.8))

    for domain in DOMAIN_ORDER_ZH:
        axis.axhline(y_positions[domain], color="#e5e7eb", linewidth=0.6, zorder=0)
    axis.axvline(0, color="#6b7280", linestyle="--", linewidth=0.7, zorder=1)

    for rater_group in RATER_GROUP_ORDER:
        rater_frame = summary.filter(pl.col("rater_group") == rater_group)
        domains = rater_frame["dimension_1"].to_list()
        x_values = rater_frame["mean_delta_norm"].to_numpy() * SCALE_MAX_3PT
        ci_low_values = rater_frame["ci_low"].to_numpy() * SCALE_MAX_3PT
        ci_high_values = rater_frame["ci_high"].to_numpy() * SCALE_MAX_3PT
        y_values = np.array(
            [y_positions[domain] + RATER_VERTICAL_OFFSET[rater_group] for domain in domains]
        )
        lower_error = x_values - ci_low_values
        upper_error = ci_high_values - x_values

        axis.errorbar(
            x_values,
            y_values,
            xerr=np.vstack([lower_error, upper_error]),
            fmt=RATER_MARKER[rater_group],
            markersize=5.0,
            color=RATER_PALETTE[rater_group],
            ecolor=RATER_PALETTE[rater_group],
            elinewidth=1.0,
            capsize=2.2,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=rater_group,
            zorder=3,
        )

    axis.set_yticks([y_positions[domain] for domain in DOMAIN_ORDER_ZH])
    axis.set_yticklabels(
        [DOMAIN_SHORT_LABELS_EN[domain] for domain in DOMAIN_ORDER_ZH], fontsize=18,
    )
    axis.set_ylim(-0.6, len(DOMAIN_ORDER_ZH) - 0.2)

    x_upper = max(0.45, float(summary["ci_high"].max()) * 1.08) * SCALE_MAX_3PT
    axis.set_xlim(-0.06 * SCALE_MAX_3PT, x_upper)
    axis.set_xlabel(r"$\Delta_{F-M}$ rating (3-point)", fontsize=18)
    axis.tick_params(axis="x", labelsize=16)

    for tick_label in axis.get_yticklabels()[:2]:
        tick_label.set_fontweight("bold")

    axis.grid(axis="x", color="#e5e7eb", linewidth=0.6)
    axis.grid(axis="y", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    legend_display_labels = {
        "Overall humans":      "Overall humans",
        "Female participants": "Female",
        "Male participants":   "Male",
        "LLM median":          "LLM",
    }
    handles, labels = axis.get_legend_handles_labels()
    short_labels = [legend_display_labels.get(lab, lab) for lab in labels]
    # The max CI in the data is ~0.42 in raw 3-point units, so the band
    # x in [0.50, 0.90] is a 100% empty strip across all forest rows.
    # Anchor the legend INSIDE the axes at data x = 0.50, which keeps it
    # within the x-axis range (per requirement 4) and clear of every dot.
    axis.legend(
        handles, short_labels,
        loc="upper left",
        bbox_to_anchor=(0.50, 9.0),
        bbox_transform=axis.transData,
        fontsize=14,
        frameon=False,
        handletextpad=0.3,
        labelspacing=0.45,
        borderpad=0.15,
        ncol=1,
    )

    # (b) raw: 5.0 inch wide figure. left=0.38 reserves 1.9 inch for the
    # longest y-tick label ("Moral character" at fontsize 18); right=0.99
    # stretches the axes to the figure's right edge -- the legend lives
    # inside the axes box in the empty x>0.45 band, so no extra figure
    # margin is needed.
    figure.subplots_adjust(top=0.96, bottom=0.13, left=0.38, right=0.99)
    figure.savefig(DOM_RAW_OUT_PDF)
    figure.savefig(DOM_RAW_OUT_PNG, dpi=200)
    figure.savefig(DOM_RAW_OUT_PDF_COPY)
    plt.close(figure)
    print(f"Wrote {DOM_RAW_OUT_PDF.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {DOM_RAW_OUT_PDF_COPY.relative_to(PROJECT_ROOT)}  (for icsc_poster.tex)")


def render_magnitude_3pt_forest_raw() -> None:
    """Same as render_magnitude_3pt_forest but in original 3-point units."""
    sns.set_theme(context="paper", style="white")

    summary = load_magnitude_3pt()
    by_rater = {row["rater_id"]: row for row in summary.iter_rows(named=True)}

    human_rows = []
    for rater_id, display_label in HUMAN_DISPLAY_ORDER:
        if rater_id not in by_rater:
            continue
        row = by_rater[rater_id]
        human_rows.append((display_label, row["mean_delta_norm"], row["ci_low"], row["ci_high"]))

    llm_rows_raw = []
    for rater_id, display_label in LLM_DISPLAY_LABELS.items():
        if rater_id not in by_rater:
            continue
        row = by_rater[rater_id]
        llm_rows_raw.append((display_label, row["mean_delta_norm"], row["ci_low"], row["ci_high"]))
    llm_rows_raw.sort(key=lambda triple: triple[1])

    all_rows = human_rows + llm_rows_raw
    n_rows = len(all_rows)
    y_positions = np.arange(n_rows)[::-1]

    figure, axis = plt.subplots(figsize=(5.0, 5.8))
    HUMAN_COLOR = "#2b6cb0"
    LLM_COLOR = "#c23a3a"

    for index, (label, mean_value, ci_low, ci_high) in enumerate(all_rows):
        y = y_positions[index]
        is_human = index < len(human_rows)
        color = HUMAN_COLOR if is_human else LLM_COLOR
        mean_value = mean_value * SCALE_MAX_3PT
        ci_low = ci_low * SCALE_MAX_3PT
        ci_high = ci_high * SCALE_MAX_3PT
        axis.errorbar(
            mean_value, y,
            xerr=[[mean_value - ci_low], [ci_high - mean_value]],
            fmt="o", markersize=8.5,
            color=color, ecolor=color,
            elinewidth=1.5, capsize=3.6,
            zorder=3,
        )

    axis.set_yticks(y_positions)
    axis.set_yticklabels([label for label, _, _, _ in all_rows], fontsize=18)
    axis.set_ylim(-0.6, n_rows - 0.4)

    axis.set_xlabel(r"$\Delta_{F-M}$ rating (3-point)", fontsize=18)
    axis.tick_params(axis="x", labelsize=16)
    # xlim upper = 0.38 leaves room for the GPT-5.1 ci_high tick (~0.348)
    # so the right-side errorbar cap is not clipped.
    axis.set_xlim(0.0, 0.38)
    axis.set_xticks([0.10, 0.20, 0.30])

    axis.grid(axis="x", color="#E6EAF0", linewidth=0.6)
    axis.grid(axis="y", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.subplots_adjust(top=0.96, bottom=0.13, left=0.46, right=0.99)
    figure.savefig(MAG_RAW_OUT_PDF)
    figure.savefig(MAG_RAW_OUT_PNG, dpi=200)
    figure.savefig(MAG_RAW_OUT_PDF_COPY)
    plt.close(figure)
    print(f"Wrote {MAG_RAW_OUT_PDF.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {MAG_RAW_OUT_PDF_COPY.relative_to(PROJECT_ROOT)}  (for icsc_poster.tex)")


if __name__ == "__main__":
    render_domain_errorbar_forest()
    render_magnitude_3pt_forest()
    render_domain_errorbar_forest_raw()
    render_magnitude_3pt_forest_raw()
