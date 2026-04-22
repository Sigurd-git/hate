"""Per-dimension effect sizes across models and scales.

Generates three figures from the three stats_level1.csv files and three
stats_overall.csv files of Study 1b:

    F5: model_by_level1_by_scale_dz_heatmap.pdf/.png
        1 x 3 heatmap (9 models x 10 level-1 attack domains x 3 scales),
        shared diverging colour scale, FDR-star overlay, hatched low-n cells.

    F6: overall_dz_cross_scale_scatter.pdf/.png
        Three pairwise scatter plots (one point per model) of the overall
        Cohen's d_z across scale pairs; per-panel Spearman rho over 9 models.

    F8: dimension_level_cross_scale_rho_forest.pdf/.png
        Per-model Spearman rho / Kendall tau on the 10-domain d_z vector
        between pairs of scales, with bootstrap 95% CIs. 10 domains only,
        so CIs are wide.

Transparency CSVs:
    model_by_level1_by_scale.csv
    dimension_level_cross_scale_similarity.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from plot_f1_brier import (
    SPINE_COLOR,
    SUBTLE_TEXT_COLOR,
    TEXT_COLOR,
    build_color_mapping,
    configure_plot_theme,
)


PROJECT_ROOT = Path(__file__).resolve().parent
STATS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "group_swap_1b"
    / "1b_groupswap_demensionsentence"
    / "analysis_1b"
    / "figures"
)
ARTIFACTS = PROJECT_ROOT / "artifacts"

SCALE_ORDER: list[tuple[str, str]] = [
    ("attack_3pt", "3-point"),
    ("attack_7pt_likert", "7-point Likert"),
    ("attack_slider_0_100", "0\u2013100 slider"),
]

SCALE_PAIRS: list[tuple[str, str, str]] = [
    ("attack_3pt", "attack_7pt_likert", "3pt vs 7pt"),
    ("attack_7pt_likert", "attack_slider_0_100", "7pt vs slider"),
    ("attack_3pt", "attack_slider_0_100", "3pt vs slider"),
]

# Fixed column order for the heatmap so all three panels align. The "gender
# role / expression" domain is intentionally placed after body/sex/appearance
# because its n=14 makes it a small-sample corner rather than a main column.
LEVEL1_ORDER: list[str] = [
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

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260420
LOW_N_THRESHOLD = 20  # below this, the cell gets a hatched overlay.


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_level1_long() -> pd.DataFrame:
    """Stack the three stats_level1.csv files into one long frame with scale."""
    frames: list[pd.DataFrame] = []
    for scale_key, scale_label in SCALE_ORDER:
        csv_path = STATS_DIR / scale_key / "stats_level1.csv"
        frame = pd.read_csv(csv_path)
        frame = frame.assign(scale_key=scale_key, scale_label=scale_label)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_overall_long() -> pd.DataFrame:
    """Stack the three stats_overall.csv files into one long frame."""
    frames: list[pd.DataFrame] = []
    for scale_key, scale_label in SCALE_ORDER:
        csv_path = STATS_DIR / scale_key / "stats_overall.csv"
        frame = pd.read_csv(csv_path)
        frame = frame.assign(scale_key=scale_key, scale_label=scale_label)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# F5: 9 x 10 x 3 heatmap
# ---------------------------------------------------------------------------
def _stars_for_q(q_value: float) -> str:
    """Significance star string for a BH-FDR q value."""
    if pd.isna(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def build_f5_heatmap(level1_long: pd.DataFrame) -> plt.Figure:
    """9 x 10 x 3 heatmap; shared symmetric diverging colour scale."""
    configure_plot_theme()
    # Use a sans-serif font chain that actually covers CJK glyphs; the default
    # "DejaVu Sans" does not ship Chinese characters, so we must point the
    # sans-serif family at Noto Sans CJK before DejaVu.
    # Matplotlib exposes the .ttc file under the first alias it encounters
    # (here "Noto Sans CJK JP"); SC/HK names are not recognised even though
    # the file covers all CJK scripts. JP is used purely as the font handle.
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "DejaVu Sans",
    ]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    # configure_plot_theme sets pdf.fonttype=42 (Type 42, TrueType PDF
    # embedding). Noto Sans CJK ships as a TrueType Collection (.ttc), which
    # matplotlib mislabels as CFF/OpenType when writing Type 42 — this
    # produces a PDF that embeds the correct font file but with an incorrect
    # font-type declaration, so PDF readers render CJK as tofu boxes.
    # Falling back to Type 3 (pdf.fonttype=3) sidesteps the issue by
    # converting text to path outlines.
    plt.rcParams["pdf.fonttype"] = 3

    model_order = sorted(level1_long["model_label"].astype(str).unique())

    max_abs = float(level1_long["cohens_dz"].abs().max())
    vmax = max_abs * 1.02 if max_abs > 0 else 1.0
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    # sharey=False so the leftmost panel keeps its model labels reliably; we
    # manually clear y-ticklabels on the non-leftmost panels below.
    figure, axes = plt.subplots(1, 3, figsize=(19.0, 6.4), sharey=False)
    last_image = None

    for panel_idx, (scale_key, scale_label) in enumerate(SCALE_ORDER):
        axis = axes[panel_idx]
        panel_frame = level1_long[level1_long["scale_key"] == scale_key]

        matrix = np.full((len(model_order), len(LEVEL1_ORDER)), np.nan)
        n_matrix = np.full_like(matrix, np.nan)
        q_matrix = np.full_like(matrix, np.nan)

        for _, row in panel_frame.iterrows():
            model_label = str(row["model_label"])
            domain_label = str(row["一级-攻击领域"])
            if model_label not in model_order:
                continue
            if domain_label not in LEVEL1_ORDER:
                continue
            row_idx = model_order.index(model_label)
            col_idx = LEVEL1_ORDER.index(domain_label)
            matrix[row_idx, col_idx] = float(row["cohens_dz"])
            n_matrix[row_idx, col_idx] = float(row["n"])
            q_matrix[row_idx, col_idx] = float(row["wilcoxon_p_fdr"])

        image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        last_image = image

        # Overlay d_z values and FDR stars inside each cell.
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if np.isnan(value):
                    continue
                star = _stars_for_q(q_matrix[row_idx, col_idx])
                text_value = f"{value:.2f}"
                text_label = text_value + (f"\n{star}" if star else "")
                text_color = (
                    "white" if abs(value) > vmax * 0.55 else TEXT_COLOR
                )
                axis.text(
                    col_idx,
                    row_idx,
                    text_label,
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color=text_color,
                    linespacing=0.9,
                )
                if n_matrix[row_idx, col_idx] < LOW_N_THRESHOLD:
                    axis.add_patch(
                        plt.Rectangle(
                            (col_idx - 0.5, row_idx - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            hatch="///",
                            edgecolor=TEXT_COLOR,
                            alpha=0.35,
                            linewidth=0.0,
                        )
                    )

        axis.set_xticks(np.arange(len(LEVEL1_ORDER)))
        axis.set_xticklabels(LEVEL1_ORDER, rotation=35, ha="right", fontsize=8.6)
        if panel_idx == 0:
            axis.set_yticks(np.arange(len(model_order)))
            axis.set_yticklabels(model_order, fontsize=9.2)
        else:
            axis.set_yticks(np.arange(len(model_order)))
            axis.set_yticklabels([])
        axis.set_title(scale_label, fontsize=12.5, color=TEXT_COLOR)
        axis.tick_params(axis="both", length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)

    # Shared colour bar on the right, spanning all three panels.
    cbar = figure.colorbar(
        last_image, ax=axes, fraction=0.018, pad=0.02, shrink=0.85
    )
    cbar.set_label(
        r"Cohen's $d_z$ (female $-$ male)",
        fontsize=10.5,
        color=TEXT_COLOR,
    )

    figure.suptitle(
        "Study 1b per-dimension effect sizes: 9 models \u00d7 10 level-1 "
        "attack domains \u00d7 3 rating scales",
        fontsize=13.2,
        color=TEXT_COLOR,
        y=1.00,
    )
    figure.text(
        0.5,
        -0.015,
        "Stars: BH-FDR q (* <.05, ** <.01, *** <.001). "
        f"Hatched cells: n < {LOW_N_THRESHOLD} items (statistical power "
        "limited).",
        ha="center",
        fontsize=9,
        color=SUBTLE_TEXT_COLOR,
    )
    return figure


# ---------------------------------------------------------------------------
# F6: aggregate-level cross-scale d_z scatter
# ---------------------------------------------------------------------------
def build_f6_scatter(overall_long: pd.DataFrame) -> plt.Figure:
    """Three pairwise scatter plots of overall d_z (one point per model)."""
    configure_plot_theme()
    plt.rcParams["font.family"] = [
        "Noto Sans CJK JP",
        *plt.rcParams.get("font.family", []),
    ]

    models = sorted(overall_long["model_label"].astype(str).unique())
    color_map = build_color_mapping(models)

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.9))
    for panel_idx, (scale_a, scale_b, pair_label) in enumerate(SCALE_PAIRS):
        axis = axes[panel_idx]
        sub_a = overall_long[overall_long["scale_key"] == scale_a].set_index(
            "model_label"
        )
        sub_b = overall_long[overall_long["scale_key"] == scale_b].set_index(
            "model_label"
        )
        joined = (
            sub_a[["cohens_dz"]]
            .join(sub_b[["cohens_dz"]], lsuffix="_a", rsuffix="_b")
            .dropna()
        )
        joined = joined.reindex(models).dropna()

        x_values = joined["cohens_dz_a"].to_numpy()
        y_values = joined["cohens_dz_b"].to_numpy()
        combined = np.concatenate([x_values, y_values])
        lo = float(combined.min()) * 0.95
        hi = float(combined.max()) * 1.05
        axis.plot(
            [lo, hi],
            [lo, hi],
            linestyle="--",
            color="#99A3AE",
            linewidth=0.9,
        )

        for name, x_val, y_val in zip(joined.index, x_values, y_values):
            axis.scatter(
                x_val,
                y_val,
                s=70,
                color=color_map.get(name, "#6C7A89"),
                edgecolors="#22303C",
                linewidths=0.8,
                zorder=3,
            )
            axis.annotate(
                name,
                (x_val, y_val),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=8.2,
                color=TEXT_COLOR,
            )

        rho = float(stats.spearmanr(x_values, y_values).correlation)
        axis.set_title(
            f"{pair_label}\nSpearman " + rf"$\rho_s$ = {rho:+.2f}",
            fontsize=11.5,
            color=TEXT_COLOR,
        )
        axis.set_xlabel(f"{scale_a} \u2014 $d_z$")
        axis.set_ylabel(f"{scale_b} \u2014 $d_z$")
        axis.set_xlim(lo, hi)
        axis.set_ylim(lo, hi)
        axis.tick_params(axis="both", length=0)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    figure.suptitle(
        r"Aggregate-level cross-scale alignment of 9 models' overall $d_z$",
        fontsize=13.2,
        color=TEXT_COLOR,
        y=1.00,
    )
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.93))
    return figure


# ---------------------------------------------------------------------------
# F8: dimension-level cross-scale rho forest
# ---------------------------------------------------------------------------
def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation via pandas for clean tie handling."""
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau-b via scipy; returns NaN when the pair is degenerate."""
    if len(a) < 2:
        return float("nan")
    return float(stats.kendalltau(a, b).correlation)


def _bootstrap_ci(
    metric_fn, a: np.ndarray, b: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float]:
    """Bootstrap 95% CI by index-resampling two aligned vectors in lockstep."""
    rng = np.random.default_rng(seed)
    n = len(a)
    values = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        pick = rng.integers(0, n, size=n)
        values[idx] = metric_fn(a[pick], b[pick])
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def compute_dim_level_cross_scale(
    level1_long: pd.DataFrame,
) -> pd.DataFrame:
    """Per-model 10-domain d_z vectors and their cross-scale Spearman/Kendall."""
    rows: list[dict] = []
    model_order = sorted(level1_long["model_label"].astype(str).unique())

    for model_label in model_order:
        model_rows = level1_long[level1_long["model_label"] == model_label]

        vectors: dict[str, np.ndarray] = {}
        for scale_key, _ in SCALE_ORDER:
            scale_rows = model_rows[model_rows["scale_key"] == scale_key]
            scale_rows = scale_rows.set_index(
                "一级-攻击领域"
            )
            vec = np.array(
                [
                    float(scale_rows.loc[dom, "cohens_dz"])
                    if dom in scale_rows.index
                    else np.nan
                    for dom in LEVEL1_ORDER
                ],
                dtype=float,
            )
            vectors[scale_key] = vec

        for scale_a, scale_b, pair_label in SCALE_PAIRS:
            vec_a = vectors[scale_a]
            vec_b = vectors[scale_b]
            mask = ~(np.isnan(vec_a) | np.isnan(vec_b))
            vec_a_filt = vec_a[mask]
            vec_b_filt = vec_b[mask]

            rho = _spearman_rho(vec_a_filt, vec_b_filt)
            tau = _kendall_tau(vec_a_filt, vec_b_filt)
            rho_lo, rho_hi = _bootstrap_ci(
                _spearman_rho,
                vec_a_filt,
                vec_b_filt,
                N_BOOTSTRAP,
                BOOTSTRAP_SEED,
            )
            tau_lo, tau_hi = _bootstrap_ci(
                _kendall_tau,
                vec_a_filt,
                vec_b_filt,
                N_BOOTSTRAP,
                BOOTSTRAP_SEED + 1,
            )

            rows.append(
                {
                    "model": model_label,
                    "scale_a": scale_a,
                    "scale_b": scale_b,
                    "scale_pair_label": pair_label,
                    "n_dimensions": int(mask.sum()),
                    "spearman_rho": rho,
                    "spearman_rho_ci_low": rho_lo,
                    "spearman_rho_ci_high": rho_hi,
                    "kendall_tau": tau,
                    "kendall_tau_ci_low": tau_lo,
                    "kendall_tau_ci_high": tau_hi,
                }
            )
    return pd.DataFrame(rows)


def build_f8_forest(metric_table: pd.DataFrame) -> plt.Figure:
    """Three-panel forest: 9 models x 3 scale pairs; Spearman rho + Kendall tau."""
    configure_plot_theme()
    plt.rcParams["font.family"] = [
        "Noto Sans CJK JP",
        *plt.rcParams.get("font.family", []),
    ]

    color_map = build_color_mapping(metric_table["model"].astype(str).unique())

    sort_pair = SCALE_PAIRS[0][2]
    sorted_frame = metric_table.loc[
        metric_table["scale_pair_label"] == sort_pair
    ].sort_values("spearman_rho", ascending=True)
    model_order = sorted_frame["model"].tolist()
    y_positions = np.arange(len(model_order))

    x_lo = float(metric_table["spearman_rho_ci_low"].min())
    x_hi = float(metric_table["spearman_rho_ci_high"].max())
    x_pad = 0.04 * (x_hi - x_lo)
    x_lim = (x_lo - x_pad, x_hi + x_pad)

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 5.5), sharex=True, sharey=True)
    for panel_idx, (_, _, pair_label) in enumerate(SCALE_PAIRS):
        axis = axes[panel_idx]
        panel_frame = (
            metric_table.loc[metric_table["scale_pair_label"] == pair_label]
            .set_index("model")
            .reindex(model_order)
            .reset_index()
        )

        axis.hlines(
            y=y_positions,
            xmin=panel_frame["spearman_rho_ci_low"],
            xmax=panel_frame["spearman_rho_ci_high"],
            color="#22303C",
            linewidth=1.4,
            alpha=0.85,
        )
        axis.scatter(
            panel_frame["spearman_rho"],
            y_positions,
            s=66,
            color=[color_map[m] for m in model_order],
            edgecolors="#22303C",
            linewidths=0.8,
            zorder=3,
        )

        axis.hlines(
            y=y_positions - 0.22,
            xmin=panel_frame["kendall_tau_ci_low"],
            xmax=panel_frame["kendall_tau_ci_high"],
            color="#8897A4",
            linewidth=1.0,
            alpha=0.85,
        )
        axis.scatter(
            panel_frame["kendall_tau"],
            y_positions - 0.22,
            s=44,
            facecolors="white",
            edgecolors="#22303C",
            linewidths=1.0,
            marker="D",
            zorder=3,
        )

        axis.set_yticks(y_positions)
        axis.set_yticklabels(model_order, fontsize=9.4)
        axis.axvline(
            0.0, color=SPINE_COLOR, linewidth=0.8, linestyle="--", zorder=1
        )
        axis.set_xlim(x_lim)
        axis.set_title(pair_label, fontsize=12, color=TEXT_COLOR)
        axis.set_xlabel(
            r"Cross-scale $\rho$ on 10-domain $d_z$ vector ($n=10$)"
        )
        axis.tick_params(axis="both", length=0)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=8,
            markerfacecolor="#2F3E4D",
            markeredgecolor="#22303C",
            label=r"Spearman $\rho$",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="#22303C",
            label=r"Kendall $\tau$",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    figure.suptitle(
        r"Dimension-level cross-scale similarity (Spearman $\rho$ on "
        r"10-domain $d_z$ vector)",
        fontsize=13.2,
        color=TEXT_COLOR,
        y=0.995,
    )
    figure.tight_layout(rect=(0.01, 0.06, 0.99, 0.96))
    return figure


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Build F5, F6, F8 figures and emit transparency CSVs."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    level1_long = load_level1_long()
    overall_long = load_overall_long()

    keep_cols = [
        "model_label",
        "scale_label",
        "scale_key",
        "一级-攻击领域",
        "n",
        "cohens_dz",
        "ci_low",
        "ci_high",
        "wilcoxon_p_fdr",
    ]
    level1_out_path = ARTIFACTS / "model_by_level1_by_scale.csv"
    level1_long[keep_cols].to_csv(level1_out_path, index=False)
    print(f"Wrote: {level1_out_path}")

    f5_fig = build_f5_heatmap(level1_long)
    f5_pdf = ARTIFACTS / "model_by_level1_by_scale_dz_heatmap.pdf"
    f5_png = ARTIFACTS / "model_by_level1_by_scale_dz_heatmap.png"
    f5_fig.savefig(f5_pdf, bbox_inches="tight")
    f5_fig.savefig(f5_png, dpi=220, bbox_inches="tight")
    plt.close(f5_fig)
    print(f"Wrote: {f5_pdf}")

    f6_fig = build_f6_scatter(overall_long)
    f6_pdf = ARTIFACTS / "overall_dz_cross_scale_scatter.pdf"
    f6_png = ARTIFACTS / "overall_dz_cross_scale_scatter.png"
    f6_fig.savefig(f6_pdf, bbox_inches="tight")
    f6_fig.savefig(f6_png, dpi=220, bbox_inches="tight")
    plt.close(f6_fig)
    print(f"Wrote: {f6_pdf}")

    print("\nComputing dimension-level cross-scale similarity (bootstrap n=2000) ...")
    dim_metric = compute_dim_level_cross_scale(level1_long)
    dim_out_path = ARTIFACTS / "dimension_level_cross_scale_similarity.csv"
    dim_metric.to_csv(dim_out_path, index=False)
    print(f"Wrote: {dim_out_path}")

    pivot = dim_metric.pivot(
        index="model", columns="scale_pair_label", values="spearman_rho"
    )[[lbl for _, _, lbl in SCALE_PAIRS]]
    pivot.loc["__median__"] = pivot.median()
    print("\nDimension-level cross-scale Spearman rho (models x pairs):")
    print(pivot.round(3).to_string())

    f8_fig = build_f8_forest(dim_metric)
    f8_pdf = ARTIFACTS / "dimension_level_cross_scale_rho_forest.pdf"
    f8_png = ARTIFACTS / "dimension_level_cross_scale_rho_forest.png"
    f8_fig.savefig(f8_pdf, bbox_inches="tight")
    f8_fig.savefig(f8_png, dpi=220, bbox_inches="tight")
    plt.close(f8_fig)
    print(f"Wrote: {f8_pdf}")


if __name__ == "__main__":
    main()
