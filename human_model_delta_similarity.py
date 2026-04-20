"""Item-level bias-pattern similarity between human raters and each LLM.

For each of the three response scales (attack_3pt, attack_7pt_likert,
attack_slider_0_100), this script computes, across the 371 paired sentences:

    Delta_i^human  = mean_female_score_i - mean_male_score_i   (across humans)
    Delta_i^model  = female_score_i - male_score_i              (per model)

and asks how similar the model's per-item bias vector is to the human's.
Primary metric: ICC(2,1) absolute-agreement (single rater, two-way random).
Also reports: ICC(3,1) consistency (two-way mixed, single rater).

Scale-specific robustness checks follow the user's plan:
  - 3-point scale:   quadratic-weighted Cohen's kappa on per-item Delta signs
                     binned into {-1, 0, +1}; Spearman rho as extra.
  - 7-point Likert:  Spearman rho on Delta_i values.
  - 0-100 slider:    Lin's concordance correlation coefficient; RMSE; MAE.

All metrics are reported with 95% bootstrap CIs resampling items with
replacement (n_boot = 2000 by default).

Outputs:
    artifacts/human_model_delta_similarity.csv    long-form metric table
    artifacts/human_model_delta_similarity.pdf    forest figure
    artifacts/human_model_delta_similarity.png    figure preview
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# Reuse visual tokens and model labels from the existing plotting module.
from plot_f1_brier import (
    BASE_FONT_FAMILY,
    MODEL_LABELS,
    MUTED_MODEL_COLORS,
    SPINE_COLOR,
    SUBTLE_TEXT_COLOR,
    TEXT_COLOR,
    build_color_mapping,
    configure_plot_theme,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
HUMAN_CSV = PROJECT_ROOT / "human" / "outputs" / "final_clean_long.csv"
MODEL_DIR = PROJECT_ROOT / "outputs" / "group_swap_1b" / "1b_groupswap_demensionsentence"
OUTPUT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_CSV = OUTPUT_DIR / "human_model_delta_similarity.csv"
OUTPUT_PDF = OUTPUT_DIR / "human_model_delta_similarity.pdf"
OUTPUT_PNG = OUTPUT_DIR / "human_model_delta_similarity.png"

SCALE_ORDER = [
    ("attack_3pt", "3-point"),
    ("attack_7pt_likert", "7-point Likert"),
    ("attack_slider_0_100", "0-100 slider"),
]

# Column suffix in the model-output csv differs per scale.
MODEL_FEMALE_COLS = {
    "attack_3pt": "女人版-attack_3pt攻击性评分",
    "attack_7pt_likert": "女人版-attack_7pt_likert攻击性评分",
    "attack_slider_0_100": "女人版-attack_slider_0_100攻击性评分",
}
MODEL_MALE_COLS = {
    "attack_3pt": "男人版-attack_3pt攻击性评分",
    "attack_7pt_likert": "男人版-attack_7pt_likert攻击性评分",
    "attack_slider_0_100": "男人版-attack_slider_0_100攻击性评分",
}

# Model filename pattern -> display label (matches Study 1a).
MODEL_FILE_LABELS = {
    "anthropic_claude-opus-4.5_results.csv": "Claude-4.5",
    "deepseek_deepseek-r1-0528_results.csv": "DeepSeek-R1",
    "deepseek_deepseek-v3.2-exp_results.csv": "DeepSeek-V3.2",
    "google_gemma-4-31b-it_results.csv": "Gemma-4-31B",
    "meta-llama_llama-4-maverick_results.csv": "Llama-4-Maverick",
    "moonshotai_kimi-k2-thinking_results.csv": "Kimi-K2",
    "openai_gpt-5.1_results.csv": "GPT-5.1",
    "qwen_qwen-2.5-72b-instruct_results.csv": "Qwen-2.5-72B",
    "z-ai_glm-4.6_results.csv": "GLM-4.6",
}

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260420


# ---------------------------------------------------------------------------
# Metric implementations (paired per-item vectors of length n_items)
# ---------------------------------------------------------------------------
def icc_two_raters(rater_a: np.ndarray, rater_b: np.ndarray, variant: str) -> float:
    """ICC for two raters scoring the same targets.

    Implements the closed-form expressions from Shrout & Fleiss (1979) for the
    two-rater case with each target rated by both.

    Parameters
    ----------
    rater_a, rater_b
        Equal-length 1-D arrays; NaNs must already be removed upstream.
    variant
        "2,1" for two-way random absolute agreement, single measurement.
        "3,1" for two-way mixed consistency, single measurement.
    """
    matrix = np.column_stack([rater_a, rater_b]).astype(float)
    n_targets, n_raters = matrix.shape
    assert n_raters == 2, "this implementation is specialised to two raters"

    target_means = matrix.mean(axis=1)
    rater_means = matrix.mean(axis=0)
    grand_mean = matrix.mean()

    ss_between_targets = n_raters * np.sum((target_means - grand_mean) ** 2)
    ss_between_raters = n_targets * np.sum((rater_means - grand_mean) ** 2)
    ss_total = np.sum((matrix - grand_mean) ** 2)
    ss_residual = ss_total - ss_between_targets - ss_between_raters

    ms_between_targets = ss_between_targets / (n_targets - 1)
    ms_between_raters = ss_between_raters / (n_raters - 1)
    ms_residual = ss_residual / ((n_targets - 1) * (n_raters - 1))

    if variant == "3,1":
        # Consistency: rater main effect is absorbed as fixed, not penalised.
        numerator = ms_between_targets - ms_residual
        denominator = ms_between_targets + (n_raters - 1) * ms_residual
    elif variant == "2,1":
        # Absolute agreement: between-rater variance enters the denominator.
        numerator = ms_between_targets - ms_residual
        denominator = (
            ms_between_targets
            + (n_raters - 1) * ms_residual
            + (n_raters / n_targets) * (ms_between_raters - ms_residual)
        )
    else:
        raise ValueError(f"unknown ICC variant: {variant}")

    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; wrapper using pandas for NaN handling."""
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def lin_ccc(a: np.ndarray, b: np.ndarray) -> float:
    """Lin's concordance correlation coefficient between two vectors."""
    mean_a = a.mean()
    mean_b = b.mean()
    var_a = a.var(ddof=0)
    var_b = b.var(ddof=0)
    covariance = ((a - mean_a) * (b - mean_b)).mean()
    denominator = var_a + var_b + (mean_a - mean_b) ** 2
    if denominator <= 0:
        return float("nan")
    return float(2.0 * covariance / denominator)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean squared difference between two vectors."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two vectors."""
    return float(np.mean(np.abs(a - b)))


def quadratic_weighted_kappa_on_signs(
    human_deltas: np.ndarray, model_deltas: np.ndarray
) -> float:
    """Quadratic-weighted Cohen's kappa on {-1, 0, +1} bias signs.

    On the 3-point scale, the per-item Delta is continuous but small-integer-
    valued, so we bin into sign classes (female higher / tied / male higher)
    to obtain the coarse-ordinal agreement the user specified.
    """

    def to_sign(values: np.ndarray) -> np.ndarray:
        return np.sign(values).astype(int)

    signs_human = to_sign(human_deltas)
    signs_model = to_sign(model_deltas)
    categories = np.array([-1, 0, 1])
    n_categories = len(categories)

    confusion = np.zeros((n_categories, n_categories), dtype=float)
    for s_h, s_m in zip(signs_human, signs_model):
        i = int(np.where(categories == s_h)[0][0])
        j = int(np.where(categories == s_m)[0][0])
        confusion[i, j] += 1.0

    total = confusion.sum()
    if total == 0:
        return float("nan")

    observed = confusion / total
    marginal_rows = observed.sum(axis=1, keepdims=True)
    marginal_cols = observed.sum(axis=0, keepdims=True)
    expected = marginal_rows @ marginal_cols

    weights = np.zeros((n_categories, n_categories), dtype=float)
    denom_weight = (n_categories - 1) ** 2
    for i in range(n_categories):
        for j in range(n_categories):
            weights[i, j] = ((i - j) ** 2) / denom_weight

    numerator = (weights * observed).sum()
    denominator = (weights * expected).sum()
    if denominator == 0:
        return float("nan")
    return float(1.0 - numerator / denominator)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def load_human_deltas() -> dict[str, pd.DataFrame]:
    """Return per-scale wide-form table with one column of mean(F)-mean(M)/item.

    Each row corresponds to one paired item (T0001..T0371). Human means are
    taken across all participants who completed the given scale.
    """
    human_long = pd.read_csv(HUMAN_CSV, low_memory=False)
    human_long["response_value"] = pd.to_numeric(
        human_long["response_value"], errors="coerce"
    )
    result_per_scale: dict[str, pd.DataFrame] = {}
    for scale_key, _ in SCALE_ORDER:
        scale_rows = human_long.loc[human_long["condition"] == scale_key].copy()
        # mean score per (item, shown_version)
        version_means = (
            scale_rows.groupby(["item_id", "shown_version"])["response_value"]
            .mean()
            .unstack("shown_version")
        )
        # enforce column order; may have Chinese labels
        female_column = "女人版"
        male_column = "男人版"
        delta = version_means[female_column] - version_means[male_column]
        delta.name = "delta_human"
        result_per_scale[scale_key] = delta.reset_index()
    return result_per_scale


def load_model_deltas() -> dict[str, pd.DataFrame]:
    """Return per-model wide-form per-item delta dataframes.

    The result maps "<model_display_label>" to a DataFrame with columns
    [item_id, delta_<scale>] for all three scales.
    """
    per_model: dict[str, pd.DataFrame] = {}
    for filename, display_label in MODEL_FILE_LABELS.items():
        model_path = MODEL_DIR / filename
        if not model_path.exists():
            continue
        frame = pd.read_csv(model_path, low_memory=False)
        # Build item_id from row_index: 0 -> T0001, 1 -> T0002, ...
        frame["item_id"] = frame["row_index"].apply(
            lambda ridx: f"T{int(ridx) + 1:04d}"
        )
        delta_cols: dict[str, pd.Series] = {}
        for scale_key, _ in SCALE_ORDER:
            female_column = MODEL_FEMALE_COLS[scale_key]
            male_column = MODEL_MALE_COLS[scale_key]
            if female_column not in frame or male_column not in frame:
                continue
            # Cast to float; non-numeric (e.g., refusal) -> NaN -> dropped later
            female_values = pd.to_numeric(frame[female_column], errors="coerce")
            male_values = pd.to_numeric(frame[male_column], errors="coerce")
            delta_cols[f"delta_{scale_key}"] = female_values - male_values
        merged = pd.DataFrame({"item_id": frame["item_id"], **delta_cols})
        per_model[display_label] = merged
    return per_model


# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------
def bootstrap_ci(
    metric_function,
    human_vector: np.ndarray,
    model_vector: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Return (ci_low, ci_high) via item-resampling bootstrap."""
    rng = np.random.default_rng(seed)
    n_items = len(human_vector)
    boot_values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        indices = rng.integers(0, n_items, size=n_items)
        try:
            boot_values[i] = metric_function(
                human_vector[indices], model_vector[indices]
            )
        except Exception:
            boot_values[i] = float("nan")
    valid = boot_values[~np.isnan(boot_values)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def compute_all_metrics() -> pd.DataFrame:
    """Build one long-form dataframe with all metrics for every (scale, model)."""
    human_deltas_per_scale = load_human_deltas()
    model_deltas = load_model_deltas()

    rows: list[dict] = []
    for scale_key, scale_label in SCALE_ORDER:
        human_frame = human_deltas_per_scale[scale_key].rename(
            columns={"delta_human": "delta_human"}
        )
        for model_label, model_frame in model_deltas.items():
            merged = human_frame.merge(
                model_frame[["item_id", f"delta_{scale_key}"]],
                on="item_id",
                how="inner",
            ).dropna()
            human_vector = merged["delta_human"].to_numpy(dtype=float)
            model_vector = merged[f"delta_{scale_key}"].to_numpy(dtype=float)
            n_items = len(merged)

            row = {
                "scale": scale_key,
                "scale_label": scale_label,
                "model": model_label,
                "n_items": n_items,
            }

            # Primary: ICC(2,1) absolute agreement.
            icc_21_value = icc_two_raters(human_vector, model_vector, "2,1")
            icc_21_low, icc_21_high = bootstrap_ci(
                lambda a, b: icc_two_raters(a, b, "2,1"),
                human_vector,
                model_vector,
                N_BOOTSTRAP,
                BOOTSTRAP_SEED,
            )
            row["icc_2_1"] = icc_21_value
            row["icc_2_1_ci_low"] = icc_21_low
            row["icc_2_1_ci_high"] = icc_21_high

            # Also: ICC(3,1) consistency.
            icc_31_value = icc_two_raters(human_vector, model_vector, "3,1")
            icc_31_low, icc_31_high = bootstrap_ci(
                lambda a, b: icc_two_raters(a, b, "3,1"),
                human_vector,
                model_vector,
                N_BOOTSTRAP,
                BOOTSTRAP_SEED + 1,
            )
            row["icc_3_1"] = icc_31_value
            row["icc_3_1_ci_low"] = icc_31_low
            row["icc_3_1_ci_high"] = icc_31_high

            # Robustness metrics, scale-dependent.
            if scale_key == "attack_3pt":
                kappa_value = quadratic_weighted_kappa_on_signs(
                    human_vector, model_vector
                )
                kappa_low, kappa_high = bootstrap_ci(
                    quadratic_weighted_kappa_on_signs,
                    human_vector,
                    model_vector,
                    N_BOOTSTRAP,
                    BOOTSTRAP_SEED + 2,
                )
                row["weighted_kappa"] = kappa_value
                row["weighted_kappa_ci_low"] = kappa_low
                row["weighted_kappa_ci_high"] = kappa_high

            rho_value = spearman_rho(human_vector, model_vector)
            rho_low, rho_high = bootstrap_ci(
                spearman_rho, human_vector, model_vector, N_BOOTSTRAP, BOOTSTRAP_SEED + 3
            )
            row["spearman_rho"] = rho_value
            row["spearman_rho_ci_low"] = rho_low
            row["spearman_rho_ci_high"] = rho_high

            if scale_key == "attack_slider_0_100":
                ccc_value = lin_ccc(human_vector, model_vector)
                ccc_low, ccc_high = bootstrap_ci(
                    lin_ccc, human_vector, model_vector, N_BOOTSTRAP, BOOTSTRAP_SEED + 4
                )
                rmse_value = rmse(human_vector, model_vector)
                rmse_low, rmse_high = bootstrap_ci(
                    rmse, human_vector, model_vector, N_BOOTSTRAP, BOOTSTRAP_SEED + 5
                )
                mae_value = mae(human_vector, model_vector)
                mae_low, mae_high = bootstrap_ci(
                    mae, human_vector, model_vector, N_BOOTSTRAP, BOOTSTRAP_SEED + 6
                )
                row["lin_ccc"] = ccc_value
                row["lin_ccc_ci_low"] = ccc_low
                row["lin_ccc_ci_high"] = ccc_high
                row["rmse"] = rmse_value
                row["rmse_ci_low"] = rmse_low
                row["rmse_ci_high"] = rmse_high
                row["mae"] = mae_value
                row["mae_ci_low"] = mae_low
                row["mae_ci_high"] = mae_high

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_forest_figure(metric_table: pd.DataFrame) -> plt.Figure:
    """One forest panel per scale, models sorted by ICC(2,1) descending."""
    configure_plot_theme()
    color_mapping = build_color_mapping(metric_table["model"].unique())

    figure, axes = plt.subplots(1, 3, figsize=(11.6, 5.4), sharex=False)
    axes = np.atleast_1d(axes)

    for panel_index, (scale_key, scale_label) in enumerate(SCALE_ORDER):
        axis = axes[panel_index]
        panel_frame = metric_table.loc[metric_table["scale"] == scale_key].copy()
        panel_frame = panel_frame.sort_values("icc_2_1", ascending=True)
        model_order = panel_frame["model"].tolist()
        y_positions = np.arange(len(model_order))

        # ICC(2,1) primary (filled dots).
        icc21 = panel_frame["icc_2_1"].to_numpy()
        icc21_low = panel_frame["icc_2_1_ci_low"].to_numpy()
        icc21_high = panel_frame["icc_2_1_ci_high"].to_numpy()
        axis.hlines(
            y=y_positions,
            xmin=icc21_low,
            xmax=icc21_high,
            color="#22303C",
            linewidth=1.4,
            alpha=0.85,
        )
        axis.scatter(
            icc21,
            y_positions,
            s=60,
            color=[color_mapping[model_name] for model_name in model_order],
            edgecolors="#22303C",
            linewidths=0.8,
            zorder=3,
            label="ICC(2,1) absolute",
        )

        # ICC(3,1) secondary (hollow diamonds), offset vertically -0.25.
        icc31 = panel_frame["icc_3_1"].to_numpy()
        icc31_low = panel_frame["icc_3_1_ci_low"].to_numpy()
        icc31_high = panel_frame["icc_3_1_ci_high"].to_numpy()
        axis.hlines(
            y=y_positions - 0.22,
            xmin=icc31_low,
            xmax=icc31_high,
            color="#8897A4",
            linewidth=1.1,
            alpha=0.85,
        )
        axis.scatter(
            icc31,
            y_positions - 0.22,
            s=46,
            facecolors="white",
            edgecolors="#22303C",
            linewidths=1.0,
            marker="D",
            zorder=3,
            label="ICC(3,1) consistency",
        )

        axis.set_yticks(y_positions)
        axis.set_yticklabels(model_order, fontsize=9.4)
        axis.set_title(scale_label, fontsize=12, color=TEXT_COLOR)
        axis.axvline(0.0, color=SPINE_COLOR, linewidth=0.8, linestyle="--", zorder=1)
        axis.set_xlabel(r"ICC (bias-pattern $\Delta_i$)")
        axis.tick_params(axis="both", length=0)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

        # Shared legend once, at figure level.
        if panel_index == 0:
            legend_handles = [
                Line2D([0], [0], marker="o", linestyle="none",
                       markersize=8, markerfacecolor="#2F3E4D",
                       markeredgecolor="#22303C",
                       label="ICC(2,1) absolute agreement"),
                Line2D([0], [0], marker="D", linestyle="none",
                       markersize=7, markerfacecolor="white",
                       markeredgecolor="#22303C",
                       label="ICC(3,1) consistency"),
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
        r"Human-vs-model bias-pattern similarity (per-item $\Delta_i = F - M$)",
        fontsize=13.6,
        color=TEXT_COLOR,
        y=0.995,
    )
    figure.tight_layout(rect=(0.01, 0.05, 0.99, 0.96))
    return figure


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full pipeline and write outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_table = compute_all_metrics()
    metric_table.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote: {OUTPUT_CSV}")
    print(metric_table[["scale_label", "model", "n_items", "icc_2_1", "icc_3_1"]]
          .to_string(index=False))

    figure = build_forest_figure(metric_table)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote: {OUTPUT_PDF}")
    print(f"Wrote: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
