# Generate two derived artifacts for the formal paper (slides/paper.tex):
#   (1) Per-cell two-sided permutation p-values for Spearman rho over 10 domains.
#       Output:  artifacts/paper_revision/domain_spearman_permutation_pvals.csv
#   (2) An exploratory scatter of LLM 3-point d_z against approximate parameter
#       count, coloured by tuning regime. Output:
#       artifacts/paper_revision/fig_r7_llm_3pt_dz_by_params_tuning.pdf and .png
#
# Both products feed Q5 (LLM-to-LLM variability) and Q7 (domain-Spearman chance
# baseline) write-ups in slides/paper.tex. The permutation null for Q7 is the
# distribution of Spearman rho between two random rank vectors of length 10;
# its shape does not depend on the data, so the null is computed once and
# applied to every (reference, scale, model) cell in the existing
# artifacts/paper_revision/domain_level_human_model_similarity.csv.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS = PROJECT_ROOT / "artifacts" / "paper_revision"
DOMAIN_SIM_CSV = ARTIFACTS / "domain_level_human_model_similarity.csv"
PVAL_OUT_CSV = ARTIFACTS / "domain_spearman_permutation_pvals.csv"
SCATTER_PDF = ARTIFACTS / "fig_r7_llm_3pt_dz_by_params_tuning.pdf"
SCATTER_PNG = ARTIFACTS / "fig_r7_llm_3pt_dz_by_params_tuning.png"


# ---------------------------------------------------------------------------
# (1) Permutation p-values for Spearman rho over 10 attack domains.
# ---------------------------------------------------------------------------

def compute_permutation_null(n_items: int = 10, n_sim: int = 100_000, seed: int = 0) -> np.ndarray:
    """Sample the null Spearman-rho distribution between two random rankings.

    The null does not depend on the observed data: for a fixed item count, the
    Spearman rho between two independent random permutations has a closed
    distribution. We use Monte Carlo for clarity.
    """
    rng = np.random.default_rng(seed)
    null = np.empty(n_sim, dtype=np.float64)
    for index in range(n_sim):
        permutation_a = rng.permutation(n_items)
        permutation_b = rng.permutation(n_items)
        null[index] = spearmanr(permutation_a, permutation_b).statistic
    return null


def attach_two_sided_pvalues(rho_values: np.ndarray, null_distribution: np.ndarray) -> np.ndarray:
    """For every observed rho, compute the two-sided permutation p value.

    p_two_sided = mean( |null| >= |observed| ); a Laplace add-one smoothing is
    applied to avoid exact zeros for ties at high rho.
    """
    abs_null = np.abs(null_distribution)
    abs_obs = np.abs(rho_values)
    p_two_sided = np.empty_like(abs_obs)
    n_null = abs_null.size
    for index, value in enumerate(abs_obs):
        n_extreme = int(np.sum(abs_null >= value))
        p_two_sided[index] = (n_extreme + 1) / (n_null + 1)
    return p_two_sided


def write_pvalue_csv() -> None:
    """Read the existing per-cell Spearman csv and add a permutation-p column."""
    domain_similarity = pl.read_csv(DOMAIN_SIM_CSV)
    null_distribution = compute_permutation_null(n_items=10, n_sim=100_000, seed=0)

    rho_array = domain_similarity["spearman_rho"].to_numpy()
    p_values = attach_two_sided_pvalues(rho_array, null_distribution)

    null_ci_lower, null_ci_upper = np.quantile(null_distribution, [0.025, 0.975])

    with_pvalues = domain_similarity.with_columns(
        pl.Series("spearman_perm_p_two_sided", p_values),
        pl.Series("null_ci_lower_95", np.full_like(rho_array, null_ci_lower)),
        pl.Series("null_ci_upper_95", np.full_like(rho_array, null_ci_upper)),
        pl.Series(
            "exceeds_null_cutoff",
            np.abs(rho_array) > max(abs(null_ci_lower), abs(null_ci_upper)),
        ),
    )
    with_pvalues.write_csv(PVAL_OUT_CSV)

    print(
        f"[Q7] wrote per-cell permutation p csv  -> {PVAL_OUT_CSV.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"     null 95% CI for n=10 Spearman: [{null_ci_lower:+.3f}, {null_ci_upper:+.3f}]"
    )
    n_cells_above_cutoff = int(with_pvalues["exceeds_null_cutoff"].sum())
    n_cells_total = with_pvalues.height
    print(
        f"     cells with |rho| above null cutoff: {n_cells_above_cutoff} / {n_cells_total}"
    )


# ---------------------------------------------------------------------------
# (2) Exploratory scatter of LLM 3-point d_z by parameter count and tuning.
# ---------------------------------------------------------------------------

# Approximate published parameter counts and tuning regime labels for the nine
# production LLMs that appear in Table 1 / Table 2 of paper.tex. Closed-source
# models (Claude, GPT) are reported as None for parameter count and the scatter
# falls back to a "closed" bucket on the x axis.
LLM_PANEL_METADATA: list[dict[str, object]] = [
    {"model": "DeepSeek-R1",      "dz_3pt": 0.289, "dz_7pt": 0.353, "dz_slider": 0.409, "params_b": 671.0, "tuning": "reasoning_RL"},
    {"model": "DeepSeek-V3.2",    "dz_3pt": 0.289, "dz_7pt": 0.371, "dz_slider": 0.423, "params_b": 685.0, "tuning": "instruct"},
    {"model": "GLM-4.6",          "dz_3pt": 0.297, "dz_7pt": 0.422, "dz_slider": 0.460, "params_b": 355.0, "tuning": "instruct"},
    {"model": "Llama-4-Maverick", "dz_3pt": 0.379, "dz_7pt": 0.477, "dz_slider": 0.510, "params_b": 400.0, "tuning": "instruct_MoE"},
    {"model": "Gemma-4-31B",      "dz_3pt": 0.408, "dz_7pt": 0.773, "dz_slider": 1.016, "params_b": 31.0,  "tuning": "instruct"},
    {"model": "Claude-Opus-4.5",  "dz_3pt": 0.498, "dz_7pt": 0.505, "dz_slider": 0.593, "params_b": None,  "tuning": "safety_RLHF"},
    {"model": "Qwen-2.5-72B",     "dz_3pt": 0.553, "dz_7pt": 0.578, "dz_slider": 0.686, "params_b": 72.0,  "tuning": "instruct"},
    {"model": "Kimi-K2-Thinking", "dz_3pt": 0.572, "dz_7pt": 0.693, "dz_slider": 0.714, "params_b": 1000.0, "tuning": "reasoning_RL"},
    {"model": "GPT-5.1",          "dz_3pt": 0.653, "dz_7pt": 0.840, "dz_slider": 0.748, "params_b": None,  "tuning": "safety_RLHF"},
]


def build_scatter_dataframe() -> pl.DataFrame:
    """Assemble the long-form dataframe used for the exploratory scatter."""
    return pl.DataFrame(LLM_PANEL_METADATA)


def plot_dz_by_parameters_and_tuning() -> None:
    """Render an exploratory scatter that is explicitly non-inferential.

    The x axis uses a log-transformed parameter count (closed-source models
    plotted as a separate "closed" stripe). The y axis is 3-point d_z. The
    marker shape encodes tuning regime. No regression or significance test is
    drawn -- the figure is a descriptive aid for the Discussion section, and
    the figure caption in paper.tex states this explicitly.
    """
    sns.set_theme(context="paper", style="whitegrid")

    scatter_data = build_scatter_dataframe()
    open_panel = scatter_data.filter(pl.col("params_b").is_not_null()).with_columns(
        pl.col("params_b").log10().alias("log_params_b"),
    )
    closed_panel = scatter_data.filter(pl.col("params_b").is_null())

    figure, axis = plt.subplots(figsize=(6.0, 4.2))

    tuning_palette = {
        "instruct": "#1f77b4",
        "instruct_MoE": "#17becf",
        "reasoning_RL": "#2ca02c",
        "safety_RLHF": "#d62728",
    }
    tuning_marker = {
        "instruct": "o",
        "instruct_MoE": "D",
        "reasoning_RL": "s",
        "safety_RLHF": "^",
    }

    # Plot open-source models on the log-axis.
    for (tuning_label,), group in open_panel.group_by(["tuning"], maintain_order=True):
        axis.scatter(
            group["log_params_b"].to_list(),
            group["dz_3pt"].to_list(),
            label=tuning_label,
            color=tuning_palette[tuning_label],
            marker=tuning_marker[tuning_label],
            s=80,
            edgecolor="black",
            linewidth=0.4,
        )
        for row in group.iter_rows(named=True):
            axis.annotate(
                row["model"],
                xy=(row["log_params_b"], row["dz_3pt"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
            )

    # Plot closed-source models on a dedicated right-hand stripe.
    closed_anchor_x = np.log10(2000.0)
    for (tuning_label,), group in closed_panel.group_by(["tuning"], maintain_order=True):
        axis.scatter(
            [closed_anchor_x] * group.height,
            group["dz_3pt"].to_list(),
            label=f"{tuning_label} (closed)",
            color=tuning_palette[tuning_label],
            marker=tuning_marker[tuning_label],
            s=80,
            edgecolor="black",
            linewidth=0.4,
        )
        for row in group.iter_rows(named=True):
            axis.annotate(
                row["model"],
                xy=(closed_anchor_x, row["dz_3pt"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
            )

    axis.set_xlabel("Parameter count (log10 of billions; closed-source on right stripe)")
    axis.set_ylabel(r"3-point $d_z$ (LLM rating of $\Delta_{F-M}$)")
    axis.set_title("Exploratory: 3-point $d_z$ by parameter count and tuning regime")
    axis.set_xlim(1.0, 3.6)
    axis.axvspan(np.log10(1500.0), 3.6, color="lightgray", alpha=0.4, zorder=0)
    axis.text(
        np.log10(1700.0),
        axis.get_ylim()[1] - 0.05,
        "closed-source",
        fontsize=8,
        color="gray",
        ha="left",
        va="top",
    )

    axis.legend(title="Tuning regime", loc="lower right", fontsize=8, title_fontsize=8)

    figure.tight_layout()
    figure.savefig(SCATTER_PDF)
    figure.savefig(SCATTER_PNG, dpi=200)
    plt.close(figure)

    print(f"[Q5] wrote exploratory scatter (PDF)  -> {SCATTER_PDF.relative_to(PROJECT_ROOT)}")
    print(f"[Q5] wrote exploratory scatter (PNG)  -> {SCATTER_PNG.relative_to(PROJECT_ROOT)}")


def main() -> None:
    write_pvalue_csv()
    plot_dz_by_parameters_and_tuning()


if __name__ == "__main__":
    main()
