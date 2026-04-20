from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FactorAnalysis


SCRIPT_DIR = Path(__file__).resolve().parent
HUMAN_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT_LONG_PATH = HUMAN_DIR / "outputs" / "final_clean_long.csv"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
COMMON_CJK_FONT_FAMILIES = [
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
    "Droid Sans Fallback",
    "AR PL UMing CN",
    "DejaVu Sans",
]
CONDITION_SCALE_LIMITS = {
    "attack_3pt": (0.0, 2.0),
    "attack_7pt_likert": (0.0, 6.0),
    "attack_slider_0_100": (0.0, 100.0),
}
VERSION_ORDER = ["男人版", "女人版"]
REQUIRED_COLUMNS = [
    "session_id",
    "condition",
    "response_value",
    "is_repeat_trial",
    "template_id",
    "shown_version",
    "shown_text",
    "dimension_1",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run item-level EFA on the original 371x2 sentence pool using the final cleaned human participant data."
        )
    )
    parser.add_argument(
        "--input-long-path",
        type=Path,
        default=DEFAULT_INPUT_LONG_PATH,
        help="Path to the final cleaned long-format participant file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root directory for outputs. Defaults to human/efa_item_level_model/outputs.",
    )
    parser.add_argument(
        "--shown-version",
        choices=["both", "男人版", "女人版"],
        default="both",
        help="Run EFA for one shown version or for both versions separately.",
    )
    parser.add_argument(
        "--include-repeat-trials",
        action="store_true",
        help="Include repeat trials. By default, repeat trials are excluded.",
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=None,
        help="Optional manual number of factors. By default, the Kaiser criterion (eigenvalue > 1) is used.",
    )
    parser.add_argument(
        "--min-item-observations",
        type=int,
        default=50,
        help="Minimum number of participants who must rate an item for the item to be retained.",
    )
    parser.add_argument(
        "--matrix-rank",
        type=int,
        default=20,
        help="Rank used in iterative low-rank imputation before EFA.",
    )
    parser.add_argument(
        "--imputation-iterations",
        type=int,
        default=25,
        help="Number of low-rank imputation iterations.",
    )
    parser.add_argument(
        "--loading-threshold",
        type=float,
        default=0.30,
        help="Threshold used when exporting salient loading tables.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260419,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    rcParams["font.sans-serif"] = COMMON_CJK_FONT_FAMILIES
    rcParams["font.family"] = "sans-serif"
    rcParams["axes.unicode_minus"] = False


def normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    normalized_text = str(value).strip()
    if not normalized_text or normalized_text.lower() in {"nan", "none"}:
        return None
    return normalized_text


def ensure_required_columns(dataframe: pd.DataFrame, required_columns: list[str], context: str) -> None:
    missing_columns = [column_name for column_name in required_columns if column_name not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {context}: {missing_columns}")


def scale_response(row: pd.Series) -> float:
    lower_limit, upper_limit = CONDITION_SCALE_LIMITS[row["condition"]]
    return (row["response_value_num"] - lower_limit) / (upper_limit - lower_limit)


def load_sentence_level_long_frame(input_long_path: Path, include_repeat_trials: bool) -> pd.DataFrame:
    logging.info("Loading long-format file: %s", input_long_path)
    long_frame = pd.read_csv(input_long_path, encoding="utf-8-sig", dtype=str, low_memory=False)
    ensure_required_columns(long_frame, REQUIRED_COLUMNS, str(input_long_path))

    if not include_repeat_trials:
        long_frame = long_frame.loc[
            long_frame["is_repeat_trial"].astype(str).str.upper() == "FALSE"
        ].copy()

    long_frame["response_value_num"] = pd.to_numeric(long_frame["response_value"], errors="coerce")
    long_frame = long_frame.loc[long_frame["response_value_num"].notna()].copy()
    long_frame["condition"] = long_frame["condition"].map(normalize_text)
    long_frame = long_frame.loc[long_frame["condition"].isin(CONDITION_SCALE_LIMITS)].copy()

    for column_name in ["template_id", "shown_version", "shown_text", "dimension_1"]:
        long_frame[column_name] = long_frame[column_name].map(normalize_text)
    long_frame = long_frame.dropna(subset=["template_id", "shown_version", "shown_text", "dimension_1"])
    long_frame["session_id"] = long_frame["session_id"].astype(str)
    long_frame["normalized_score"] = long_frame.apply(scale_response, axis=1)
    long_frame["sentence_key"] = long_frame["template_id"] + "||" + long_frame["shown_version"]

    logging.info(
        "Loaded %s participants, %s rows, and %s unique sentence items.",
        long_frame["session_id"].nunique(),
        len(long_frame),
        long_frame["sentence_key"].nunique(),
    )
    return long_frame


def build_sentence_metadata(long_frame: pd.DataFrame) -> pd.DataFrame:
    metadata_frame = (
        long_frame[
            ["sentence_key", "template_id", "shown_version", "shown_text", "dimension_1"]
        ]
        .drop_duplicates()
        .sort_values("sentence_key")
        .reset_index(drop=True)
    )
    return metadata_frame


def build_participant_sentence_matrix(long_frame: pd.DataFrame) -> pd.DataFrame:
    participant_sentence_matrix = (
        long_frame.groupby(["session_id", "sentence_key"], dropna=False)["normalized_score"]
        .mean()
        .unstack()
        .sort_index(axis=1)
    )
    logging.info(
        "Constructed participant-by-sentence matrix with shape %s x %s.",
        participant_sentence_matrix.shape[0],
        participant_sentence_matrix.shape[1],
    )
    return participant_sentence_matrix


def filter_to_one_version(long_frame: pd.DataFrame, shown_version: str) -> pd.DataFrame:
    version_frame = long_frame.loc[long_frame["shown_version"] == shown_version].copy()
    version_frame["sentence_key"] = version_frame["template_id"]
    logging.info(
        "Prepared %s subset with %s participants, %s rows, and %s sentence items.",
        shown_version,
        version_frame["session_id"].nunique(),
        len(version_frame),
        version_frame["sentence_key"].nunique(),
    )
    return version_frame


def filter_items_by_observations(
    participant_sentence_matrix: pd.DataFrame,
    metadata_frame: pd.DataFrame,
    min_item_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    item_observation_count = participant_sentence_matrix.notna().sum(axis=0)
    retained_item_names = item_observation_count[item_observation_count >= min_item_observations].index.tolist()
    filtered_matrix = participant_sentence_matrix.loc[:, retained_item_names].copy()
    filtered_metadata = metadata_frame.loc[metadata_frame["sentence_key"].isin(retained_item_names)].copy()
    item_summary_frame = pd.DataFrame(
        {
            "sentence_key": item_observation_count.index,
            "n_participants_observed": item_observation_count.values,
            "coverage_rate": item_observation_count.values / participant_sentence_matrix.shape[0],
        }
    )
    filtered_metadata = filtered_metadata.merge(item_summary_frame, on="sentence_key", how="left")
    filtered_metadata = filtered_metadata.sort_values("sentence_key").reset_index(drop=True)

    logging.info(
        "Retained %s of %s sentence items using min_item_observations=%s.",
        len(retained_item_names),
        participant_sentence_matrix.shape[1],
        min_item_observations,
    )
    return filtered_matrix, filtered_metadata, item_summary_frame


def standardize_observed_matrix(filtered_matrix: pd.DataFrame) -> tuple[np.ndarray, pd.Series, pd.Series]:
    column_means = filtered_matrix.mean(axis=0)
    column_sds = filtered_matrix.std(axis=0, ddof=1).replace(0.0, 1.0).fillna(1.0)
    standardized_matrix = (filtered_matrix - column_means) / column_sds
    return standardized_matrix.to_numpy(dtype=float), column_means, column_sds


def iterative_low_rank_imputation(
    standardized_values: np.ndarray,
    matrix_rank: int,
    imputation_iterations: int,
) -> np.ndarray:
    observed_mask = ~np.isnan(standardized_values)
    filled_values = standardized_values.copy()
    column_means = np.nanmean(filled_values, axis=0)
    column_means = np.where(np.isnan(column_means), 0.0, column_means)
    missing_locations = ~observed_mask
    filled_values[missing_locations] = np.take(column_means, np.where(missing_locations)[1])

    effective_rank = min(matrix_rank, min(filled_values.shape) - 1)
    if effective_rank < 1:
        raise ValueError("The requested matrix rank is too small for the observed data shape.")

    for _ in range(imputation_iterations):
        singular_vectors_left, singular_values, singular_vectors_right = np.linalg.svd(
            filled_values,
            full_matrices=False,
        )
        reconstructed_values = (
            singular_vectors_left[:, :effective_rank]
            @ np.diag(singular_values[:effective_rank])
            @ singular_vectors_right[:effective_rank, :]
        )
        filled_values[missing_locations] = reconstructed_values[missing_locations]
        filled_values[observed_mask] = standardized_values[observed_mask]

    return filled_values


def compute_eigenvalue_table(imputed_standardized_values: np.ndarray) -> pd.DataFrame:
    correlation_matrix = np.corrcoef(imputed_standardized_values, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(correlation_matrix)[::-1]
    eigenvalue_table = pd.DataFrame(
        {
            "factor_index": np.arange(1, len(eigenvalues) + 1),
            "eigenvalue": eigenvalues,
            "retain_kaiser": eigenvalues > 1.0,
        }
    )
    eigenvalue_table["variance_ratio"] = eigenvalue_table["eigenvalue"] / len(eigenvalues)
    eigenvalue_table["cumulative_variance_ratio"] = eigenvalue_table["variance_ratio"].cumsum()
    return eigenvalue_table


def choose_number_of_factors(
    eigenvalue_table: pd.DataFrame,
    requested_n_factors: int | None,
) -> tuple[int, str]:
    if requested_n_factors is not None:
        return max(1, int(requested_n_factors)), "manual"

    chosen_n_factors = int(eigenvalue_table["retain_kaiser"].sum())
    if chosen_n_factors < 1:
        chosen_n_factors = 1
    return chosen_n_factors, "kaiser_eigen_gt_1"


def varimax_rotation(
    loading_matrix: np.ndarray,
    gamma: float = 1.0,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_columns = loading_matrix.shape
    rotation_matrix = np.eye(n_columns)
    previous_objective = 0.0

    for _ in range(max_iterations):
        rotated_loadings = loading_matrix @ rotation_matrix
        singular_vectors_left, singular_values, singular_vectors_right = np.linalg.svd(
            loading_matrix.T
            @ (
                rotated_loadings**3
                - (gamma / n_rows) * rotated_loadings @ np.diag(np.diag(rotated_loadings.T @ rotated_loadings))
            )
        )
        rotation_matrix = singular_vectors_left @ singular_vectors_right
        objective_value = singular_values.sum()
        if previous_objective != 0 and objective_value - previous_objective < tolerance:
            break
        previous_objective = objective_value

    return loading_matrix @ rotation_matrix, rotation_matrix


def align_factor_signs(loading_matrix: np.ndarray, score_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aligned_loadings = loading_matrix.copy()
    aligned_scores = score_matrix.copy()
    for factor_index in range(aligned_loadings.shape[1]):
        largest_loading_index = int(np.argmax(np.abs(aligned_loadings[:, factor_index])))
        if aligned_loadings[largest_loading_index, factor_index] < 0:
            aligned_loadings[:, factor_index] *= -1
            aligned_scores[:, factor_index] *= -1
    return aligned_loadings, aligned_scores


def fit_rotated_factor_model(
    imputed_standardized_values: np.ndarray,
    sentence_names: list[str],
    participant_ids: list[str],
    n_factors: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor_model = FactorAnalysis(n_components=n_factors, random_state=random_seed)
    unrotated_scores = factor_model.fit_transform(imputed_standardized_values)
    unrotated_loadings = factor_model.components_.T
    rotated_loadings, rotation_matrix = varimax_rotation(unrotated_loadings)
    rotated_scores = unrotated_scores @ rotation_matrix
    rotated_loadings, rotated_scores = align_factor_signs(rotated_loadings, rotated_scores)

    factor_names = [f"Factor_{factor_index + 1}" for factor_index in range(n_factors)]
    loading_frame = pd.DataFrame(rotated_loadings, index=sentence_names, columns=factor_names)
    loading_frame["communality"] = (loading_frame[factor_names] ** 2).sum(axis=1)
    loading_frame["uniqueness"] = np.clip(1.0 - loading_frame["communality"], 0.0, 1.0)
    loading_frame["primary_factor"] = loading_frame[factor_names].abs().idxmax(axis=1)
    loading_frame["primary_loading"] = [
        loading_frame.loc[sentence_name, factor_name]
        for sentence_name, factor_name in loading_frame["primary_factor"].items()
    ]
    loading_frame["primary_loading_abs"] = loading_frame["primary_loading"].abs()

    factor_score_frame = pd.DataFrame(rotated_scores, index=participant_ids, columns=factor_names).reset_index(names="session_id")
    return loading_frame.reset_index(names="sentence_key"), factor_score_frame


def build_factor_dimension_crosstab(loading_frame: pd.DataFrame) -> pd.DataFrame:
    crosstab_frame = pd.crosstab(loading_frame["dimension_1"], loading_frame["primary_factor"])
    return crosstab_frame


def build_hungarian_mapping(crosstab_frame: pd.DataFrame) -> pd.DataFrame:
    if crosstab_frame.empty:
        return pd.DataFrame(columns=["dimension_1", "matched_factor", "matched_count"])

    cost_matrix = -crosstab_frame.to_numpy(dtype=float)
    row_indices, column_indices = linear_sum_assignment(cost_matrix)
    mapping_rows: list[dict[str, object]] = []
    for row_index, column_index in zip(row_indices, column_indices, strict=True):
        mapping_rows.append(
            {
                "dimension_1": crosstab_frame.index[row_index],
                "matched_factor": crosstab_frame.columns[column_index],
                "matched_count": int(crosstab_frame.iat[row_index, column_index]),
            }
        )
    return pd.DataFrame(mapping_rows)


def build_dimension_alignment_summary(
    loading_frame: pd.DataFrame,
    crosstab_frame: pd.DataFrame,
    mapping_frame: pd.DataFrame,
) -> pd.DataFrame:
    total_items = len(loading_frame)
    mapped_count = int(mapping_frame["matched_count"].sum()) if not mapping_frame.empty else 0
    factor_majority_rows: list[dict[str, object]] = []
    for factor_name in crosstab_frame.columns:
        factor_counts = crosstab_frame[factor_name]
        majority_dimension = str(factor_counts.idxmax())
        majority_count = int(factor_counts.max())
        factor_total = int(factor_counts.sum())
        factor_majority_rows.append(
            {
                "factor_name": factor_name,
                "majority_dimension_1": majority_dimension,
                "majority_count": majority_count,
                "factor_total_items": factor_total,
                "factor_purity": majority_count / factor_total if factor_total else np.nan,
            }
        )

    summary_frame = pd.DataFrame(
        {
            "metric_name": [
                "total_sentence_items",
                "matched_items_under_hungarian_mapping",
                "matched_ratio_under_hungarian_mapping",
                "mean_factor_purity",
            ],
            "metric_value": [
                total_items,
                mapped_count,
                mapped_count / total_items if total_items else np.nan,
                float(pd.DataFrame(factor_majority_rows)["factor_purity"].mean()) if factor_majority_rows else np.nan,
            ],
        }
    )
    return summary_frame, pd.DataFrame(factor_majority_rows)


def save_crosstab_heatmap(crosstab_frame: pd.DataFrame, output_path: Path) -> None:
    figure_width = max(9.0, 0.9 * len(crosstab_frame.columns) + 4.0)
    figure_height = max(6.0, 0.45 * len(crosstab_frame.index) + 3.0)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    sns.heatmap(
        crosstab_frame,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Sentence count"},
        ax=axis,
    )
    axis.set_xlabel("EFA primary factor")
    axis.set_ylabel("Original dimension_1")
    axis.set_title("Original 10 dimensions vs sentence-level EFA factors")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_factor_composition_heatmap(crosstab_frame: pd.DataFrame, output_path: Path) -> None:
    column_normalized_frame = crosstab_frame.div(crosstab_frame.sum(axis=0), axis=1).fillna(0.0)
    figure_width = max(9.0, 0.9 * len(column_normalized_frame.columns) + 4.0)
    figure_height = max(6.0, 0.45 * len(column_normalized_frame.index) + 3.0)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    sns.heatmap(
        column_normalized_frame,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        linewidths=0.5,
        cbar_kws={"label": "Within-factor proportion"},
        ax=axis,
    )
    axis.set_xlabel("EFA primary factor")
    axis.set_ylabel("Original dimension_1")
    axis.set_title("Within-factor composition by original dimension")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_salient_loading_table(
    loading_frame: pd.DataFrame,
    loading_threshold: float,
) -> pd.DataFrame:
    factor_names = [column_name for column_name in loading_frame.columns if column_name.startswith("Factor_")]
    long_frame = loading_frame.melt(
        id_vars=[
            "sentence_key",
            "template_id",
            "shown_version",
            "shown_text",
            "dimension_1",
            "n_participants_observed",
            "coverage_rate",
            "communality",
            "uniqueness",
            "primary_factor",
            "primary_loading",
            "primary_loading_abs",
        ],
        value_vars=factor_names,
        var_name="factor_name",
        value_name="loading",
    )
    long_frame["absolute_loading"] = long_frame["loading"].abs()
    long_frame = long_frame.loc[long_frame["absolute_loading"] >= loading_threshold].copy()
    long_frame = long_frame.sort_values(
        ["factor_name", "absolute_loading", "sentence_key"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return long_frame


def build_default_output_dir(output_root: Path, shown_version: str, factor_mode: str) -> Path:
    version_slug = "male_version" if shown_version == "男人版" else "female_version"
    return output_root / f"{version_slug}_{factor_mode}_efa"


def build_summary_markdown(
    loading_frame: pd.DataFrame,
    alignment_summary_frame: pd.DataFrame,
    factor_majority_frame: pd.DataFrame,
    mapping_frame: pd.DataFrame,
    eigenvalue_table: pd.DataFrame,
    chosen_n_factors: int,
    factor_selection_method: str,
) -> str:
    summary_lookup = alignment_summary_frame.set_index("metric_name")["metric_value"].to_dict()
    lines = [
        "# Sentence-Level EFA Summary",
        "",
        f"- Sentence items retained: `{int(summary_lookup['total_sentence_items'])}`",
        f"- Factor selection method: `{factor_selection_method}`",
        f"- Chosen number of factors: `{chosen_n_factors}`",
        f"- Number of eigenvalues > 1: `{int(eigenvalue_table['retain_kaiser'].sum())}`",
        f"- Matched ratio under optimal factor-to-dimension mapping: `{summary_lookup['matched_ratio_under_hungarian_mapping']:.4f}`",
        f"- Mean factor purity: `{summary_lookup['mean_factor_purity']:.4f}`",
        "",
        "## Leading eigenvalues",
    ]
    for row in eigenvalue_table.head(15).itertuples(index=False):
        lines.append(
            f"- `Factor {int(row.factor_index)}`: eigenvalue `{row.eigenvalue:.4f}`, retain_kaiser `{bool(row.retain_kaiser)}`"
        )

    lines.extend([
        "",
        "## Factor majorities",
    ])
    for row in factor_majority_frame.itertuples(index=False):
        lines.append(
            f"- `{row.factor_name}`: majority dimension `{row.majority_dimension_1}`, "
            f"purity `{row.factor_purity:.4f}` ({row.majority_count}/{row.factor_total_items})"
        )

    lines.extend(["", "## Optimal factor-to-dimension mapping"])
    for row in mapping_frame.itertuples(index=False):
        lines.append(
            f"- `{row.dimension_1}` -> `{row.matched_factor}` with `{row.matched_count}` sentence items"
        )

    lines.extend(["", "## Highest-loading example items"])
    factor_names = [column_name for column_name in loading_frame.columns if column_name.startswith("Factor_")]
    for factor_name in factor_names:
        factor_subset = loading_frame.sort_values(factor_name, ascending=False).head(5)
        lines.append(f"### {factor_name}")
        for row in factor_subset.itertuples(index=False):
            lines.append(
                f"- `{row.sentence_key}` | `{row.dimension_1}` | loading `{getattr(row, factor_name):.4f}` | {row.shown_text}"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def save_scree_plot(eigenvalue_table: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    axis.plot(
        eigenvalue_table["factor_index"],
        eigenvalue_table["eigenvalue"],
        marker="o",
        linewidth=2.0,
        color="#4C78A8",
    )
    axis.axhline(1.0, color="#C44E52", linestyle="--", linewidth=1.2, label="Eigenvalue = 1")
    axis.set_xlabel("Factor index")
    axis.set_ylabel("Eigenvalue")
    axis.set_title("Scree plot with Kaiser criterion")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_one_version(long_frame: pd.DataFrame, shown_version: str, args: argparse.Namespace, output_root: Path) -> None:
    factor_mode = "kaiser" if args.n_factors is None else f"{args.n_factors}factor"
    output_dir = build_default_output_dir(output_root, shown_version, factor_mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    version_frame = filter_to_one_version(long_frame, shown_version)
    metadata_frame = build_sentence_metadata(version_frame)
    participant_sentence_matrix = build_participant_sentence_matrix(version_frame)
    filtered_matrix, filtered_metadata, item_summary_frame = filter_items_by_observations(
        participant_sentence_matrix=participant_sentence_matrix,
        metadata_frame=metadata_frame,
        min_item_observations=args.min_item_observations,
    )

    standardized_values, _, _ = standardize_observed_matrix(filtered_matrix)
    imputed_standardized_values = iterative_low_rank_imputation(
        standardized_values=standardized_values,
        matrix_rank=args.matrix_rank,
        imputation_iterations=args.imputation_iterations,
    )
    eigenvalue_table = compute_eigenvalue_table(imputed_standardized_values)
    chosen_n_factors, factor_selection_method = choose_number_of_factors(
        eigenvalue_table=eigenvalue_table,
        requested_n_factors=args.n_factors,
    )
    loading_frame, factor_score_frame = fit_rotated_factor_model(
        imputed_standardized_values=imputed_standardized_values,
        sentence_names=filtered_matrix.columns.tolist(),
        participant_ids=filtered_matrix.index.tolist(),
        n_factors=chosen_n_factors,
        random_seed=args.random_seed,
    )

    loading_frame = loading_frame.merge(filtered_metadata, on="sentence_key", how="left")
    loading_frame = loading_frame.sort_values(
        ["primary_factor", "primary_loading_abs", "sentence_key"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    crosstab_frame = build_factor_dimension_crosstab(loading_frame)
    mapping_frame = build_hungarian_mapping(crosstab_frame)
    alignment_summary_frame, factor_majority_frame = build_dimension_alignment_summary(
        loading_frame=loading_frame,
        crosstab_frame=crosstab_frame,
        mapping_frame=mapping_frame,
    )
    salient_loading_frame = build_salient_loading_table(
        loading_frame=loading_frame,
        loading_threshold=args.loading_threshold,
    )

    filtered_matrix.to_csv(output_dir / "participant_sentence_matrix_observed.csv", encoding="utf-8-sig")
    pd.DataFrame(
        imputed_standardized_values,
        index=filtered_matrix.index,
        columns=filtered_matrix.columns,
    ).to_csv(output_dir / "participant_sentence_matrix_imputed_standardized.csv", encoding="utf-8-sig")
    metadata_frame.to_csv(output_dir / "sentence_metadata_all.csv", index=False, encoding="utf-8-sig")
    item_summary_frame.to_csv(output_dir / "sentence_observation_summary.csv", index=False, encoding="utf-8-sig")
    loading_frame.to_csv(output_dir / "sentence_factor_loadings.csv", index=False, encoding="utf-8-sig")
    salient_loading_frame.to_csv(output_dir / "sentence_factor_loadings_salient.csv", index=False, encoding="utf-8-sig")
    factor_score_frame.to_csv(output_dir / "participant_factor_scores.csv", index=False, encoding="utf-8-sig")
    eigenvalue_table.to_csv(output_dir / "eigenvalues_kaiser_table.csv", index=False, encoding="utf-8-sig")
    crosstab_frame.to_csv(output_dir / "original_dimension_by_factor_crosstab.csv", encoding="utf-8-sig")
    mapping_frame.to_csv(output_dir / "optimal_factor_dimension_mapping.csv", index=False, encoding="utf-8-sig")
    alignment_summary_frame.to_csv(output_dir / "dimension_alignment_summary.csv", index=False, encoding="utf-8-sig")
    factor_majority_frame.to_csv(output_dir / "factor_majority_dimensions.csv", index=False, encoding="utf-8-sig")

    save_crosstab_heatmap(crosstab_frame, output_dir / "original_dimension_by_factor_heatmap.png")
    save_factor_composition_heatmap(crosstab_frame, output_dir / "factor_composition_by_original_dimension.png")
    save_scree_plot(eigenvalue_table, output_dir / "scree_plot_kaiser.png")
    summary_markdown = build_summary_markdown(
        loading_frame=loading_frame,
        alignment_summary_frame=alignment_summary_frame,
        factor_majority_frame=factor_majority_frame,
        mapping_frame=mapping_frame,
        eigenvalue_table=eigenvalue_table,
        chosen_n_factors=chosen_n_factors,
        factor_selection_method=factor_selection_method,
    )
    (output_dir / "summary.md").write_text(summary_markdown, encoding="utf-8")

    matched_ratio = float(
        alignment_summary_frame.loc[
            alignment_summary_frame["metric_name"] == "matched_ratio_under_hungarian_mapping",
            "metric_value",
        ].iloc[0]
    )
    mean_factor_purity = float(
        alignment_summary_frame.loc[
            alignment_summary_frame["metric_name"] == "mean_factor_purity",
            "metric_value",
        ].iloc[0]
    )

    logging.info("%s sentence-level EFA finished successfully.", shown_version)
    logging.info("%s retained sentence items: %s", shown_version, len(loading_frame))
    logging.info("%s chosen number of factors: %s", shown_version, chosen_n_factors)
    logging.info("%s matched ratio under optimal mapping: %.4f", shown_version, matched_ratio)
    logging.info("%s mean factor purity: %.4f", shown_version, mean_factor_purity)


def main() -> None:
    configure_logging()
    configure_plot_style()
    args = parse_arguments()
    output_root = args.output_dir or DEFAULT_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    long_frame = load_sentence_level_long_frame(
        input_long_path=args.input_long_path,
        include_repeat_trials=args.include_repeat_trials,
    )
    version_list = VERSION_ORDER if args.shown_version == "both" else [args.shown_version]
    for shown_version in version_list:
        run_one_version(long_frame=long_frame, shown_version=shown_version, args=args, output_root=output_root)


if __name__ == "__main__":
    main()
