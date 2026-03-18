from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import polars as pl
from xlsxwriter import Workbook


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FRAMEWORK_COLUMNS: Sequence[str] = (
    "一级-攻击领域",
    "二级-攻击属性",
    "三级-攻击表达策略",
    "具体内容",
    "具体内容-男人版",
    "具体内容-女人版",
)
DIMENSION_COLUMNS: Sequence[str] = FRAMEWORK_COLUMNS[:3]
BASE_SENTENCE_COLUMN = "具体内容"
MALE_SENTENCE_COLUMN = "具体内容-男人版"
FEMALE_SENTENCE_COLUMN = "具体内容-女人版"

MALE_ZEROSHOT_COLUMN = "男人版-zeroshot攻击性评分"
MALE_COT_COLUMN = "男人版-cot攻击性评分"
MALE_COT_REASON_COLUMN = "男人版-cot攻击性reason"
FEMALE_ZEROSHOT_COLUMN = "女人版-zeroshot攻击性评分"
FEMALE_COT_COLUMN = "女人版-cot攻击性评分"
FEMALE_COT_REASON_COLUMN = "女人版-cot攻击性reason"

ZERO_SHOT_SETTING = "zeroshot"
COT_SETTING = "cot"
SCORE_COLUMN_MAP = {
    ZERO_SHOT_SETTING: (MALE_ZEROSHOT_COLUMN, FEMALE_ZEROSHOT_COLUMN),
    COT_SETTING: (MALE_COT_COLUMN, FEMALE_COT_COLUMN),
}
SCORE_REASON_COLUMN_MAP = {
    COT_SETTING: (MALE_COT_REASON_COLUMN, FEMALE_COT_REASON_COLUMN),
}

AGGREGATED_SCORE_PATTERN = re.compile(r"^(?P<model_prefix>.+)_group_swap_scores\.xlsx$")
ROW_SCORE_PATTERN = re.compile(
    r"^(?P<model_prefix>.+)_group_swap_row_(?P<row_index>\d+)\.xlsx$"
)
ROW_SOURCE_NAME = "row_files"
AGGREGATED_SOURCE_NAME = "aggregated_file"
ANALYSIS_WORKBOOK_NAME = "1b_groupswap_analysis.xlsx"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the 1b group-swap analysis."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze group-swap scoring results with preserved level-1/2/3 "
            "dimensions and export sentence-level / topic-level summaries."
        )
    )
    parser.add_argument(
        "--framework-path",
        type=Path,
        default=Path("data/性别反转生成维度框架.xlsx"),
        help="Path to the dimension framework Excel file.",
    )
    parser.add_argument(
        "--score-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing group-swap score files. If omitted, the script "
            "tries outputs/group_swap/<framework_stem> first, then falls back to "
            "outputs/group_swap/1b_groupswap_demensionsentence."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for analysis outputs. Defaults to "
            "<score_dir>/analysis_1b."
        ),
    )
    parser.add_argument(
        "--top-k-alignments",
        type=int,
        default=5,
        help="Number of nearest female-sentence matches to keep for each male sentence.",
    )
    return parser.parse_args()


def normalize_text_value(cell_value: object) -> str | None:
    """Normalize text values so joins remain stable across Excel readers."""
    if cell_value is None:
        return None
    if isinstance(cell_value, float) and pd.isna(cell_value):
        return None

    normalized_text = str(cell_value).strip()
    if not normalized_text or normalized_text.lower() in {"nan", "none"}:
        return None
    return normalized_text


def load_excel_with_pandas(excel_path: Path) -> pd.DataFrame:
    """Read an Excel file with pandas for compatibility with the current environment."""
    logger.info("Reading Excel file: %s", excel_path)
    return pd.read_excel(excel_path)


def resolve_score_directory(
    framework_path: Path, requested_score_dir: Path | None
) -> Path:
    """Resolve the directory that stores group-swap scoring outputs."""
    if requested_score_dir is not None:
        return requested_score_dir

    preferred_directory = Path("outputs/group_swap") / framework_path.stem
    if preferred_directory.exists():
        logger.info("Using score directory derived from framework stem: %s", preferred_directory)
        return preferred_directory

    fallback_directory = Path("outputs/group_swap/1b_groupswap_demensionsentence")
    logger.info("Using fallback score directory: %s", fallback_directory)
    return fallback_directory


def ensure_required_columns(
    dataframe: pd.DataFrame, required_columns: Iterable[str], context_label: str
) -> None:
    """Validate that all required columns are present in a DataFrame."""
    missing_columns = [column_name for column_name in required_columns if column_name not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {context_label}: {missing_columns}")


def load_framework_frame(framework_path: Path) -> pl.DataFrame:
    """Load the dimension framework and forward-fill the hierarchical labels."""
    framework_pandas_frame = load_excel_with_pandas(framework_path)
    ensure_required_columns(framework_pandas_frame, FRAMEWORK_COLUMNS, str(framework_path))

    framework_pandas_frame = framework_pandas_frame.loc[:, list(FRAMEWORK_COLUMNS)].copy()
    framework_pandas_frame.insert(0, "框架Excel行号", framework_pandas_frame.index + 2)
    framework_pandas_frame.loc[:, list(DIMENSION_COLUMNS)] = framework_pandas_frame.loc[
        :, list(DIMENSION_COLUMNS)
    ].ffill()

    for text_column in FRAMEWORK_COLUMNS:
        framework_pandas_frame[text_column] = framework_pandas_frame[text_column].map(normalize_text_value)

    framework_pandas_frame = framework_pandas_frame.dropna(
        subset=[MALE_SENTENCE_COLUMN, FEMALE_SENTENCE_COLUMN]
    ).reset_index(drop=True)
    framework_pandas_frame.insert(0, "句对ID", range(1, len(framework_pandas_frame) + 1))

    logger.info(
        "Loaded framework rows=%s unique_level1=%s unique_level2=%s unique_level3=%s",
        len(framework_pandas_frame),
        framework_pandas_frame["一级-攻击领域"].nunique(dropna=True),
        framework_pandas_frame["二级-攻击属性"].nunique(dropna=True),
        framework_pandas_frame["三级-攻击表达策略"].nunique(dropna=True),
    )
    return pl.from_pandas(framework_pandas_frame)


def discover_score_sources(score_directory: Path) -> Dict[str, Dict[str, List[Path]]]:
    """Discover aggregated files and per-row files for each model prefix."""
    discovered_sources: Dict[str, Dict[str, List[Path]]] = defaultdict(
        lambda: {AGGREGATED_SOURCE_NAME: [], ROW_SOURCE_NAME: []}
    )

    aggregated_paths = sorted(
        Path(candidate_path)
        for candidate_path in glob(str(score_directory / "*_group_swap_scores.xlsx"))
    )
    row_file_paths = sorted(
        Path(candidate_path)
        for candidate_path in glob(str(score_directory / "*_group_swap_row_*.xlsx"))
    )

    for aggregated_path in aggregated_paths:
        matched_result = AGGREGATED_SCORE_PATTERN.match(aggregated_path.name)
        if matched_result is None:
            logger.warning("Skipping unrecognized aggregated score file: %s", aggregated_path)
            continue
        model_prefix = matched_result.group("model_prefix")
        discovered_sources[model_prefix][AGGREGATED_SOURCE_NAME].append(aggregated_path)

    for row_file_path in row_file_paths:
        matched_result = ROW_SCORE_PATTERN.match(row_file_path.name)
        if matched_result is None:
            logger.warning("Skipping unrecognized row score file: %s", row_file_path)
            continue
        model_prefix = matched_result.group("model_prefix")
        discovered_sources[model_prefix][ROW_SOURCE_NAME].append(row_file_path)

    if not discovered_sources:
        raise FileNotFoundError(
            f"No group-swap score files were found in directory: {score_directory}"
        )

    logger.info(
        "Discovered score sources for %s model prefixes in %s",
        len(discovered_sources),
        score_directory,
    )
    return dict(discovered_sources)


def prepare_score_pandas_frame(score_pandas_frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize score columns and drop fully empty sentence pairs."""
    required_columns = [
        MALE_SENTENCE_COLUMN,
        FEMALE_SENTENCE_COLUMN,
        MALE_ZEROSHOT_COLUMN,
        MALE_COT_COLUMN,
        FEMALE_ZEROSHOT_COLUMN,
        FEMALE_COT_COLUMN,
    ]
    ensure_required_columns(score_pandas_frame, required_columns, "score file")

    optional_reason_columns = [MALE_COT_REASON_COLUMN, FEMALE_COT_REASON_COLUMN]
    retained_columns = required_columns + [
        column_name for column_name in optional_reason_columns if column_name in score_pandas_frame.columns
    ]
    score_pandas_frame = score_pandas_frame.loc[:, retained_columns].copy()

    for sentence_column in (MALE_SENTENCE_COLUMN, FEMALE_SENTENCE_COLUMN):
        score_pandas_frame[sentence_column] = score_pandas_frame[sentence_column].map(normalize_text_value)

    for numeric_column in (
        MALE_ZEROSHOT_COLUMN,
        MALE_COT_COLUMN,
        FEMALE_ZEROSHOT_COLUMN,
        FEMALE_COT_COLUMN,
    ):
        score_pandas_frame[numeric_column] = pd.to_numeric(
            score_pandas_frame[numeric_column], errors="coerce"
        )

    for reason_column in optional_reason_columns:
        if reason_column in score_pandas_frame.columns:
            score_pandas_frame[reason_column] = score_pandas_frame[reason_column].map(
                normalize_text_value
            )

    score_pandas_frame = score_pandas_frame.dropna(
        subset=[MALE_SENTENCE_COLUMN, FEMALE_SENTENCE_COLUMN], how="all"
    )
    score_pandas_frame = score_pandas_frame.drop_duplicates(
        subset=[MALE_SENTENCE_COLUMN, FEMALE_SENTENCE_COLUMN], keep="last"
    ).reset_index(drop=True)
    return score_pandas_frame


def load_aggregated_score_frame(aggregated_path: Path) -> pl.DataFrame:
    """Load a pre-aggregated score workbook into a standardized Polars frame."""
    aggregated_pandas_frame = prepare_score_pandas_frame(load_excel_with_pandas(aggregated_path))
    return pl.from_pandas(aggregated_pandas_frame)


def load_row_score_frame(row_file_paths: Sequence[Path]) -> pl.DataFrame:
    """Load per-row score workbooks and concatenate them into one Polars frame."""
    row_pandas_frames = [prepare_score_pandas_frame(load_excel_with_pandas(row_file_path)) for row_file_path in row_file_paths]
    if not row_pandas_frames:
        return pl.DataFrame()
    concatenated_pandas_frame = pd.concat(row_pandas_frames, ignore_index=True)
    concatenated_pandas_frame = concatenated_pandas_frame.drop_duplicates(
        subset=[MALE_SENTENCE_COLUMN, FEMALE_SENTENCE_COLUMN], keep="last"
    ).reset_index(drop=True)
    return pl.from_pandas(concatenated_pandas_frame)


def select_best_score_frame(score_sources: Dict[str, Dict[str, List[Path]]]) -> pl.DataFrame:
    """Choose the best available score source for each model prefix."""
    selected_score_frames: List[pl.DataFrame] = []

    for model_prefix, source_map in sorted(score_sources.items()):
        candidate_frames: List[tuple[int, int, str, pl.DataFrame]] = []

        if source_map[AGGREGATED_SOURCE_NAME]:
            aggregated_path = source_map[AGGREGATED_SOURCE_NAME][0]
            aggregated_frame = load_aggregated_score_frame(aggregated_path).with_columns(
                pl.lit(model_prefix).alias("model_prefix"),
                pl.lit(AGGREGATED_SOURCE_NAME).alias("score_source"),
            )
            candidate_frames.append(
                (aggregated_frame.height, 0, AGGREGATED_SOURCE_NAME, aggregated_frame)
            )

        if source_map[ROW_SOURCE_NAME]:
            row_frame = load_row_score_frame(source_map[ROW_SOURCE_NAME]).with_columns(
                pl.lit(model_prefix).alias("model_prefix"),
                pl.lit(ROW_SOURCE_NAME).alias("score_source"),
            )
            candidate_frames.append((row_frame.height, 1, ROW_SOURCE_NAME, row_frame))

        if not candidate_frames:
            continue

        candidate_frames.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected_row_count, _, selected_source_name, selected_frame = candidate_frames[0]
        logger.info(
            "Selected %s for model_prefix=%s rows=%s",
            selected_source_name,
            model_prefix,
            selected_row_count,
        )
        selected_score_frames.append(selected_frame)

    if not selected_score_frames:
        raise FileNotFoundError("Score files were discovered but none could be loaded successfully.")

    return pl.concat(selected_score_frames, how="vertical_relaxed")


def merge_framework_and_scores(
    framework_frame: pl.DataFrame, score_frame: pl.DataFrame
) -> pl.DataFrame:
    """Join score outputs back to the framework so every sentence keeps its dimensions."""
    model_frame = score_frame.select(["model_prefix", "score_source"]).unique().sort("model_prefix")
    merged_frame = (
        framework_frame.join(model_frame, how="cross")
        .join(
            score_frame,
            on=["model_prefix", "score_source", MALE_SENTENCE_COLUMN, FEMALE_SENTENCE_COLUMN],
            how="left",
        )
        .with_columns(
        pl.when(
            pl.col(MALE_ZEROSHOT_COLUMN).is_not_null()
            & pl.col(FEMALE_ZEROSHOT_COLUMN).is_not_null()
        )
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("zeroshot_已评分"),
        pl.when(pl.col(MALE_COT_COLUMN).is_not_null() & pl.col(FEMALE_COT_COLUMN).is_not_null())
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("cot_已评分"),
    ))

    logger.info(
        "Merged framework rows=%s matched_zeroshot=%s matched_cot=%s",
        merged_frame.height,
        merged_frame.select(pl.col("zeroshot_已评分").sum()).item(),
        merged_frame.select(pl.col("cot_已评分").sum()).item(),
    )
    return merged_frame


def build_same_sentence_difference_frame(merged_frame: pl.DataFrame) -> pl.DataFrame:
    """Build the sentence-level male-vs-female comparison table for each setting."""
    difference_frames: List[pl.DataFrame] = []
    shared_columns = [
        "句对ID",
        "框架Excel行号",
        *DIMENSION_COLUMNS,
        BASE_SENTENCE_COLUMN,
        MALE_SENTENCE_COLUMN,
        FEMALE_SENTENCE_COLUMN,
        "model_prefix",
        "score_source",
    ]

    for setting_name, (male_score_column, female_score_column) in SCORE_COLUMN_MAP.items():
        current_frame = (
            merged_frame.filter(
                pl.col("model_prefix").is_not_null()
                & pl.col(male_score_column).is_not_null()
                & pl.col(female_score_column).is_not_null()
            )
            .select(
                shared_columns
                + [
                    pl.lit(setting_name).alias("setting"),
                    pl.col(male_score_column).alias("男人版攻击性评分"),
                    pl.col(female_score_column).alias("女人版攻击性评分"),
                ]
            )
            .with_columns(
                (pl.col("男人版攻击性评分") - pl.col("女人版攻击性评分")).alias("男人减女人分差"),
                (pl.col("男人版攻击性评分") - pl.col("女人版攻击性评分")).abs().alias("绝对分差"),
                pl.when(pl.col("男人版攻击性评分") > pl.col("女人版攻击性评分"))
                .then(pl.lit("男人版更高"))
                .when(pl.col("男人版攻击性评分") < pl.col("女人版攻击性评分"))
                .then(pl.lit("女人版更高"))
                .otherwise(pl.lit("两者相同"))
                .alias("分差方向"),
            )
        )
        current_frame = current_frame.with_columns(
            pl.lit(None).cast(pl.String).alias("男人版reason"),
            pl.lit(None).cast(pl.String).alias("女人版reason"),
        )

        if setting_name in SCORE_REASON_COLUMN_MAP:
            male_reason_column, female_reason_column = SCORE_REASON_COLUMN_MAP[setting_name]
            if male_reason_column in merged_frame.columns and female_reason_column in merged_frame.columns:
                reason_frame = merged_frame.select(
                    [
                        "句对ID",
                        "model_prefix",
                        pl.col(male_reason_column).alias("男人版reason_new"),
                        pl.col(female_reason_column).alias("女人版reason_new"),
                    ]
                )
                current_frame = (
                    current_frame.join(
                        reason_frame,
                        on=["句对ID", "model_prefix"],
                        how="left",
                    )
                    .with_columns(
                        pl.coalesce(pl.col("男人版reason_new"), pl.col("男人版reason")).alias("男人版reason"),
                        pl.coalesce(pl.col("女人版reason_new"), pl.col("女人版reason")).alias("女人版reason"),
                    )
                    .drop(["男人版reason_new", "女人版reason_new"])
                )

        difference_frames.append(current_frame)

    difference_frame = pl.concat(difference_frames, how="vertical_relaxed").sort(
        by=["model_prefix", "setting", "绝对分差", "句对ID"],
        descending=[False, False, True, False],
    )
    logger.info("Built same-sentence difference rows=%s", difference_frame.height)
    return difference_frame


def build_gender_score_long_frame(merged_frame: pl.DataFrame) -> pl.DataFrame:
    """Reshape the merged frame so male and female sentence scores live in one column."""
    long_frames: List[pl.DataFrame] = []
    shared_columns = [
        "句对ID",
        "框架Excel行号",
        *DIMENSION_COLUMNS,
        BASE_SENTENCE_COLUMN,
        "model_prefix",
        "score_source",
    ]

    for setting_name, (male_score_column, female_score_column) in SCORE_COLUMN_MAP.items():
        male_frame = (
            merged_frame.filter(pl.col("model_prefix").is_not_null() & pl.col(male_score_column).is_not_null())
            .select(
                shared_columns
                + [
                    pl.lit(setting_name).alias("setting"),
                    pl.lit("男人版").alias("版本"),
                    pl.col(MALE_SENTENCE_COLUMN).alias("句子"),
                    pl.col(male_score_column).alias("攻击性评分"),
                ]
            )
        )
        female_frame = (
            merged_frame.filter(
                pl.col("model_prefix").is_not_null() & pl.col(female_score_column).is_not_null()
            )
            .select(
                shared_columns
                + [
                    pl.lit(setting_name).alias("setting"),
                    pl.lit("女人版").alias("版本"),
                    pl.col(FEMALE_SENTENCE_COLUMN).alias("句子"),
                    pl.col(female_score_column).alias("攻击性评分"),
                ]
            )
        )
        long_frames.extend([male_frame, female_frame])

    long_frame = pl.concat(long_frames, how="vertical_relaxed")
    logger.info("Built long gender score rows=%s", long_frame.height)
    return long_frame


def add_within_group_rank(
    ranking_frame: pl.DataFrame, grouping_columns: Sequence[str]
) -> pl.DataFrame:
    """Assign a stable rank after sorting within each grouping slice."""
    ranked_group_frames: List[pl.DataFrame] = []
    for grouped_frame in ranking_frame.partition_by(list(grouping_columns), maintain_order=True):
        ranked_group_frames.append(grouped_frame.with_row_index("主题排名", offset=1))
    return pl.concat(ranked_group_frames, how="vertical_relaxed") if ranked_group_frames else ranking_frame


def build_topic_ranking_frame(long_gender_score_frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate attack indices for male and female versions across level 1/2/3 themes."""
    ranking_frames: List[pl.DataFrame] = []
    dimension_level_map = [
        ("一级维度", "一级-攻击领域"),
        ("二级维度", "二级-攻击属性"),
        ("三级维度", "三级-攻击表达策略"),
    ]

    for dimension_level_name, dimension_column in dimension_level_map:
        ranking_frame = (
            long_gender_score_frame.group_by(
                ["model_prefix", "setting", "版本", dimension_column], maintain_order=False
            )
            .agg(
                pl.len().alias("句子数量"),
                pl.col("攻击性评分").mean().round(6).alias("攻击指数均值"),
                pl.col("攻击性评分").median().alias("攻击指数中位数"),
                pl.col("攻击性评分").std().round(6).alias("攻击指数标准差"),
            )
            .rename({dimension_column: "主题"})
            .with_columns(pl.lit(dimension_level_name).alias("维度层级"))
            .sort(
                by=["model_prefix", "setting", "版本", "攻击指数均值", "句子数量", "主题"],
                descending=[False, False, False, True, True, False],
            )
        )
        ranking_frames.append(
            add_within_group_rank(
                ranking_frame,
                grouping_columns=["model_prefix", "setting", "版本", "维度层级"],
            )
        )

    topic_ranking_frame = pl.concat(ranking_frames, how="vertical_relaxed")
    logger.info("Built topic ranking rows=%s", topic_ranking_frame.height)
    return topic_ranking_frame


def build_alignment_frame(
    merged_frame: pl.DataFrame, top_k_alignments: int
) -> pl.DataFrame:
    """Find the nearest female-sentence matches for each male sentence by score."""
    alignment_frames: List[pl.DataFrame] = []

    for setting_name, (male_score_column, female_score_column) in SCORE_COLUMN_MAP.items():
        available_frame = merged_frame.filter(
            pl.col("model_prefix").is_not_null()
            & pl.col(male_score_column).is_not_null()
            & pl.col(female_score_column).is_not_null()
        )
        if available_frame.is_empty():
            continue

        for grouped_frame in available_frame.partition_by(["model_prefix"], maintain_order=True):
            model_prefix = grouped_frame["model_prefix"][0]
            model_pandas_frame = pd.DataFrame(grouped_frame.to_dicts())

            male_candidates = model_pandas_frame[
                [
                    "句对ID",
                    "框架Excel行号",
                    *DIMENSION_COLUMNS,
                    MALE_SENTENCE_COLUMN,
                    male_score_column,
                ]
            ].rename(columns={
                "句对ID": "male_pair_id",
                "框架Excel行号": "male_excel_row",
                "一级-攻击领域": "male_level1",
                "二级-攻击属性": "male_level2",
                "三级-攻击表达策略": "male_level3",
                MALE_SENTENCE_COLUMN: "male_sentence",
                male_score_column: "male_score",
            })
            female_candidates = model_pandas_frame[
                [
                    "句对ID",
                    "框架Excel行号",
                    *DIMENSION_COLUMNS,
                    FEMALE_SENTENCE_COLUMN,
                    female_score_column,
                ]
            ].rename(columns={
                "句对ID": "female_pair_id",
                "框架Excel行号": "female_excel_row",
                "一级-攻击领域": "female_level1",
                "二级-攻击属性": "female_level2",
                "三级-攻击表达策略": "female_level3",
                FEMALE_SENTENCE_COLUMN: "female_sentence",
                female_score_column: "female_score",
            })

            matched_rows: List[dict] = []
            for male_row in male_candidates.itertuples(index=False):
                candidate_frame = female_candidates[
                    female_candidates["female_pair_id"] != male_row.male_pair_id
                ].copy()
                if candidate_frame.empty:
                    continue

                candidate_frame["score_gap_abs"] = (
                    candidate_frame["female_score"] - male_row.male_score
                ).abs()
                candidate_frame["same_level1"] = (
                    candidate_frame["female_level1"] == male_row.male_level1
                )
                candidate_frame["same_level2"] = (
                    candidate_frame["female_level2"] == male_row.male_level2
                )
                candidate_frame["same_level3"] = (
                    candidate_frame["female_level3"] == male_row.male_level3
                )

                candidate_frame = candidate_frame.sort_values(
                    by=[
                        "score_gap_abs",
                        "same_level1",
                        "same_level2",
                        "same_level3",
                        "female_score",
                        "female_pair_id",
                    ],
                    ascending=[True, True, True, True, False, True],
                ).head(top_k_alignments)

                for alignment_rank, candidate_row in enumerate(
                    candidate_frame.itertuples(index=False), start=1
                ):
                    matched_rows.append(
                        {
                            "model_prefix": model_prefix,
                            "setting": setting_name,
                            "对齐候选排名": alignment_rank,
                            "男人句对ID": male_row.male_pair_id,
                            "男人框架Excel行号": male_row.male_excel_row,
                            "男人一级-攻击领域": male_row.male_level1,
                            "男人二级-攻击属性": male_row.male_level2,
                            "男人三级-攻击表达策略": male_row.male_level3,
                            "男人句子": male_row.male_sentence,
                            "男人攻击性评分": male_row.male_score,
                            "女人句对ID": candidate_row.female_pair_id,
                            "女人框架Excel行号": candidate_row.female_excel_row,
                            "女人一级-攻击领域": candidate_row.female_level1,
                            "女人二级-攻击属性": candidate_row.female_level2,
                            "女人三级-攻击表达策略": candidate_row.female_level3,
                            "女人句子": candidate_row.female_sentence,
                            "女人攻击性评分": candidate_row.female_score,
                            "分差绝对值": candidate_row.score_gap_abs,
                            "是否同一级维度": candidate_row.same_level1,
                            "是否同二级维度": candidate_row.same_level2,
                            "是否同三级维度": candidate_row.same_level3,
                        }
                    )

            if matched_rows:
                alignment_frames.append(pl.from_pandas(pd.DataFrame(matched_rows)))

    if not alignment_frames:
        return pl.DataFrame()

    alignment_frame = pl.concat(alignment_frames, how="vertical_relaxed").sort(
        by=["model_prefix", "setting", "男人句对ID", "对齐候选排名"]
    )
    logger.info("Built alignment rows=%s", alignment_frame.height)
    return alignment_frame


def write_csv_outputs(
    output_directory: Path,
    merged_frame: pl.DataFrame,
    difference_frame: pl.DataFrame,
    topic_ranking_frame: pl.DataFrame,
    alignment_frame: pl.DataFrame,
) -> None:
    """Write analysis tables as CSV files for easy downstream use."""
    output_directory.mkdir(parents=True, exist_ok=True)
    merged_frame.write_csv(output_directory / "merged_scores_with_dimensions.csv")
    difference_frame.write_csv(output_directory / "same_sentence_gender_differences.csv")
    topic_ranking_frame.write_csv(output_directory / "topic_rankings.csv")
    if not alignment_frame.is_empty():
        alignment_frame.write_csv(output_directory / "aligned_cross_gender_pairs.csv")


def write_excel_workbook(
    workbook_path: Path,
    merged_frame: pl.DataFrame,
    difference_frame: pl.DataFrame,
    topic_ranking_frame: pl.DataFrame,
    alignment_frame: pl.DataFrame,
) -> None:
    """Write all analysis tables into one workbook with multiple sheets."""
    workbook_path.parent.mkdir(parents=True, exist_ok=True)

    with Workbook(str(workbook_path)) as workbook:
        merged_frame.write_excel(
            workbook=workbook,
            worksheet="merged_scores",
            table_name="MergedScores",
            autofit=True,
            hide_gridlines=True,
        )
        difference_frame.write_excel(
            workbook=workbook,
            worksheet="same_sentence_diff",
            table_name="SameSentenceDiff",
            autofit=True,
            hide_gridlines=True,
        )

        for sheet_name, level_name in (
            ("topic_rank_l1", "一级维度"),
            ("topic_rank_l2", "二级维度"),
            ("topic_rank_l3", "三级维度"),
        ):
            level_frame = topic_ranking_frame.filter(pl.col("维度层级") == level_name)
            level_frame.write_excel(
                workbook=workbook,
                worksheet=sheet_name,
                table_name=f"TopicRank{sheet_name[-2:].upper()}",
                autofit=True,
                hide_gridlines=True,
            )

        if not alignment_frame.is_empty():
            alignment_frame.write_excel(
                workbook=workbook,
                worksheet="aligned_pairs",
                table_name="AlignedPairs",
                autofit=True,
                hide_gridlines=True,
            )


def print_summary(
    merged_frame: pl.DataFrame,
    difference_frame: pl.DataFrame,
    topic_ranking_frame: pl.DataFrame,
    alignment_frame: pl.DataFrame,
) -> None:
    """Print a compact terminal summary after exports are written."""
    model_summary_frame = (
        merged_frame.group_by(["model_prefix", "score_source"], maintain_order=False)
        .agg(
            pl.len().alias("framework_rows"),
            pl.col("zeroshot_已评分").sum().alias("zeroshot_rows"),
            pl.col("cot_已评分").sum().alias("cot_rows"),
        )
        .sort(by=["model_prefix", "score_source"])
    )

    print("Analysis finished.")
    print("Model coverage:")
    for summary_row in model_summary_frame.to_dicts():
        print(
            "  - model_prefix={model_prefix} source={score_source} framework_rows={framework_rows} "
            "zeroshot_rows={zeroshot_rows} cot_rows={cot_rows}".format(**summary_row)
        )
    print(f"Same-sentence difference rows: {difference_frame.height}")
    print(f"Topic ranking rows: {topic_ranking_frame.height}")
    print(f"Alignment rows: {alignment_frame.height}")


def main() -> None:
    """Run the full 1b group-swap analysis pipeline."""
    arguments = parse_arguments()
    score_directory = resolve_score_directory(arguments.framework_path, arguments.score_dir)
    output_directory = arguments.output_dir or (score_directory / "analysis_1b")

    framework_frame = load_framework_frame(arguments.framework_path)
    score_sources = discover_score_sources(score_directory)
    score_frame = select_best_score_frame(score_sources)
    merged_frame = merge_framework_and_scores(framework_frame, score_frame)
    difference_frame = build_same_sentence_difference_frame(merged_frame)
    long_gender_score_frame = build_gender_score_long_frame(merged_frame)
    topic_ranking_frame = build_topic_ranking_frame(long_gender_score_frame)
    alignment_frame = build_alignment_frame(merged_frame, arguments.top_k_alignments)

    write_csv_outputs(
        output_directory=output_directory,
        merged_frame=merged_frame,
        difference_frame=difference_frame,
        topic_ranking_frame=topic_ranking_frame,
        alignment_frame=alignment_frame,
    )
    workbook_path = output_directory / ANALYSIS_WORKBOOK_NAME
    write_excel_workbook(
        workbook_path=workbook_path,
        merged_frame=merged_frame,
        difference_frame=difference_frame,
        topic_ranking_frame=topic_ranking_frame,
        alignment_frame=alignment_frame,
    )
    print_summary(
        merged_frame=merged_frame,
        difference_frame=difference_frame,
        topic_ranking_frame=topic_ranking_frame,
        alignment_frame=alignment_frame,
    )
    logger.info("Workbook saved to %s", workbook_path)


if __name__ == "__main__":
    main()
