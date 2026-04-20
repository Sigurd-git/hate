"""Run research 1b group-swap attackiveness experiments for configured target models.

This script reads the 1b Excel dataset, extracts the male/female swapped sentence
columns, scores each sentence under three human-style attackiveness conditions,
and writes one consolidated results file per model.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path("data/1b_groupswap_demensionsentence.xlsx")
OUTPUT_ROOT = Path("outputs/group_swap_1b")
LANGUAGE = "zh"
REQUEST_TIMEOUT_SECONDS = 60
MAX_WORKERS = 4
SAVE_EVERY_COMPLETED_ROWS = 10
DEFAULT_MODELS = (
    "openai/gpt-5.1",
    "anthropic/claude-opus-4.5",
    "z-ai/glm-4.6",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-v3.2-exp",
    "moonshotai/kimi-k2-thinking",
    "qwen/qwen-2.5-72b-instruct",
    "google/gemma-4-31b-it",
)

MALE_COLUMN = "具体内容-男人版"
FEMALE_COLUMN = "具体内容-女人版"


@dataclass(frozen=True)
class ConditionSpec:
    """Describe one 1b scoring condition and its score bounds."""

    prompt_mode: str
    score_max: int
    output_suffix: str


CONDITION_SPECS = (
    ConditionSpec(prompt_mode="attack_3pt", score_max=2, output_suffix="attack_3pt"),
    ConditionSpec(
        prompt_mode="attack_7pt_likert",
        score_max=6,
        output_suffix="attack_7pt_likert",
    ),
    ConditionSpec(
        prompt_mode="attack_slider_0_100",
        score_max=100,
        output_suffix="attack_slider_0_100",
    ),
)

REQUIRED_COLUMNS = (
    "一级-攻击领域",
    "二级-攻击属性",
    "三级-攻击表达策略",
    "具体内容",
    MALE_COLUMN,
    FEMALE_COLUMN,
)

OUTPUT_COLUMNS = [
    "row_index",
    "一级-攻击领域",
    "二级-攻击属性",
    "三级-攻击表达策略",
    "具体内容",
    MALE_COLUMN,
    FEMALE_COLUMN,
    "model",
    "language",
    "dataset_label",
    "男人版-attack_3pt攻击性评分",
    "男人版-attack_7pt_likert攻击性评分",
    "男人版-attack_slider_0_100攻击性评分",
    "女人版-attack_3pt攻击性评分",
    "女人版-attack_7pt_likert攻击性评分",
    "女人版-attack_slider_0_100攻击性评分",
    "request_status",
    "error_message",
]
DEFAULT_OUTPUT_SUFFIX = ".csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for targeted experiment runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="One or more OpenRouter model identifiers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute rows even if an output file already exists.",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "xlsx"),
        default="csv",
        help="Primary output format. Default keeps only CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate loading/output flow without making model requests.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    """Convert spreadsheet cells into clean strings suitable for prompting."""
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text


def load_dataset_rows(dataset_path: Path) -> pd.DataFrame:
    """Load and validate the 1b dataset while preserving original row order."""
    dataset_frame = pd.read_excel(dataset_path)
    missing_columns = [column_name for column_name in REQUIRED_COLUMNS if column_name not in dataset_frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    dataset_frame = dataset_frame.copy()
    dataset_frame.insert(0, "row_index", range(len(dataset_frame)))

    for column_name in REQUIRED_COLUMNS:
        dataset_frame[column_name] = dataset_frame[column_name].apply(normalize_text)

    dataset_frame = dataset_frame.loc[
        (dataset_frame[MALE_COLUMN] != "") | (dataset_frame[FEMALE_COLUMN] != "")
    ].copy()
    logger.info("Loaded %s usable rows from dataset=%s", len(dataset_frame), dataset_path.name)
    return dataset_frame


def apply_row_window(dataset_frame: pd.DataFrame, start_row: int, end_row: int | None, limit: int | None) -> pd.DataFrame:
    """Return the requested contiguous row subset for the experiment run."""
    filtered_frame = dataset_frame.loc[dataset_frame["row_index"] >= start_row]
    if end_row is not None:
        filtered_frame = filtered_frame.loc[filtered_frame["row_index"] < end_row]
    if limit is not None:
        filtered_frame = filtered_frame.head(limit)
    logger.info(
        "Selected rows for execution count=%s start_row=%s end_row=%s limit=%s",
        len(filtered_frame),
        start_row,
        end_row,
        limit,
    )
    return filtered_frame.reset_index(drop=True)


def sanitize_model_name(model_name: str) -> str:
    """Convert model names into file-system-safe stems."""
    return model_name.replace("/", "_").replace(":", "_")


def build_output_path(output_dir: Path, dataset_label: str, model_name: str, output_format: str) -> Path:
    """Build the consolidated output path for one model run."""
    return output_dir / dataset_label / f"{sanitize_model_name(model_name)}_results.{output_format}"


def load_existing_results(output_path: Path) -> pd.DataFrame:
    """Load a prior results table if it exists, otherwise return an empty frame."""
    if not output_path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if output_path.suffix == ".csv":
        existing_frame = pd.read_csv(output_path)
    else:
        existing_frame = pd.read_excel(output_path)
    for column_name in OUTPUT_COLUMNS:
        if column_name not in existing_frame.columns:
            existing_frame[column_name] = ""
    return existing_frame[OUTPUT_COLUMNS].copy()


def is_row_complete(existing_row: pd.Series) -> bool:
    """Decide whether a saved row has all requested condition-specific scores."""
    score_columns = [
        "男人版-attack_3pt攻击性评分",
        "男人版-attack_7pt_likert攻击性评分",
        "男人版-attack_slider_0_100攻击性评分",
        "女人版-attack_3pt攻击性评分",
        "女人版-attack_7pt_likert攻击性评分",
        "女人版-attack_slider_0_100攻击性评分",
    ]
    for column_name in score_columns:
        value = existing_row.get(column_name, "")
        if pd.isna(value) or value == "":
            return False
    return True


def score_sentence(
    text: str,
    model_name: str,
    condition_spec: ConditionSpec,
    dry_run: bool,
) -> Dict[str, object]:
    """Score one sentence with the prompt dedicated to a single rating condition."""
    if not text:
        return {"score": ""}
    if dry_run:
        return {"score": -1}

    response = pipeline.request_score(
        pipeline.Sample(text=text, language=LANGUAGE),
        model=model_name,
        prompt_paradigm="zero_shot",
        timeout=REQUEST_TIMEOUT_SECONDS,
        prompt_mode=condition_spec.prompt_mode,
        score_max=condition_spec.score_max,
    )
    return {
        "score": response.payload.get("score", ""),
    }


def score_dataset_row(row: pd.Series, model_name: str, dataset_label: str, dry_run: bool) -> Dict[str, object]:
    """Score all required male/female prompt variants for one dataset row."""
    result_row: Dict[str, object] = {
        "row_index": int(row["row_index"]),
        "一级-攻击领域": row["一级-攻击领域"],
        "二级-攻击属性": row["二级-攻击属性"],
        "三级-攻击表达策略": row["三级-攻击表达策略"],
        "具体内容": row["具体内容"],
        MALE_COLUMN: row[MALE_COLUMN],
        FEMALE_COLUMN: row[FEMALE_COLUMN],
        "model": model_name,
        "language": LANGUAGE,
        "dataset_label": dataset_label,
        "男人版-attack_3pt攻击性评分": "",
        "男人版-attack_7pt_likert攻击性评分": "",
        "男人版-attack_slider_0_100攻击性评分": "",
        "女人版-attack_3pt攻击性评分": "",
        "女人版-attack_7pt_likert攻击性评分": "",
        "女人版-attack_slider_0_100攻击性评分": "",
        "request_status": "success",
        "error_message": "",
    }

    request_plan = {}
    for condition_spec in CONDITION_SPECS:
        request_plan[f"男人版-{condition_spec.output_suffix}攻击性评分"] = (
            row[MALE_COLUMN],
            condition_spec,
        )
        request_plan[f"女人版-{condition_spec.output_suffix}攻击性评分"] = (
            row[FEMALE_COLUMN],
            condition_spec,
        )

    response_cache: Dict[tuple[str, str], Dict[str, object]] = {}
    try:
        with ThreadPoolExecutor(max_workers=4) as row_executor:
            future_map = {}
            for text_value, condition_spec in request_plan.values():
                cache_key = (text_value, condition_spec.prompt_mode)
                if cache_key in response_cache or not text_value:
                    continue
                future = row_executor.submit(
                    score_sentence,
                    text_value,
                    model_name,
                    condition_spec,
                    dry_run,
                )
                future_map[future] = cache_key

            for future, cache_key in future_map.items():
                response_cache[cache_key] = future.result()
    except Exception as error:
        result_row["request_status"] = "error"
        result_row["error_message"] = f"{type(error).__name__}: {error}"
        return result_row

    for output_column, (text_value, condition_spec) in request_plan.items():
        if not text_value:
            result_row[output_column] = ""
            continue
        cache_key = (text_value, condition_spec.prompt_mode)
        result_row[output_column] = response_cache[cache_key].get("score", "")

    return result_row


def merge_and_save_results(output_path: Path, rows: List[Dict[str, object]]) -> None:
    """Write the consolidated per-model result table to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(rows)
    if not result_frame.empty:
        for column_name in OUTPUT_COLUMNS:
            if column_name not in result_frame.columns:
                result_frame[column_name] = ""
        result_frame = result_frame[OUTPUT_COLUMNS].sort_values("row_index").reset_index(drop=True)
    else:
        result_frame = pd.DataFrame(columns=OUTPUT_COLUMNS)

    if output_path.suffix == ".csv":
        result_frame.to_csv(output_path, index=False)
    else:
        result_frame.to_excel(output_path, index=False)
    logger.info("Saved %s rows to %s", len(result_frame), output_path)


def run_model_experiment(
    dataset_frame: pd.DataFrame,
    dataset_label: str,
    model_name: str,
    output_dir: Path,
    overwrite: bool,
    output_format: str,
    dry_run: bool,
) -> Path:
    """Execute or resume one model-specific experiment table using a thread pool."""
    output_path = build_output_path(output_dir, dataset_label, model_name, output_format)
    existing_frame = load_existing_results(output_path)
    retained_rows_by_index: Dict[int, Dict[str, object]] = {}
    if not overwrite and not existing_frame.empty:
        retained_rows_by_index = {
            int(row["row_index"]): row.to_dict() for _, row in existing_frame.iterrows()
        }

    pending_rows: List[pd.Series] = []
    skipped_count = 0
    for _, row in dataset_frame.iterrows():
        row_index = int(row["row_index"])
        existing_row = retained_rows_by_index.get(row_index)
        if not overwrite and existing_row is not None and is_row_complete(pd.Series(existing_row)):
            skipped_count += 1
            continue
        pending_rows.append(row)

    logger.info(
        "Starting model=%s pending_rows=%s skipped_existing=%s max_workers=%s",
        model_name,
        len(pending_rows),
        skipped_count,
        MAX_WORKERS,
    )

    completed_count = 0
    error_count = 0
    if pending_rows:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_row_index = {
                executor.submit(score_dataset_row, row, model_name, dataset_label, dry_run): int(row["row_index"])
                for _, row in pd.DataFrame(pending_rows).iterrows()
            }
            pending_futures = set(future_to_row_index.keys())

            while pending_futures:
                done_futures, pending_futures = wait(pending_futures, return_when=FIRST_COMPLETED)
                for future in done_futures:
                    row_index = future_to_row_index[future]
                    scored_row = future.result()
                    retained_rows_by_index[row_index] = scored_row
                    if scored_row["request_status"] == "success":
                        completed_count += 1
                    else:
                        error_count += 1
                        logger.error(
                            "Row failed model=%s row_index=%s error=%s",
                            model_name,
                            row_index,
                            scored_row["error_message"],
                        )

                    processed_count = completed_count + error_count
                    if (
                        processed_count % SAVE_EVERY_COMPLETED_ROWS == 0
                        or not pending_futures
                    ):
                        merge_and_save_results(
                            output_path,
                            list(retained_rows_by_index.values()),
                        )
                        logger.info(
                            "Progress model=%s processed=%s/%s success=%s error=%s skipped=%s",
                            model_name,
                            processed_count,
                            len(pending_rows),
                            completed_count,
                            error_count,
                            skipped_count,
                        )
    else:
        merge_and_save_results(output_path, list(retained_rows_by_index.values()))

    merge_and_save_results(output_path, list(retained_rows_by_index.values()))
    logger.info(
        "Model summary model=%s completed=%s skipped=%s errors=%s output=%s",
        model_name,
        completed_count,
        skipped_count,
        error_count,
        output_path,
    )
    return output_path


def main() -> None:
    """Run the requested 1b experiment configuration."""
    args = parse_args()
    dataset_frame = load_dataset_rows(args.dataset)
    dataset_frame = apply_row_window(dataset_frame, args.start_row, args.end_row, args.limit)
    dataset_label = args.dataset.stem

    if dataset_frame.empty:
        raise ValueError("No rows selected for execution after applying filters.")

    output_paths: List[Path] = []
    for model_name in args.models:
        output_path = run_model_experiment(
            dataset_frame=dataset_frame,
            dataset_label=dataset_label,
            model_name=model_name,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            output_format=args.output_format,
            dry_run=args.dry_run,
        )
        output_paths.append(output_path)

    logger.info("Completed research 1b runs. Outputs: %s", ", ".join(str(path) for path in output_paths))


if __name__ == "__main__":
    main()
