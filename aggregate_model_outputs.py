from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "natural"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_NAME_MAP = {
    "anthropic/claude-opus-4.5": "claude4.5",
    "openai/chatgpt-5.1": "chatgpt5.1",
    "baidu/ernie-4.5-21b-a3b": "baidu_ernie-4.5-21b-a3b",
    "deepseek/deepseek-r1-0528": "deepseek_deepseek-r1-0528",
    "deepseek/deepseek-v3.2-exp": "deepseek_deepseek-v3.2-exp",
    "meta-llama/llama-4-maverick": "meta-llama_llama-4-maverick",
    "moonshotai/kimi-k2-thinking": "moonshotai_kimi-k2-thinking",
    "qwen/qwen-2.5-72b-instruct": "qwen_qwen-2.5-72b-instruct",
    "z-ai/glm-4.6": "z-ai_glm-4.6",
}

PROMPT_NAME_MAP = {
    "zero_shot": "zeroshot",
    "chain_of_thought": "cot",
}


@dataclass(frozen=True)
class DatasetConfig:
    language: str
    data_path: Path
    output_root: Path
    merged_output_path: Path


DATASET_CONFIGS = [
    DatasetConfig(
        language="zh",
        data_path=DATA_DIR / "5_new_chinesehatedata_2400_balanced.xlsx",
        output_root=OUTPUTS_DIR / "5_new_chinesehatedata_2400_balanced" / "zh",
        merged_output_path=ARTIFACTS_DIR / "merged_zh_models_validated.xlsx",
    ),
    DatasetConfig(
        language="en",
        data_path=DATA_DIR / "5_new_englishhatedata_2400_balanced.xlsx",
        output_root=OUTPUTS_DIR / "5_new_englishhatedata_2400_balanced" / "en",
        merged_output_path=ARTIFACTS_DIR / "merged_en_models_validated.xlsx",
    ),
]


def normalize_model_name(raw_model_name: str) -> str:
    """Map raw provider/model names into the short identifiers used elsewhere in the project."""
    raw_model_name = str(raw_model_name).strip()
    return MODEL_NAME_MAP.get(raw_model_name, raw_model_name.replace("/", "_"))


def load_source_dataframe(dataset_config: DatasetConfig) -> pd.DataFrame:
    """Load the source dataset and attach a reliable sample_index column."""
    source_dataframe = pd.read_excel(dataset_config.data_path).copy()
    source_dataframe["sample_index"] = range(len(source_dataframe))
    source_dataframe["language"] = dataset_config.language
    return source_dataframe


def collect_output_files(output_root: Path) -> list[tuple[str, Path]]:
    """Return all output files grouped by prompt paradigm."""
    collected_files: list[tuple[str, Path]] = []
    for prompt_directory in sorted([path for path in output_root.iterdir() if path.is_dir()]):
        prompt_name = PROMPT_NAME_MAP.get(prompt_directory.name, prompt_directory.name)
        for file_path in sorted(prompt_directory.glob("*.xlsx")):
            collected_files.append((prompt_name, file_path))
    return collected_files


def load_raw_outputs(dataset_config: DatasetConfig) -> pd.DataFrame:
    """Load every raw model-output file into one normalized long table."""
    output_rows: list[pd.DataFrame] = []
    for prompt_name, file_path in collect_output_files(dataset_config.output_root):
        dataframe = pd.read_excel(file_path).copy()
        dataframe["source_file"] = file_path.name
        dataframe["prompt_name"] = prompt_name
        output_rows.append(dataframe)

    if not output_rows:
        raise ValueError(f"No raw output files found under {dataset_config.output_root}")

    outputs_dataframe = pd.concat(output_rows, ignore_index=True)
    outputs_dataframe["model_key"] = outputs_dataframe["model"].map(normalize_model_name)
    outputs_dataframe["prompt_name"] = outputs_dataframe["prompt_name"].astype(str)
    outputs_dataframe["sample_index"] = outputs_dataframe["sample_index"].astype("Int64")
    return outputs_dataframe


def normalize_text_for_match(text_value: object) -> str:
    """Normalize whitespace so harmless Excel trailing-space noise does not trigger false mismatches."""
    if pd.isna(text_value):
        return ""
    return " ".join(str(text_value).split())


def build_source_text_index(source_dataframe: pd.DataFrame) -> tuple[dict[str, list[int]], list[str]]:
    """Map normalized source texts to every raw row where they occur."""
    normalized_source_texts = [
        normalize_text_for_match(text_value) for text_value in source_dataframe["text"].tolist()
    ]
    text_to_indices: dict[str, list[int]] = defaultdict(list)
    for source_order_index, normalized_text in enumerate(normalized_source_texts):
        text_to_indices[normalized_text].append(source_order_index)
    return dict(text_to_indices), normalized_source_texts


def parse_source_file_order(source_file_name: object) -> int:
    """Extract the numeric sample order encoded in a workbook filename."""
    source_file_stem = Path(str(source_file_name)).stem
    if "_sample_" not in source_file_stem:
        return 10**9

    sample_marker = source_file_stem.rsplit("_sample_", 1)[1]
    return int(sample_marker) if sample_marker.isdigit() else 10**9


def resolve_row_to_source_index(
    normalized_row_text: str,
    reported_sample_index: int | None,
    text_to_indices: dict[str, list[int]],
    normalized_source_texts: list[str],
    assigned_source_indices: set[int],
    previous_aligned_index: int | None,
) -> tuple[int | None, str]:
    """Resolve one model-output row back to a unique source row."""
    if not normalized_row_text:
        return None, "unresolved"

    if (
        reported_sample_index is not None
        and 0 <= reported_sample_index < len(normalized_source_texts)
        and normalized_source_texts[reported_sample_index] == normalized_row_text
        and reported_sample_index not in assigned_source_indices
    ):
        return reported_sample_index, "sample_index"

    candidate_indices = [
        candidate_index
        for candidate_index in text_to_indices.get(normalized_row_text, [])
        if candidate_index not in assigned_source_indices
    ]
    if not candidate_indices:
        return None, "unresolved"

    if len(candidate_indices) == 1:
        return candidate_indices[0], "text"

    if reported_sample_index in candidate_indices:
        return reported_sample_index, "text_exact_match"

    if previous_aligned_index is not None:
        monotonic_candidates = [
            candidate_index
            for candidate_index in candidate_indices
            if candidate_index >= previous_aligned_index
        ]
        if monotonic_candidates:
            candidate_indices = monotonic_candidates

    if reported_sample_index is None:
        return min(candidate_indices), "text_ambiguous"

    resolved_index = min(
        candidate_indices,
        key=lambda candidate_index: abs(candidate_index - reported_sample_index),
    )
    return resolved_index, "text_ambiguous"


def realign_outputs_to_source(
    source_dataframe: pd.DataFrame,
    outputs_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Repair systematic sample-index drift by reassigning rows using source text order."""
    text_to_indices, normalized_source_texts = build_source_text_index(source_dataframe)
    realigned_outputs = outputs_dataframe.copy()
    realigned_outputs["reported_sample_index"] = realigned_outputs["sample_index"]
    realigned_outputs["source_file_order"] = realigned_outputs["source_file"].map(parse_source_file_order)
    realigned_outputs["normalized_text"] = realigned_outputs["text"].map(normalize_text_for_match)
    realigned_outputs["alignment_method"] = "unresolved"
    realigned_outputs["sample_index"] = pd.Series(pd.NA, index=realigned_outputs.index, dtype="Int64")

    for (_, _), group_indices in realigned_outputs.groupby(
        ["model_key", "prompt_name"],
        sort=False,
    ).groups.items():
        assigned_source_indices: set[int] = set()
        previous_aligned_index: int | None = None
        group_dataframe = realigned_outputs.loc[group_indices].sort_values(
            ["source_file_order", "reported_sample_index", "source_file"],
            kind="mergesort",
        )

        for row_index, row in group_dataframe.iterrows():
            reported_sample_index = (
                None if pd.isna(row["reported_sample_index"]) else int(row["reported_sample_index"])
            )
            resolved_index, alignment_method = resolve_row_to_source_index(
                normalized_row_text=row["normalized_text"],
                reported_sample_index=reported_sample_index,
                text_to_indices=text_to_indices,
                normalized_source_texts=normalized_source_texts,
                assigned_source_indices=assigned_source_indices,
                previous_aligned_index=previous_aligned_index,
            )
            realigned_outputs.at[row_index, "alignment_method"] = alignment_method
            if resolved_index is None:
                continue

            realigned_outputs.at[row_index, "sample_index"] = resolved_index
            assigned_source_indices.add(resolved_index)
            previous_aligned_index = resolved_index

    return realigned_outputs


def validate_outputs_against_source(
    source_dataframe: pd.DataFrame,
    outputs_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Check duplicate keys, text alignment, and expected coverage before pivoting."""
    outputs_dataframe = realign_outputs_to_source(
        source_dataframe=source_dataframe,
        outputs_dataframe=outputs_dataframe,
    )

    dedupe_counts = (
        outputs_dataframe[outputs_dataframe["sample_index"].notna()]
        .groupby(["model_key", "prompt_name", "sample_index"], dropna=False)
        .size()
        .reset_index(name="row_count")
    )
    duplicate_key_rows = dedupe_counts[dedupe_counts["row_count"] > 1].copy()

    merged_validation = outputs_dataframe.merge(
        source_dataframe[["sample_index", "text"]].rename(columns={"text": "source_text"}),
        on="sample_index",
        how="left",
    )
    merged_validation["normalized_source_text"] = merged_validation["source_text"].map(normalize_text_for_match)
    merged_validation["text_matches_source"] = merged_validation["normalized_text"] == merged_validation["normalized_source_text"]
    text_mismatch_rows = merged_validation[
        merged_validation["source_text"].notna() & (~merged_validation["text_matches_source"])
    ][
        [
            "model_key",
            "prompt_name",
            "sample_index",
            "reported_sample_index",
            "alignment_method",
            "source_file",
            "text",
            "source_text",
        ]
    ].copy()

    validated_outputs = merged_validation[
        merged_validation["source_text"].notna() & merged_validation["text_matches_source"]
    ].copy().sort_values(["sample_index", "model_key", "prompt_name", "source_file"], kind="mergesort").reset_index(drop=True)

    coverage_counts = (
        validated_outputs.drop_duplicates(["model_key", "prompt_name", "sample_index"])
        .groupby(["model_key", "prompt_name"], dropna=False)
        .size()
        .reset_index(name="observed_samples")
    )
    expected_sample_count = int(len(source_dataframe))
    coverage_counts["expected_samples"] = expected_sample_count
    coverage_counts["missing_samples"] = expected_sample_count - coverage_counts["observed_samples"]

    missing_reason_counts = (
        validated_outputs.assign(reason_missing=validated_outputs["reason"].isna())
        .groupby(["model_key", "prompt_name"], dropna=False)["reason_missing"]
        .sum()
        .reset_index(name="missing_reason_rows")
    )

    coverage_summary = coverage_counts.merge(
        missing_reason_counts,
        on=["model_key", "prompt_name"],
        how="left",
    ).sort_values(["prompt_name", "model_key"]).reset_index(drop=True)
    coverage_summary["missing_reason_rows"] = coverage_summary["missing_reason_rows"].fillna(0).astype(int)

    return duplicate_key_rows, text_mismatch_rows, coverage_summary, merged_validation, validated_outputs


def build_wide_merged_dataframe(
    source_dataframe: pd.DataFrame,
    outputs_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create one wide table with score/reason columns per model and prompt paradigm."""
    unique_rows = outputs_dataframe.drop_duplicates(
        subset=["model_key", "prompt_name", "sample_index"],
        keep="first",
    ).copy()

    score_pivot = unique_rows.pivot(
        index="sample_index",
        columns=["model_key", "prompt_name"],
        values="score",
    )
    reason_pivot = unique_rows.pivot(
        index="sample_index",
        columns=["model_key", "prompt_name"],
        values="reason",
    )

    score_pivot.columns = [f"{model}_{prompt}_score" for model, prompt in score_pivot.columns]
    reason_pivot.columns = [f"{model}_{prompt}_reason" for model, prompt in reason_pivot.columns]

    merged_dataframe = source_dataframe.copy().set_index("sample_index")
    merged_dataframe = merged_dataframe.join(score_pivot, how="left")
    merged_dataframe = merged_dataframe.join(reason_pivot, how="left")
    merged_dataframe = merged_dataframe.reset_index()

    preferred_front_columns = [column for column in ["sample_index", "text", "language"] if column in merged_dataframe.columns]
    remaining_columns = [column for column in merged_dataframe.columns if column not in preferred_front_columns]
    merged_dataframe = merged_dataframe[preferred_front_columns + remaining_columns]
    return merged_dataframe


def build_missing_sample_table(
    source_dataframe: pd.DataFrame,
    outputs_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """List exactly which sample_index values are missing for each model/prompt pair."""
    missing_rows: list[dict[str, object]] = []
    expected_sample_indices = set(source_dataframe["sample_index"].tolist())

    for (model_key, prompt_name), group_dataframe in outputs_dataframe.groupby(["model_key", "prompt_name"], dropna=False):
        observed_sample_indices = set(group_dataframe["sample_index"].dropna().astype(int).tolist())
        missing_sample_indices = sorted(expected_sample_indices.difference(observed_sample_indices))
        for sample_index in missing_sample_indices:
            source_row = source_dataframe.iloc[sample_index]
            missing_rows.append(
                {
                    "model_key": model_key,
                    "prompt_name": prompt_name,
                    "sample_index": sample_index,
                    "text": source_row["text"],
                }
            )

    return pd.DataFrame(missing_rows)


def aggregate_one_dataset(dataset_config: DatasetConfig) -> dict[str, object]:
    """Aggregate and validate one dataset, then write a new Excel workbook."""
    source_dataframe = load_source_dataframe(dataset_config)
    outputs_dataframe = load_raw_outputs(dataset_config)
    duplicate_key_rows, text_mismatch_rows, coverage_summary, merged_validation, validated_outputs = validate_outputs_against_source(
        source_dataframe=source_dataframe,
        outputs_dataframe=outputs_dataframe,
    )
    merged_dataframe = build_wide_merged_dataframe(
        source_dataframe=source_dataframe,
        outputs_dataframe=validated_outputs,
    )
    missing_sample_table = build_missing_sample_table(
        source_dataframe=source_dataframe,
        outputs_dataframe=validated_outputs,
    )

    dataset_config.merged_output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(dataset_config.merged_output_path, engine="openpyxl") as writer:
        merged_dataframe.to_excel(writer, index=False, sheet_name="merged")
        coverage_summary.to_excel(writer, index=False, sheet_name="coverage_summary")
        duplicate_key_rows.to_excel(writer, index=False, sheet_name="duplicate_keys")
        text_mismatch_rows.to_excel(writer, index=False, sheet_name="text_mismatches")
        missing_sample_table.to_excel(writer, index=False, sheet_name="missing_samples")
        validated_outputs[
            [
                "sample_index",
                "reported_sample_index",
                "alignment_method",
                "text",
                "model_key",
                "prompt_name",
                "score",
                "reason",
                "source_file",
            ]
        ].to_excel(
            writer,
            index=False,
            sheet_name="validated_long",
        )

    return {
        "language": dataset_config.language,
        "output_path": str(dataset_config.merged_output_path),
        "source_rows": int(len(source_dataframe)),
        "raw_output_rows": int(len(outputs_dataframe)),
        "duplicate_key_rows": int(len(duplicate_key_rows)),
        "text_mismatch_rows": int(len(text_mismatch_rows)),
        "missing_sample_rows": int(len(missing_sample_table)),
        "coverage_summary": coverage_summary,
    }


def main() -> None:
    run_summaries: list[dict[str, object]] = []
    for dataset_config in DATASET_CONFIGS:
        summary = aggregate_one_dataset(dataset_config)
        run_summaries.append(summary)

    for summary in run_summaries:
        print(f"\n=== {summary['language']} aggregation complete ===")
        print(f"Output workbook: {summary['output_path']}")
        print(f"Source rows: {summary['source_rows']}")
        print(f"Raw output rows: {summary['raw_output_rows']}")
        print(f"Duplicate key rows: {summary['duplicate_key_rows']}")
        print(f"Text mismatch rows: {summary['text_mismatch_rows']}")
        print(f"Missing sample rows: {summary['missing_sample_rows']}")
        print("Coverage summary:")
        print(summary["coverage_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
