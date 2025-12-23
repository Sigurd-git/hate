from __future__ import annotations

import logging
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_ROOT = Path("outputs/natural")
LANGUAGE_OUTPUT_ROOTS = {
    "zh": OUTPUT_ROOT / "5_new_chinesehatedata_2400_balanced/zh",
    "en": OUTPUT_ROOT / "5_new_englishhatedata_2400_balanced/en",
}

PARADIGM_FOLDERS = {
    "zeroshot": "zero_shot",
    "cot": "chain_of_thought",
}

MODEL_LABEL_OVERRIDES = {
    "openai_gpt-5.1": "chatgpt5.1",
    "anthropic_claude-opus-4.5": "claude4.5",
}

REQUIRED_COLUMNS = ("text", "language", "score", "reason", "sample_index")
EXPECTED_ROW_COUNT = 2400


@dataclass(frozen=True)
class ModelInputs:
    model_name: str
    model_label: str
    zero_shot_files: Sequence[Path]
    cot_files: Sequence[Path]


def list_excel_files(folder: Path) -> List[Path]:
    """Return sorted Excel files in the folder."""
    pattern = str(folder / "*.xlsx")
    matches = [Path(match) for match in glob(pattern)]
    return sorted(matches)


def parse_model_and_sample_index(file_path: Path) -> Tuple[str, int]:
    """Parse the model name and sample index from a sample Excel filename."""
    file_stem = file_path.stem
    if "_sample_" not in file_stem:
        raise ValueError(f"Unexpected filename format: {file_path.name}")
    model_name, sample_marker = file_stem.rsplit("_sample_", 1)
    if not sample_marker.isdigit():
        raise ValueError(f"Unexpected sample marker in filename: {file_path.name}")
    return model_name, int(sample_marker)


def group_files_by_model(file_paths: Iterable[Path]) -> Dict[str, List[Path]]:
    """Group Excel files by model name with stable sample ordering."""
    grouped: Dict[str, List[Tuple[int, Path]]] = {}
    for file_path in file_paths:
        model_name, sample_index = parse_model_and_sample_index(file_path)
        grouped.setdefault(model_name, []).append((sample_index, file_path))
    return {
        model_name: [file_path for _, file_path in sorted(entries, key=lambda item: item[0])]
        for model_name, entries in grouped.items()
    }


def normalize_model_label(model_name: str) -> str:
    """Return a display label for the model columns."""
    return MODEL_LABEL_OVERRIDES.get(model_name, model_name)


def ensure_required_columns(frame: pd.DataFrame, source_label: str) -> None:
    """Validate the input frame has the expected columns."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{source_label} missing columns: {missing_columns}")


def read_and_concat_excel(file_paths: Sequence[Path], source_label: str) -> pd.DataFrame:
    """Read Excel files and concatenate them into a single DataFrame."""
    if not file_paths:
        raise FileNotFoundError(f"No Excel files found for {source_label}")
    frames: List[pd.DataFrame] = []
    for file_path in file_paths:
        frame = pd.read_excel(file_path)
        frames.append(frame)
    combined_frame = pd.concat(frames, ignore_index=True)
    ensure_required_columns(combined_frame, source_label)
    combined_frame["sample_index"] = combined_frame["sample_index"].astype(int)
    combined_frame = combined_frame.sort_values("sample_index").reset_index(drop=True)
    return combined_frame


def build_paradigm_frame(
    file_paths: Sequence[Path],
    model_label: str,
    paradigm_label: str,
    include_text_columns: bool,
) -> pd.DataFrame:
    """Load a paradigm frame and rename score/reason columns for the target model."""
    combined_frame = read_and_concat_excel(
        file_paths, source_label=f"{model_label}-{paradigm_label}"
    )
    selected_columns = ["sample_index", "score", "reason"]
    if include_text_columns:
        selected_columns = ["text", "language", "sample_index", "score", "reason"]
    trimmed_frame = combined_frame.loc[:, selected_columns].copy()
    renamed_columns = {
        "score": f"{model_label}_{paradigm_label}_score",
        "reason": f"{model_label}_{paradigm_label}_reason",
    }
    return trimmed_frame.rename(columns=renamed_columns)


def build_model_frame(model_inputs: ModelInputs) -> pd.DataFrame:
    """Merge zero-shot and chain-of-thought outputs for a single model."""
    zero_shot_frame = build_paradigm_frame(
        model_inputs.zero_shot_files,
        model_inputs.model_label,
        "zeroshot",
        include_text_columns=True,
    )
    cot_frame = build_paradigm_frame(
        model_inputs.cot_files,
        model_inputs.model_label,
        "cot",
        include_text_columns=False,
    )
    merged_frame = zero_shot_frame.merge(cot_frame, on="sample_index", how="left")
    return merged_frame


def merge_language_outputs(language_root: Path, output_path: Path) -> None:
    """Merge all models for a language into one Excel file."""
    zero_shot_root = language_root / PARADIGM_FOLDERS["zeroshot"]
    cot_root = language_root / PARADIGM_FOLDERS["cot"]

    if not zero_shot_root.is_dir():
        raise NotADirectoryError(f"{zero_shot_root} is not a valid directory")
    if not cot_root.is_dir():
        raise NotADirectoryError(f"{cot_root} is not a valid directory")

    zero_shot_files = list_excel_files(zero_shot_root)
    cot_files = list_excel_files(cot_root)

    zero_shot_by_model = group_files_by_model(zero_shot_files)
    cot_by_model = group_files_by_model(cot_files)

    model_names = sorted(set(zero_shot_by_model) | set(cot_by_model))
    if not model_names:
        raise FileNotFoundError(f"No models found under {language_root}")

    model_inputs_list: List[ModelInputs] = []
    for model_name in model_names:
        zero_shot_group = zero_shot_by_model.get(model_name, [])
        cot_group = cot_by_model.get(model_name, [])
        if not zero_shot_group or not cot_group:
            logging.warning(
                "Model %s missing files (zero_shot=%s, cot=%s)",
                model_name,
                len(zero_shot_group),
                len(cot_group),
            )
        if not zero_shot_group:
            continue
        if not cot_group:
            continue
        model_label = normalize_model_label(model_name)
        model_inputs_list.append(
            ModelInputs(
                model_name=model_name,
                model_label=model_label,
                zero_shot_files=zero_shot_group,
                cot_files=cot_group,
            )
        )

    if not model_inputs_list:
        raise FileNotFoundError(f"No complete model outputs for {language_root}")

    logging.info("Merging %s models for %s", len(model_inputs_list), language_root)
    model_inputs_list = sorted(model_inputs_list, key=lambda item: item.model_label)

    model_frames: List[pd.DataFrame] = []
    for model_inputs in model_inputs_list:
        logging.info(
            "Loading model=%s zero_shot_files=%s cot_files=%s",
            model_inputs.model_label,
            len(model_inputs.zero_shot_files),
            len(model_inputs.cot_files),
        )
        model_frame = build_model_frame(model_inputs)
        model_frames.append(model_frame)

    base_frame = model_frames[0][["text", "language", "sample_index"]].copy()
    for model_frame in model_frames:
        extra_columns = [
            column
            for column in model_frame.columns
            if column not in ("text", "language", "sample_index")
        ]
        base_frame = base_frame.merge(
            model_frame[["sample_index", *extra_columns]],
            on="sample_index",
            how="left",
        )

    ordered_columns = ["text", "language"]
    for model_inputs in model_inputs_list:
        model_label = model_inputs.model_label
        ordered_columns.extend(
            [
                f"{model_label}_zeroshot_score",
                f"{model_label}_zeroshot_reason",
                f"{model_label}_cot_score",
                f"{model_label}_cot_reason",
            ]
        )
    ordered_columns.append("sample_index")
    merged_frame = base_frame.loc[:, ordered_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_frame.to_excel(output_path, index=False)
    if len(merged_frame) != EXPECTED_ROW_COUNT:
        logging.warning(
            "Expected %s rows but got %s for %s",
            EXPECTED_ROW_COUNT,
            len(merged_frame),
            language_root,
        )
    logging.info(
        "Saved merged file %s with %s rows and %s columns",
        output_path,
        len(merged_frame),
        len(merged_frame.columns),
    )


if __name__ == "__main__":
    for language, language_root in LANGUAGE_OUTPUT_ROOTS.items():
        output_file = language_root / f"merged_{language}_models.xlsx"
        merge_language_outputs(language_root, output_file)
