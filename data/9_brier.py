import logging
from pathlib import Path
from typing import List

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def scores_to_probability(score_series: pd.Series) -> pd.Series:
    """
    Convert discrete 0-5 scores to calibrated probabilities via p = s / 5.
    """
    normalized_scores = score_series.fillna(0).clip(lower=0, upper=5)
    return normalized_scores / 5.0


def calculate_brier_score(probabilities: pd.Series, labels: pd.Series) -> float:
    """
    Compute the mean squared error between predicted probabilities and binary labels.
    """
    aligned_probabilities = probabilities.astype(float)
    aligned_labels = labels.astype(int)
    squared_errors = (aligned_probabilities - aligned_labels) ** 2
    return squared_errors.mean()


def compute_brier_for_all_models(
    excel_path: str,
    label_column: str = "label/2classes",
    zeroshot_suffix: str = "_zeroshot_score",
    cot_suffix: str = "_cot_score",
) -> pd.DataFrame:
    """
    Calculate Brier scores for every model column pair (zeroshot / CoT) in an Excel file.
    """
    dataset_path = Path(excel_path)
    logger.info("Loading annotated scores from %s", dataset_path)
    dataset_frame = pd.read_excel(dataset_path)

    ground_truth = dataset_frame[label_column].astype(int)
    rows: List[dict] = []

    for column_name in dataset_frame.columns:
        if not column_name.endswith(zeroshot_suffix):
            continue

        model_name = column_name[: -len(zeroshot_suffix)]
        logger.info("Computing Brier score for model=%s zeroshot", model_name)
        zeroshot_probabilities = scores_to_probability(dataset_frame[column_name])
        zeroshot_brier = calculate_brier_score(zeroshot_probabilities, ground_truth)
        rows.append(
            {
                "model": model_name,
                "setting": "zeroshot",
                "brier_score": zeroshot_brier,
            }
        )

        cot_column = f"{model_name}{cot_suffix}"
        if cot_column in dataset_frame.columns:
            logger.info("Computing Brier score for model=%s cot", model_name)
            cot_probabilities = scores_to_probability(dataset_frame[cot_column])
            cot_brier = calculate_brier_score(cot_probabilities, ground_truth)
            rows.append(
                {
                    "model": model_name,
                    "setting": "cot",
                    "brier_score": cot_brier,
                }
            )

    result_frame = pd.DataFrame(rows).sort_values(by=["model", "setting"]).reset_index(drop=True)
    logger.info("Finished computing Brier scores for %s entries", len(result_frame))
    return result_frame


if __name__ == "__main__":
    base_dir = Path("/Users/baoxuan/Desktop/研究生研究/llm毕业论文/hate/data")
    dataset_specs = [
        (
            base_dir / "8_new_chinesehatedata+zeroshot+cot.xlsx",
            base_dir / "9_new_chinesehatedata_model_brier_scores.xlsx",
        ),
        (
            base_dir / "8_new_englishhatedata_2400+zeroshot+cot.xlsx",
            base_dir / "9_new_englishhatedata_model_brier_scores.xlsx",
        ),
    ]

    for excel_path, output_path in dataset_specs:
        logger.info("Starting Brier score evaluation for %s", excel_path.name)
        brier_table = compute_brier_for_all_models(str(excel_path))
        brier_table.to_excel(output_path, index=False)
        logger.info("Brier scores saved to %s", output_path)
        print(f"Brier scores saved to: {output_path}")
