from __future__ import annotations

import argparse
import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class DatasetConfig:
    language_name: str
    language_code: str
    dataset_keyword: str
    merged_keyword: str
    text_column: str
    label_column: str
    vectorizer_settings: dict[str, object]


LLM_SCORE_COLUMNS: dict[str, str] = {
    "chatgpt5.1_zeroshot_score": "ChatGPT5.1 ZeroShot",
    "chatgpt5.1_cot_score": "ChatGPT5.1 CoT",
    "claude4.5_zeroshot_score": "Claude4.5 ZeroShot",
    "claude4.5_cot_score": "Claude4.5 CoT",
}
LLM_MAX_SCORE = 5.0


def setup_logging() -> None:
    """Configure logging for clear, stepwise processing details."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def list_sorted_paths(pattern: str, recursive: bool = False) -> list[Path]:
    """Return sorted paths matched by glob for deterministic processing."""
    matched_paths = glob.glob(pattern, recursive=recursive)
    return [Path(path) for path in sorted(matched_paths)]


def select_path_by_keyword(paths: Sequence[Path], keyword: str) -> Path:
    """Pick a single path containing the keyword to avoid ambiguous matches."""
    matching_paths = [path for path in paths if keyword in path.as_posix()]
    if len(matching_paths) != 1:
        raise ValueError(
            f"Expected exactly one path containing '{keyword}', found {len(matching_paths)}."
        )
    return matching_paths[0]


def load_dataset(
    dataset_path: Path, text_column: str, label_column: str
) -> pd.DataFrame:
    """Load dataset and ensure expected columns are present."""
    logging.info("Loading dataset: %s", dataset_path)
    data_frame = pd.read_excel(dataset_path)
    missing_columns = [
        column for column in [text_column, label_column] if column not in data_frame
    ]
    if missing_columns:
        raise ValueError(f"Missing columns in {dataset_path}: {missing_columns}")
    prepared_frame = data_frame[[text_column, label_column]].copy()
    prepared_frame["sample_index"] = prepared_frame.index.astype(int)
    prepared_frame[text_column] = prepared_frame[text_column].fillna("").astype(str)
    prepared_frame[label_column] = prepared_frame[label_column].astype(int)
    return prepared_frame


def split_indices(
    sample_indices: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices once so the baseline and LLMs use the same test subset."""
    train_indices, test_indices = train_test_split(
        sample_indices,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )
    return np.array(train_indices), np.array(test_indices)


def train_tfidf_svm(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    vectorizer_settings: dict[str, object],
    random_seed: int,
) -> tuple[TfidfVectorizer, CalibratedClassifierCV]:
    """Fit TF-IDF vectorizer and a calibrated LinearSVC classifier."""
    logging.info("Training TF-IDF + LinearSVC baseline model")
    vectorizer = TfidfVectorizer(**vectorizer_settings)
    train_features = vectorizer.fit_transform(train_texts)
    base_classifier = LinearSVC(class_weight="balanced", random_state=random_seed)
    calibrated_classifier = CalibratedClassifierCV(
        estimator=base_classifier,
        method="sigmoid",
        cv=3,
    )
    calibrated_classifier.fit(train_features, train_labels)
    return vectorizer, calibrated_classifier


def compute_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    score_values: np.ndarray | None,
    probability_values: np.ndarray | None,
) -> dict[str, float]:
    """Compute binary classification metrics with optional ROC-AUC and Brier score."""
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision": float(
            precision_score(true_labels, predicted_labels, zero_division=0)
        ),
        "recall": float(recall_score(true_labels, predicted_labels, zero_division=0)),
        "f1": float(f1_score(true_labels, predicted_labels, zero_division=0)),
    }
    if score_values is None:
        metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float(roc_auc_score(true_labels, score_values))
    if probability_values is None:
        metrics["brier"] = float("nan")
    else:
        metrics["brier"] = float(
            brier_score_loss(true_labels, y_proba=probability_values)
        )
    return metrics


def build_roc_curve_points(
    true_labels: np.ndarray,
    score_values: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Prepare ROC curve points for plotting."""
    false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, score_values)
    return pd.DataFrame(
        {
            "false_positive_rate": false_positive_rate,
            "true_positive_rate": true_positive_rate,
            "model": model_name,
        }
    )


def evaluate_baseline_model(
    dataset_frame: pd.DataFrame,
    text_column: str,
    label_column: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    vectorizer_settings: dict[str, object],
    random_seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Train the baseline and compute metrics plus ROC curve data."""
    train_texts = dataset_frame.loc[train_indices, text_column].tolist()
    train_labels = dataset_frame.loc[train_indices, label_column].to_numpy()
    test_texts = dataset_frame.loc[test_indices, text_column].tolist()
    test_labels = dataset_frame.loc[test_indices, label_column].to_numpy()

    vectorizer, classifier = train_tfidf_svm(
        train_texts,
        train_labels,
        vectorizer_settings,
        random_seed,
    )
    test_features = vectorizer.transform(test_texts)
    predicted_labels = classifier.predict(test_features)
    probability_values = classifier.predict_proba(test_features)[:, 1]
    baseline_metrics = compute_metrics(
        test_labels, predicted_labels, probability_values, probability_values
    )
    roc_curve_frame = build_roc_curve_points(
        test_labels, probability_values, "TF-IDF + LinearSVC"
    )
    return baseline_metrics, roc_curve_frame


def evaluate_llm_outputs(
    merged_path: Path,
    label_frame: pd.DataFrame,
    label_column: str,
    test_indices: np.ndarray | None,
    threshold: float,
) -> tuple[list[dict[str, float]], list[pd.DataFrame]]:
    """Evaluate LLM scores on the specified split or the full dataset."""
    logging.info("Loading merged LLM outputs: %s", merged_path)
    merged_frame = pd.read_excel(merged_path)
    if "sample_index" not in merged_frame:
        raise ValueError(f"Missing sample_index in {merged_path}")
    merged_frame = merged_frame.merge(
        label_frame[["sample_index", label_column]],
        on="sample_index",
        how="inner",
    )
    if test_indices is None:
        evaluation_frame = merged_frame.copy()
        logging.info("Evaluating LLM outputs on all %d samples", len(evaluation_frame))
    else:
        evaluation_frame = merged_frame[
            merged_frame["sample_index"].isin(test_indices)
        ].copy()
        logging.info(
            "Evaluating LLM outputs on %d test samples", len(evaluation_frame)
        )
    metrics_rows: list[dict[str, float]] = []
    roc_curve_frames: list[pd.DataFrame] = []

    for score_column, model_name in LLM_SCORE_COLUMNS.items():
        if score_column not in evaluation_frame:
            logging.warning("Missing score column: %s", score_column)
            continue
        score_values = (
            pd.to_numeric(evaluation_frame[score_column], errors="coerce")
            .fillna(0)
            .to_numpy()
        )
        probability_values = np.clip(score_values / LLM_MAX_SCORE, 0.0, 1.0)
        true_labels = evaluation_frame[label_column].to_numpy()
        predicted_labels = (score_values >= threshold).astype(int)
        model_metrics = compute_metrics(
            true_labels, predicted_labels, probability_values, probability_values
        )
        model_metrics["threshold"] = float(threshold)
        model_metrics["model"] = model_name
        metrics_rows.append(model_metrics)
        roc_curve_frames.append(
            build_roc_curve_points(true_labels, probability_values, model_name)
        )

    return metrics_rows, roc_curve_frames


def plot_roc_curves(
    roc_curve_frame: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Plot and save ROC curves using a color-blind friendly palette."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    figure, axis = plt.subplots(figsize=(7.5, 6))
    sns.lineplot(
        data=roc_curve_frame,
        x="false_positive_rate",
        y="true_positive_rate",
        hue="model",
        linewidth=2.0,
        ax=axis,
    )
    axis.plot(
        [0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Chance"
    )
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(title)
    axis.legend(title="Model", loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_metrics_table(metrics_rows: list[dict[str, float]], output_path: Path) -> None:
    """Save metrics as both Excel and CSV files."""
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_excel(output_path.with_suffix(".xlsx"), index=False)
    metrics_frame.to_csv(output_path.with_suffix(".csv"), index=False)


def evaluate_language(
    config: DatasetConfig,
    dataset_path: Path,
    merged_path: Path,
    output_dir: Path,
    test_size: float,
    random_seed: int,
    threshold: float,
) -> None:
    """Run baseline training and compare against LLM outputs for one language."""
    dataset_frame = load_dataset(dataset_path, config.text_column, config.label_column)
    sample_indices = dataset_frame["sample_index"].to_numpy()
    labels = dataset_frame[config.label_column].to_numpy()
    train_indices, test_indices = split_indices(
        sample_indices, labels, test_size, random_seed
    )
    logging.info(
        "Split sizes for %s: train=%d test=%d",
        config.language_name,
        len(train_indices),
        len(test_indices),
    )

    baseline_metrics, baseline_roc_frame = evaluate_baseline_model(
        dataset_frame,
        config.text_column,
        config.label_column,
        train_indices,
        test_indices,
        config.vectorizer_settings,
        random_seed,
    )
    baseline_metrics["model"] = "TF-IDF + LinearSVC"
    baseline_metrics["threshold"] = float("nan")

    llm_metrics_rows, llm_roc_frames = evaluate_llm_outputs(
        merged_path,
        dataset_frame,
        config.label_column,
        test_indices,
        threshold,
    )
    llm_metrics_rows_all, _ = evaluate_llm_outputs(
        merged_path,
        dataset_frame,
        config.label_column,
        None,
        threshold,
    )

    baseline_metrics["split"] = "test"
    for row in llm_metrics_rows:
        row["split"] = "test"
    for row in llm_metrics_rows_all:
        row["split"] = "all"

    metrics_rows = [baseline_metrics, *llm_metrics_rows, *llm_metrics_rows_all]
    for row in metrics_rows:
        row["language"] = config.language_name
        row["dataset"] = dataset_path.name

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_output_path = output_dir / f"metrics_{config.language_code}"
    save_metrics_table(metrics_rows, metrics_output_path)

    roc_frames = [baseline_roc_frame, *llm_roc_frames]
    roc_curve_frame = pd.concat(roc_frames, ignore_index=True)
    roc_output_path = output_dir / f"roc_{config.language_code}.png"
    plot_roc_curves(
        roc_curve_frame,
        roc_output_path,
        title=f"ROC Curves - {config.language_name}",
    )
    logging.info(
        "Saved metrics to %s.[xlsx|csv] and ROC to %s",
        metrics_output_path,
        roc_output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + SVM baseline and compare with LLM outputs.",
    )
    parser.add_argument(
        "--dataset-glob",
        default="data/5_new_*_balanced.xlsx",
        help="Glob pattern for dataset Excel files.",
    )
    parser.add_argument(
        "--merged-glob",
        default="outputs/natural/5_new_*_balanced/**/merged_*_models.xlsx",
        help="Glob pattern for merged LLM output files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/baseline_tfidf_svm",
        help="Directory to store metrics and ROC plots.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for splitting."
    )
    parser.add_argument(
        "--llm-threshold",
        type=float,
        default=0.5,
        help="Threshold to map LLM scores to binary labels.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    dataset_paths = list_sorted_paths(args.dataset_glob)
    merged_paths = list_sorted_paths(args.merged_glob, recursive=True)

    if not dataset_paths:
        raise FileNotFoundError(f"No datasets found for pattern: {args.dataset_glob}")
    if not merged_paths:
        raise FileNotFoundError(
            f"No merged outputs found for pattern: {args.merged_glob}"
        )

    configs: list[DatasetConfig] = [
        DatasetConfig(
            language_name="English",
            language_code="en",
            dataset_keyword="english",
            merged_keyword="english",
            text_column="text",
            label_column="label/2classes",
            vectorizer_settings={
                "ngram_range": (1, 2),
                "max_features": 50000,
                "stop_words": "english",
            },
        ),
        DatasetConfig(
            language_name="Chinese",
            language_code="zh",
            dataset_keyword="chinese",
            merged_keyword="chinese",
            text_column="text",
            label_column="label/2classes",
            vectorizer_settings={
                "ngram_range": (2, 4),
                "max_features": 50000,
                "analyzer": "char",
            },
        ),
    ]

    output_dir = Path(args.output_dir)

    for config in configs:
        dataset_path = select_path_by_keyword(dataset_paths, config.dataset_keyword)
        merged_path = select_path_by_keyword(merged_paths, config.merged_keyword)
        logging.info(
            "Processing language=%s dataset=%s merged=%s",
            config.language_name,
            dataset_path,
            merged_path,
        )
        evaluate_language(
            config,
            dataset_path,
            merged_path,
            output_dir,
            args.test_size,
            args.random_seed,
            args.llm_threshold,
        )


if __name__ == "__main__":
    main()
