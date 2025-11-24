"""Batch runner covering languages, prompt paradigms, datasets, and model choices."""
# 批处理运行器：遍历语言、提示范式、数据集、模型的所有组合

from __future__ import annotations  # 允许未来 Python 的类型注解行为

import logging
from glob import glob  # 用于文件通配符匹配
from pathlib import Path  # 用于处理路径对象
from typing import Dict, List, Sequence  # 类型提示

import polars as pl  # 使用 Polars 作为 DataFrame 库（比 pandas 更快）

from prompts import LANGUAGES, PARADIGMS, PromptParadigm  # 导入语言列表、范式列表、自定义范式类型
import pipeline  # 自己写的 pipeline 模块，用于加载样本、评分等

# 配置日志格式
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# 数据集类别对应的文件匹配模式
DATASET_PATTERNS: Dict[str, str] = {
    # "natural": "data/natural/*.csv",
    # "group_swap": "data/group_swap/*.csv",
    "test": "data/sample.csv",  # 示例数据集
}

# 要运行的模型列表（模型名称将用于 API 调用或 pipeline）
MODELS: Sequence[str] = (
    "openai/gpt-5.1",
    "anthropic/claude-sonnet-4.5",
    # "z-ai/glm-4.6",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-v3.2-exp",
    # "baidu/ernie-x1",
    # "bigscience/bloomz",
)

OUTPUT_ROOT = Path("outputs")  # 输出根目录
BATCH_SIZE = 10  # 每批处理的样本数


def discover_datasets(pattern: str) -> List[Path]:
    """Return sorted dataset paths for a given glob pattern."""
    # 根据文件匹配模式找到数据集文件，返回排序后的列表
    matches = sorted(Path(candidate) for candidate in glob(pattern))
    if not matches:
        logging.warning("No datasets found for pattern: %s", pattern)
    return matches


def iter_dataset_map() -> Dict[str, List[Path]]:
    """Map dataset categories to their resolved files."""
    # 构造 {数据集类型: 文件列表} 的映射
    dataset_map: Dict[str, List[Path]] = {}
    for category, pattern in DATASET_PATTERNS.items():
        dataset_map[category] = discover_datasets(pattern)
    return dataset_map


def build_output_directory(
    dataset_type: str,
    dataset_label: str,
    language: str,
    paradigm: PromptParadigm,
) -> Path:
    """Return the directory where batch XLSX files should live."""
    # 输出目录结构：outputs/{类型}/{标签}/{语言}/{范式}/
    return OUTPUT_ROOT / dataset_type / dataset_label / language / paradigm


def build_batch_prefix(model: str) -> str:
    """Return the filename prefix for a batch file."""
    # 模型名中的斜杠替换成下划线作为文件名前缀
    return model.replace("/", "_")


def build_batch_path(
    dataset_type: str,
    dataset_label: str,
    model: str,
    language: str,
    paradigm: PromptParadigm,
    batch_index: int,
) -> Path:
    """Return path for a single XLSX batch."""
    # 构建输出文件路径
    output_dir = build_output_directory(dataset_type, dataset_label, language, paradigm)
    prefix = build_batch_prefix(model)
    return output_dir / f"{prefix}_batch_{batch_index:03d}.xlsx"


def discover_existing_batches(
    dataset_type: str,
    dataset_label: str,
    model: str,
    language: str,
    paradigm: PromptParadigm,
) -> Dict[int, Path]:
    """Find already generated XLSX batches for a given configuration."""
    # 查找已有的批处理文件
    output_dir = build_output_directory(dataset_type, dataset_label, language, paradigm)
    prefix = build_batch_prefix(model)
    pattern = str(output_dir / f"{prefix}_batch_*.xlsx")
    matches = sorted(Path(candidate) for candidate in glob(pattern))
    existing: Dict[int, Path] = {}

    # 从文件名中解析 batch_index
    for match in matches:
        stem = match.stem
        marker = stem.split("_batch_")[-1]
        if marker.isdigit():
            existing[int(marker)] = match

    # 如果找到已有的 batch，则打印日志
    if matches:
        logging.info(
            "Found %s existing batches for model=%s language=%s paradigm=%s dataset=%s",
            len(matches),
            model,
            language,
            paradigm,
            dataset_label,
        )
    return existing


def save_batch_results(
    target: Path,
    rows: List[Dict[str, object]],
    metadata: Dict[str, object],
) -> None:
    """Write batch results to XLSX."""
    # 确保目标目录存在
    target.parent.mkdir(parents=True, exist_ok=True)

    # 在每一行结果中加入 metadata 字段
    enriched_rows = [
        {
            **row,
            "model": metadata["model"],
            "prompt_paradigm": metadata["paradigm"],
            "dataset_type": metadata["dataset_type"],
            "dataset_label": metadata["dataset_label"],
            "batch_index": metadata["batch_index"],
        }
        for row in rows
    ]

    # 写入 Excel 文件
    frame = pl.DataFrame(enriched_rows)
    frame.write_excel(target, worksheet="results")

    logging.info(
        "Saved batch_index=%s samples=%s target=%s",
        metadata["batch_index"],
        len(rows),
        target,
    )


def run_experiments(limit: int | None = None) -> None:
    """Iterate every combination of language, paradigm, dataset, and model."""
    # 主流程：遍历所有模型 × 语言 × 范式 × 数据集，并执行评分
    dataset_map = iter_dataset_map()
    for dataset_type, files in dataset_map.items():
        for dataset_path in files:
            dataset_label = dataset_path.stem  # 文件名的主干作为标签
            logging.info(
                "Processing dataset type=%s file=%s", dataset_type, dataset_path
            )
            for model in MODELS:
                for language in LANGUAGES:
                    # 加载数据集样本（可能按语言过滤）
                    samples = pipeline.load_samples(
                        str(dataset_path), limit=limit, language=language
                    )
                    if not samples:
                        logging.info(
                            "No samples remain for dataset=%s language=%s",
                            dataset_label,
                            language,
                        )
                        continue

                    for paradigm in PARADIGMS:
                        # 查找已有的 batch 文件以避免重复计算
                        existing_map = discover_existing_batches(
                            dataset_type,
                            dataset_label,
                            model,
                            language,
                            paradigm,
                        )

                        total_batches = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE

                        for batch_index in range(total_batches):
                            # 如果该 batch 已存在，跳过
                            if batch_index in existing_map:
                                logging.info(
                                    "Skip existing batch model=%s language=%s paradigm=%s dataset=%s batch=%s",
                                    model,
                                    language,
                                    paradigm,
                                    dataset_label,
                                    batch_index,
                                )
                                continue

                            # 计算 batch 的样本切片范围
                            start = batch_index * BATCH_SIZE
                            end = min(start + BATCH_SIZE, len(samples))
                            batch_samples = samples[start:end]

                            if not batch_samples:
                                continue

                            logging.info(
                                "Scoring batch model=%s language=%s paradigm=%s dataset=%s batch=%s",
                                model,
                                language,
                                paradigm,
                                dataset_label,
                                batch_index,
                            )

                            # 对当前 batch 进行评分
                            results, batch_aborted = pipeline.score_samples_with_status(
                                batch_samples,
                                model=model,
                                prompt_paradigm=paradigm,
                            )

                            # 如果由于 finish_reason 或其他原因被中断，跳过保存
                            if batch_aborted:
                                logging.warning(
                                    "Skip saving batch due to length finish_reason model=%s language=%s paradigm=%s dataset=%s batch=%s",
                                    model,
                                    language,
                                    paradigm,
                                    dataset_label,
                                    batch_index,
                                )
                                continue

                            # 元数据
                            metadata = {
                                "model": model,
                                "paradigm": paradigm,
                                "dataset_type": dataset_type,
                                "dataset_label": dataset_label,
                                "batch_index": batch_index,
                            }

                            # 目标路径
                            target = build_batch_path(
                                dataset_type,
                                dataset_label,
                                model,
                                language,
                                paradigm,
                                batch_index,
                            )

                            # 保存结果
                            save_batch_results(target, results, metadata)


# 程序入口
if __name__ == "__main__":
    run_experiments(limit=None)
