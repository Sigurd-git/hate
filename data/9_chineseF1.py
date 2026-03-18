import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from model_metrics_common import DATASET_SPECS, build_language_evaluation_frame


def scores_to_binary(score_series: pd.Series) -> pd.Series:
    """
    将 0–5 的评分转换为二分类的仇恨预测。

    规则：
    - 0     -> 0（非仇恨）
    - 1–5   -> 1（仇恨）

    参数
    ----
    score_series : pd.Series
        包含 [0, 5] 数值评分的 pandas Series。

    返回
    ----
    pd.Series
        由 0/1 预测组成的 Series。
    """
    # 先把 NaN 当作 0，再以 1 为阈值
    return (score_series.fillna(0) >= 1).astype(int)


def compute_f1_for_all_models(
    dataset_frame: pd.DataFrame,
    label_column: str = "label/2classes",
    zeroshot_suffix: str = "_zeroshot_score",
    cot_suffix: str = "_cot_score",
) -> pd.DataFrame:
    """
    计算单个 Excel 文件中所有模型的 zeroshot 与 CoT Precision/Recall/F1 分数。

    假设：
    - 真实标签位于 `label_column` 且取值为 0/1。
    - 每个模型至少存在一个 `{model_name}{zeroshot_suffix}` 列。
    - 若模型有 CoT 列，则命名为 `{model_name}{cot_suffix}`。

    示例列名：
    - 'openai/gpt-5.1_zeroshot_score'
    - 'openai/gpt-5.1_cot_score'
    - 'Claude4.5_zeroshot_score'
    - 'Claude4.5_cot_score'

    参数
    ----
    dataset_frame : pd.DataFrame
        包含标签与模型评分列的数据表。
    label_column : str, optional
        真实标签列名（默认 "label/2classes"）。
    zeroshot_suffix : str, optional
        zeroshot 列的后缀。
    cot_suffix : str, optional
        CoT 列的后缀。

    返回
    ----
    pd.DataFrame
        包含以下列：
        - 'model'     : 模型名（后缀前的部分）
        - 'setting'   : 'zeroshot' 或 'cot'
        - 'precision' : 精确率
        - 'recall'    : 召回率
        - 'f1'        : F1 分数
    """
    # Ground-truth labels (0/1)
    y_true = dataset_frame[label_column].astype(int)

    results = []

    for col in dataset_frame.columns:
        # Only look at zeroshot columns; infer corresponding CoT columns from them
        if col.endswith(zeroshot_suffix):
            # Extract model name by stripping the zeroshot suffix
            model_name = col[: -len(zeroshot_suffix)]

            # Zeroshot scores and predictions
            zs_scores = dataset_frame[col]
            y_pred_zs = scores_to_binary(zs_scores)
            precision_zs = precision_score(y_true, y_pred_zs, zero_division=0)
            recall_zs = recall_score(y_true, y_pred_zs, zero_division=0)
            f1_zs = f1_score(y_true, y_pred_zs)

            results.append(
                {
                    "model": model_name,
                    "setting": "zeroshot",
                    "precision": precision_zs,
                    "recall": recall_zs,
                    "f1": f1_zs,
                }
            )

            # Now check if the corresponding CoT column exists
            cot_col = f"{model_name}{cot_suffix}"
            if cot_col in dataset_frame.columns:
                cot_scores = dataset_frame[cot_col]
                y_pred_cot = scores_to_binary(cot_scores)
                precision_cot = precision_score(y_true, y_pred_cot, zero_division=0)
                recall_cot = recall_score(y_true, y_pred_cot, zero_division=0)
                f1_cot = f1_score(y_true, y_pred_cot)

                results.append(
                    {
                        "model": model_name,
                        "setting": "cot",
                    "precision": precision_cot,
                    "recall": recall_cot,
                    "f1": f1_cot,
                    }
                )

    # Collect all results into a DataFrame for easy inspection or saving
    result_df = pd.DataFrame(results).sort_values(
        by=["model", "setting"]
    ).reset_index(drop=True)

    return result_df


if __name__ == "__main__":
    dataset_spec = DATASET_SPECS["zh"]
    evaluation_frame = build_language_evaluation_frame(dataset_spec, save_combined_output=True)
    f1_table = compute_f1_for_all_models(
        evaluation_frame,
        label_column=dataset_spec.label_column,
    )
    f1_table.to_excel(dataset_spec.f1_output_path, index=False)
    print(f"F1 scores saved to: {dataset_spec.f1_output_path}")
