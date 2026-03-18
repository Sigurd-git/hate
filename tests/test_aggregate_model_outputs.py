import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import aggregate_model_outputs


def test_realign_outputs_to_source_repairs_shifted_rows():
    source_dataframe = pd.DataFrame(
        {
            "text": ["alpha", "beta", "gamma", "delta"],
            "sample_index": [0, 1, 2, 3],
        }
    )

    outputs_dataframe = pd.DataFrame(
        {
            "model_key": ["demo-model"] * 4,
            "prompt_name": ["zeroshot"] * 4,
            "text": ["alpha", "gamma", "delta", "delta"],
            "source_file": [
                "demo-model_sample_000000.xlsx",
                "demo-model_sample_000001.xlsx",
                "demo-model_sample_000002.xlsx",
                "demo-model_sample_000003.xlsx",
            ],
            "sample_index": pd.Series([0, 1, 2, 3], dtype="Int64"),
        }
    )

    realigned_outputs = aggregate_model_outputs.realign_outputs_to_source(
        source_dataframe=source_dataframe,
        outputs_dataframe=outputs_dataframe,
    )

    assert realigned_outputs["sample_index"].tolist() == [0, 2, 3, pd.NA]
    assert realigned_outputs["alignment_method"].tolist() == [
        "sample_index",
        "text",
        "text",
        "unresolved",
    ]
    assert realigned_outputs["reported_sample_index"].tolist() == [0, 1, 2, 3]
