import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "human" / "analyze_gender_difference_pipeline.py"
MODULE_SPEC = importlib.util.spec_from_file_location("analyze_gender_difference_pipeline", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
analyze_gender_difference_pipeline = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(analyze_gender_difference_pipeline)


def test_build_default_output_dir_adds_gender_slug():
    input_long_path = Path("human/outputs/final_clean_long.csv")

    assert analyze_gender_difference_pipeline.build_default_output_dir(input_long_path, None) == Path(
        "human/outputs/final_clean_long_gender_difference_analysis"
    )
    assert analyze_gender_difference_pipeline.build_default_output_dir(input_long_path, "男") == Path(
        "human/outputs/final_clean_long_male_raters_gender_difference_analysis"
    )
    assert analyze_gender_difference_pipeline.build_default_output_dir(input_long_path, "女") == Path(
        "human/outputs/final_clean_long_female_raters_gender_difference_analysis"
    )


def test_load_human_long_frame_filters_gender_and_repeat_trials(tmp_path):
    input_long_path = tmp_path / "demo_long.csv"
    input_long_path.write_text(
        "\n".join(
            [
                "condition,template_id,shown_version,shown_text,dimension_1,dimension_2,dimension_3,response_value,is_repeat_trial,gender",
                "attack_3pt,t1,男人版,alpha,d1,d2,d3,1,FALSE,男",
                "attack_3pt,t1,女人版,beta,d1,d2,d3,2,FALSE,男",
                "attack_3pt,t1,男人版,alpha,d1,d2,d3,1,TRUE,男",
                "attack_7pt_likert,t2,男人版,gamma,d1,d2,d3,3,FALSE,女",
                "attack_7pt_likert,t2,女人版,delta,d1,d2,d3,4,FALSE,女",
            ]
        )
        + "\n",
        encoding="utf-8-sig",
    )

    male_frame = analyze_gender_difference_pipeline.load_human_long_frame(
        input_path=input_long_path,
        include_repeat_trials=False,
        participant_gender="男",
    )
    female_frame = analyze_gender_difference_pipeline.load_human_long_frame(
        input_path=input_long_path,
        include_repeat_trials=False,
        participant_gender="女",
    )

    assert male_frame["gender"].tolist() == ["男", "男"]
    assert female_frame["gender"].tolist() == ["女", "女"]
    assert male_frame["is_repeat_trial"].tolist() == ["FALSE", "FALSE"]
    assert male_frame["response_value_num"].tolist() == [1, 2]
