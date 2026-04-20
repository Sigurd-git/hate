import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "human" / "plot_participant_gender_combined_forest.py"
MODULE_SPEC = importlib.util.spec_from_file_location("plot_participant_gender_combined_forest", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
plot_participant_gender_combined_forest = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(plot_participant_gender_combined_forest)


def test_build_default_output_dir_uses_combined_suffix():
    input_long_path = Path("human/outputs/final_clean_long.csv")

    assert plot_participant_gender_combined_forest.build_default_output_dir(input_long_path) == Path(
        "human/outputs/final_clean_long_participant_gender_combined_analysis"
    )
