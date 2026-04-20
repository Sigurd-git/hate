from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import prompts


def test_render_prompt_contains_sentence_marker():
    text = "Sample"
    rendered = prompts.render_prompt("en", "few_shot", text)
    assert "Sample" in rendered
    rendered_zh = prompts.render_prompt("zh", "chain_of_thought", text)
    assert "Sample" in rendered_zh


def test_render_attack_prompt_uses_condition_specific_3pt_scale():
    rendered = prompts.render_attack_prompt(
        "zh",
        "zero_shot",
        "测试句子",
        prompt_mode="attack_3pt",
    )
    assert "当前量表：attack_3pt" in rendered
    assert "2 = 严重攻击性" in rendered
    assert "attack_7pt_likert" not in rendered
    assert "attack_slider_0_100" not in rendered


def test_render_attack_prompt_uses_condition_specific_slider_scale():
    rendered = prompts.render_attack_prompt(
        "zh",
        "zero_shot",
        "测试句子",
        prompt_mode="attack_slider_0_100",
    )
    assert "当前量表：attack_slider_0_100" in rendered
    assert "100 = 极端攻击性" in rendered
    assert "attack_3pt" not in rendered
    assert "attack_7pt_likert" not in rendered


def test_render_attack_prompt_rejects_removed_legacy_attack_mode():
    with pytest.raises(KeyError):
        prompts.render_attack_prompt(
            "zh",
            "zero_shot",
            "测试句子",
            prompt_mode="attack",  # type: ignore[arg-type]
        )
