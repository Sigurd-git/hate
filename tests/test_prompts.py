from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import prompts


def test_render_prompt_contains_sentence_marker():
    text = "Sample"
    rendered = prompts.render_prompt("en", "few_shot", text)
    assert "Sample" in rendered
    rendered_zh = prompts.render_prompt("zh", "chain_of_thought", text)
    assert "Sample" in rendered_zh
