import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pipeline


def test_load_samples(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("text,language\nA,en\nB,zh\n", encoding="utf-8")
    samples = pipeline.load_samples(str(source))
    assert [s.text for s in samples] == ["A", "B"]
    assert [s.language for s in samples] == ["en", "zh"]


def test_build_prompt_injects_text():
    sample = pipeline.Sample(text="Hi", language="en")
    prompt = pipeline.build_prompt(sample)
    assert "Hi" in prompt


def test_request_score_uses_structured_output(monkeypatch):
    sample = pipeline.Sample(text="Test", language="en")

    def fake_post(url, headers, payload, timeout):
        assert payload["response_format"]["json_schema"]["schema"] == pipeline.STRUCTURE_SCHEMA
        assert payload["messages"][0]["content"].startswith("Evaluate")
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"score": 2, "reason": "short"})
                    }
                }
            ]
        }

    monkeypatch.setattr(pipeline, "_post_json", fake_post)

    result = pipeline.request_score(
        sample,
        model="openrouter/test-model",
        api_key="key",
    )
    assert result == {"score": 2, "reason": "short"}
