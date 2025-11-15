import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pipeline


def test_load_samples(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("text,language\nA,en\nB,zh\n", encoding="utf-8")
    samples = pipeline.load_samples(str(source))
    assert [s.text for s in samples] == ["A", "B"]
    assert [s.language for s in samples] == ["en", "zh"]


def test_request_score_uses_structured_output(monkeypatch):
    sample = pipeline.Sample(text="Test", language="en")

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def model_dump(self):
            return self._payload

    class DummyClient:
        def __init__(self):
            self.captured_kwargs = None
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            self.captured_kwargs = kwargs
            return DummyResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps({"score": 2, "reason": "short"})
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

    dummy_client = DummyClient()
    monkeypatch.setattr(pipeline, "_get_openrouter_client", lambda: dummy_client)

    result = pipeline.request_score(
        sample,
        model="openrouter/test-model",
        prompt_paradigm="few_shot",
        metadata={"referer": "https://example.com", "title": "demo-app"},
    )
    assert isinstance(result, pipeline.ScoreResponse)
    assert result.payload == {"score": 2, "reason": "short"}
    assert result.finish_reason == "stop"
    assert dummy_client.captured_kwargs is not None
    assert dummy_client.captured_kwargs["response_format"]["json_schema"]["schema"] == pipeline.STRUCTURE_SCHEMA
    assert sample.text in dummy_client.captured_kwargs["messages"][0]["content"]
    assert dummy_client.captured_kwargs["extra_headers"] == {
        "HTTP-Referer": "https://example.com",
        "X-Title": "demo-app",
    }
    assert dummy_client.captured_kwargs["extra_body"] == {
        "metadata": {"referer": "https://example.com", "title": "demo-app"}
    }


def test_run_batch_filters_language(monkeypatch, tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("text,language\nA,en\nB,zh\n", encoding="utf-8")

    captured = []

    def fake_request(sample, model, prompt_paradigm):
        captured.append((sample.text, sample.language, prompt_paradigm, model))
        return pipeline.ScoreResponse(payload={"score": 1}, finish_reason="stop")

    monkeypatch.setattr(pipeline, "request_score", fake_request)

    results = pipeline.run_batch(
        str(source),
        model="demo/model",
        language="en",
        prompt_paradigm="chain_of_thought",
    )
    assert all(entry["language"] == "en" for entry in results)
    assert captured == [("A", "en", "chain_of_thought", "demo/model")]
