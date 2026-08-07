"""Offline tests for the Gemini engine: response stamping, error capture,
the delay ramp, and the two answer parsers."""

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("google.genai")

from kcluster.engine import gemini  # noqa: E402
from kcluster.engine.gemini import (  # noqa: E402
    GeminiEngine,
    parse_logprob_choices,
    parse_text_choices,
    save_json_responses,
    sorted_choices,
)


class FakeModels:
    def __init__(self, behavior):
        self.behavior = behavior

    async def generate_content(self, model, contents, config):
        result = self.behavior(contents)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(to_json_dict=lambda: result)


def _engine(behavior, monkeypatch):
    monkeypatch.setattr(gemini.genai, "Client",
                        lambda **kw: SimpleNamespace(aio=SimpleNamespace(models=FakeModels(behavior))))
    return GeminiEngine("gemini-test", api_key="k")


def test_get_response_stamps_ids_and_captures_errors(monkeypatch):
    def behavior(contents):
        if "boom" in contents[0]:
            return RuntimeError("quota")
        return {"candidates": ["ok"]}

    engine = _engine(behavior, monkeypatch)
    ok = json.loads(asyncio.run(engine.get_response(["hi"], None, "r-1")))
    assert ok == {"id": "r-1", "candidates": ["ok"]}

    err = json.loads(asyncio.run(engine.get_response(["boom"], None, "r-2")))
    assert err == {"id": "r-2", "error": "quota"}


def test_gather_responses_ramps_the_delay(monkeypatch):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    engine = _engine(lambda contents: {"n": contents[0]}, monkeypatch)
    monkeypatch.setattr(gemini.asyncio, "sleep", fake_sleep)

    jobs = [(["a"], None, "r-0"), (["b"], None, "r-1"), (["c"], None, "r-2")]
    responses = asyncio.run(engine.gather_responses(jobs, delay=0.5, desc="test"))

    assert [json.loads(r)["id"] for r in responses] == ["r-0", "r-1", "r-2"]
    assert sorted(sleeps) == [0.5, 1.0, 1.5]  # each call starts later than the last


def test_parse_text_choices():
    responses = [json.dumps({"id": "q-0", "candidates": [
        {"content": {"parts": [{"text": " B "}]}}]})]
    assert parse_text_choices(responses, {"a", "b"}) == [{"id": "q-0", "answer": "b"}]

    bad = [json.dumps({"id": "q-1", "candidates": [{"content": {"parts": [{"text": "maybe"}]}}]})]
    with pytest.raises(AssertionError, match="Invalid answer"):
        parse_text_choices(bad, {"a", "b"})


def test_parse_logprob_choices():
    candidates = [
        {"token": "the", "log_probability": -0.1},   # not a choice token
        {"token": "a", "log_probability": -2.0},
        {"token": "b", "log_probability": -0.5},
    ]
    responses = [json.dumps({"id": "q-0", "candidates": [
        {"logprobs_result": {"top_candidates": [{"candidates": candidates}]}}]})]

    assert parse_logprob_choices(responses, {"a": "Yes", "b": "No"}) == [{"id": "q-0", "answer": "No"}]

    with pytest.raises(AssertionError, match="No valid choices"):
        sorted_choices({"x"}, candidates)


def test_save_json_responses(tmp_path):
    path = tmp_path / "raw.jsonl"
    save_json_responses(['{"id": "1"}', '{"id": "2"}'], str(path))
    assert path.read_text() == '{"id": "1"}\n{"id": "2"}\n'
