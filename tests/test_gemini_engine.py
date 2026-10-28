"""Offline tests for the Gemini engine: response stamping, error capture,
the delay ramp, and the two answer parsers."""

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("google.genai")

from kcluster.engine import gemini
from kcluster.engine.gemini import (
    GeminiEngine,
    as_contents,
    parse_logprob_choices,
    parse_text_choices,
    save_json_responses,
    sorted_choices,
)


class FakeModels:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = []

    async def generate_content(self, model, contents, config):
        self.calls.append((contents, config))
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


def test_a_conversation_is_written_as_gemini_turns():
    """The prompts the local engine takes (``kcluster.engine.local``) address Gemini too."""
    assert as_contents("plain") == ("plain", None)
    contents, system = as_contents([{"role": "system", "content": "Be brief."},
                                    {"role": "user", "content": "Write an exercise."},
                                    {"role": "assistant", "content": "Exercise 1:"}])
    assert system == "Be brief."
    assert [(c.role, c.parts[0].text) for c in contents] == [("user", "Write an exercise."),
                                                             ("model", "Exercise 1:")]  # the model's turn to continue
    with pytest.raises(ValueError, match="role is 'tool'"):
        as_contents([{"role": "tool", "content": "x"}])


def test_get_response_takes_a_conversation_and_puts_its_system_message_in_the_config(monkeypatch):
    engine = _engine(lambda contents: {"ok": True}, monkeypatch)
    models = engine.client.aio.models
    config = gemini.types.GenerateContentConfig(temperature=0.1)
    chat = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
    assert json.loads(asyncio.run(engine.get_response(chat, config, "r-0"))) == {"id": "r-0", "ok": True}
    contents, sent = models.calls[-1]
    assert [c.role for c in contents] == ["user"]
    assert sent.system_instruction == "Be brief." and sent.temperature == 0.1
    assert config.system_instruction is None  # the caller's config is not changed

    asyncio.run(engine.get_response(chat[1:], {"temperature": 0.2}, "r-1"))  # no system message: config as given
    assert models.calls[-1][1] == {"temperature": 0.2}
    asyncio.run(engine.get_response(["raw", "genai contents"], None, "r-2"))  # google-genai's own, as they are
    assert models.calls[-1] == (["raw", "genai contents"], None)

    with pytest.raises(ValueError, match="give one or the other"):
        asyncio.run(engine.get_response(chat, {"system_instruction": "Be long."}, "r-3"))


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
