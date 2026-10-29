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


# --- Images: a prompt places one with the engine's token, the sources go beside it (as the local engine) ---

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8  # the magic is all the engine reads: Gemini decodes the file


def _kinds(parts):
    return [("text", p.text) if p.text is not None else ("image", p.inline_data.mime_type) for p in parts]


def test_a_string_prompt_with_images_becomes_the_users_turn_in_parts(tmp_path):
    (tmp_path / "a.png").write_bytes(PNG)
    token = GeminiEngine.image_token
    contents, system = as_contents(f"{token}\nWhat does the graph show? {token}", [str(tmp_path / "a.png"), PNG])
    assert system is None
    assert [c.role for c in contents] == ["user"]
    assert _kinds(contents[0].parts) == [("image", "image/png"), ("text", "\nWhat does the graph show? "),
                                         ("image", "image/png")]  # an empty piece makes no part
    assert contents[0].parts[0].inline_data.data == PNG
    assert as_contents("plain", []) == ("plain", None)  # no images: the string as it was


def test_a_conversation_places_images_in_its_messages():
    token = GeminiEngine.image_token
    chat = [{"role": "system", "content": "Be brief."},
            {"role": "user", "content": f"Figure: {token}\nName the concept."},
            {"role": "assistant", "content": "The concept is"}]
    contents, system = as_contents(chat, [gemini_data_uri()])
    assert system == "Be brief."
    assert [(c.role, _kinds(c.parts)) for c in contents] == [
        ("user", [("text", "Figure: "), ("image", "image/jpeg"), ("text", "\nName the concept.")]),
        ("model", [("text", "The concept is")])]


def gemini_data_uri():
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 8).decode()


def test_images_must_match_the_prompts_places():
    token = GeminiEngine.image_token
    with pytest.raises(ValueError, match="places 1 image"):
        as_contents(f"{token} and?", [])
    with pytest.raises(ValueError, match="places 0 image"):
        as_contents([{"role": "user", "content": "no place"}], [PNG])
    with pytest.raises(ValueError, match="system message"):
        as_contents([{"role": "system", "content": f"See {token}"}, {"role": "user", "content": "Hi"}], [PNG])


def test_show_figures_addresses_gemini_as_it_does_a_local_model():
    from kcluster.core.prompts import FIGURE_MARKER, show_figures

    shown = show_figures(f"{FIGURE_MARKER}\nWhich graph?", GeminiEngine.image_token)
    contents, _ = as_contents(shown, [PNG])
    assert _kinds(contents[0].parts) == [("image", "image/png"), ("text", "\nWhich graph?")]


def test_get_response_sends_a_prompts_images_and_refuses_them_on_raw_contents(monkeypatch):
    engine = _engine(lambda contents: {"ok": True}, monkeypatch)
    models = engine.client.aio.models
    token = engine.image_token
    assert json.loads(asyncio.run(engine.get_response(f"{token} Describe.", None, "r-0", images=[PNG]))) == \
        {"id": "r-0", "ok": True}
    contents, _ = models.calls[-1]
    assert _kinds(contents[0].parts) == [("image", "image/png"), ("text", " Describe.")]

    jobs = [("text only", None, "r-1"), (f"{token} Describe.", None, "r-2", [PNG])]  # three- and four-tuples
    responses = asyncio.run(engine.gather_responses(jobs, desc="test"))
    assert [json.loads(r)["id"] for r in responses] == ["r-1", "r-2"]
    assert models.calls[-2][0] == "text only" and _kinds(models.calls[-1][0][0].parts)[0] == ("image", "image/png")

    with pytest.raises(ValueError, match="google-genai's own contents"):
        asyncio.run(engine.get_response(["raw", "genai contents"], None, "r-3", images=[PNG]))
