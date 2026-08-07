"""Tests for the LAK 2026 study scripts (papers/lak2026/scripts/)."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("google.genai")

from kcluster.core.question import Question  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "papers" / "lak2026" / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"lak2026_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", [p.stem for p in sorted(SCRIPTS.glob("*.py"))])
def test_scripts_import(name):
    _load(name)  # every script resolves its kcluster imports


def test_mix_n_match_plants_false_los():
    build_qualtrics = _load("build_qualtrics")

    questions = [
        Question({"id": f"q-{lo}-{i}", "type": "Multiple Choice",
                  "question": {"stem": "S?", "choices": [{"label": "a", "text": "x"}]},
                  "answerKey": "a", "lo": lo})
        for lo in ("lo-1", "lo-2", "lo-3") for i in range(4)
    ]
    mixed = build_qualtrics.mix_n_match(questions, num_los=2, group_sz=2, seed=0)

    assert len(mixed) == 2 * 2 * 2  # num_los * group_sz * (true + false)
    with_false = [q for q in mixed if "false_lo" in q]
    assert len(with_false) == 4  # half the questions get a false LO
    assert all(q["false_lo"] != q["lo"] for q in with_false)


def test_prepare_batch_requests_structure():
    gpt_validate = _load("gpt_validate")

    q = Question({"id": "q-0", "type": "Multiple Choice",
                  "question": {"stem": "S?", "choices": [{"label": "a", "text": "x"}]},
                  "answerKey": "a"})
    [req] = gpt_validate.prepare_batch_requests([q], model="gpt-4o-mini")

    assert req["custom_id"] == "q-0"
    assert req["body"]["model"] == "gpt-4o-mini"
    assert req["body"]["messages"][1]["content"] == q.prompt()
    assert req["body"]["response_format"]["json_schema"]["name"] == "mcq_response"
