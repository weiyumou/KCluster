"""End-to-end tests of the elearning24 (OLI Torus JSON) dataset drivers."""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("requests")
pytest.importorskip("PIL")

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER_DIR = REPO_ROOT / "datasets" / "elearning24"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, DRIVER_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # mcq/mfb resolve their "from processing import ..."
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def drivers():
    processing = _load("processing")
    return {"processing": processing, "mcq": _load("mcq"), "mfb": _load("mfb")}


def _p(text):
    return {"type": "p", "children": [{"text": text}]}


def _response(choice_id, score, feedback_text):
    return {"rule": f"input like {{{choice_id}}}", "score": score,
            "feedback": {"content": [_p(feedback_text)]}}


def _write_objective(raw_dir, obj_id="obj1", title="identify mass"):
    (raw_dir / f"{obj_id}.json").write_text(json.dumps({"type": "Objective", "title": title}))


def test_extract_mcqs(drivers, tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_objective(raw_dir)
    activity = {
        "objectives": {"p1": ["obj1", "missing-obj"]},
        "content": {
            "stem": {"content": [_p("Which is heavier?")]},
            "choices": [
                {"id": "c1", "content": [_p("Iron")]},
                {"id": "c2", "content": [_p("Feather")]},
            ],
            "authoring": {"parts": [{"id": "p1", "responses": [
                _response("c1", 1, "Right."),
                _response("c2", 0, "Wrong."),
                _response(".*", 0, "Try again."),
            ]}]},
        },
    }
    (raw_dir / "100.json").write_text(json.dumps(activity))

    step_df = pd.DataFrame(
        {
            "Problem Hierarchy": ["U1"],
            "Problem Name": ["Activity 100"],
            "Step Name": ["Activity 100, Part p1 Multiple choice submission"],
        }
    )
    [q] = drivers["mcq"].extract_mcqs(step_df, str(raw_dir))

    assert q["id"] == "mcq-100"
    assert q["question"]["stem"] == "Which is heavier?"
    assert [c["label"] for c in q["question"]["choices"]] == ["a", "b"]
    [ans_text] = [c["text"] for c in q["question"]["choices"] if c["label"] == q["answerKey"]]
    assert ans_text == "Iron"
    assert q["feedback"][q["answerKey"]] == "Right."
    assert q["objectives"] == ["Identify mass"]  # missing-obj has no file and is skipped
    assert q["ds-step-name"] == ["Activity 100, Part p1 Multiple choice submission"]


def test_merge_duplicate_mcqs(drivers):
    def q(qid, step):
        return {"id": qid, "question": {"stem": "S?", "choices": [{"label": "a", "text": "X"}]},
                "images": [], "objectives": ["o1"], "ds-problem-hierarchy": ["U"],
                "ds-problem-name": ["P"], "ds-step-name": [step]}

    merged = drivers["processing"].merge_duplicate_mcqs([q("mcq-1", "s1"), q("mcq-2", "s2")])
    [m] = merged
    assert m["id"] == "mcq-1-2"
    assert sorted(m["ds-step-name"]) == ["s1", "s2"]


def test_extract_mfb(drivers, tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_objective(raw_dir, title="name colors")
    activity = {
        "objectives": {"p1": ["obj1"], "p2": ["obj1"]},
        "content": {
            "inputs": [
                {"id": "i1", "partId": "p1", "choiceIds": ["c1", "c2"]},
                {"id": "i2", "partId": "p2", "choiceIds": ["c1", "c2"]},
            ],
            "choices": [
                {"id": "c1", "content": [_p("Red")]},
                {"id": "c2", "content": [_p("Blue")]},
            ],
            "stem": {"content": [{"type": "p", "children": [
                {"text": "The sky is"},
                {"type": "input_ref", "id": "i2"},
                {"text": "and fire is"},
                {"type": "input_ref", "id": "i1"},
            ]}]},
            "authoring": {"parts": [
                {"id": "p1", "responses": [_response("c1", 1, "Yes."), _response("c2", 0, "No.")]},
                {"id": "p2", "responses": [_response("c2", 1, "Yes."), _response("c1", 0, "No.")]},
            ]},
        },
    }
    (raw_dir / "200.json").write_text(json.dumps(activity))

    steps = [f"Activity 200, Part {p} Multi input submission" for p in ("p1", "p2")]
    step_df = pd.DataFrame(
        {
            "Problem Hierarchy": ["U1", "U1"],
            "Problem Name": ["Activity 200"] * 2,
            "Step Name": steps,
        }
    )
    questions = drivers["mfb"].extract_mfb(step_df, str(raw_dir))

    assert len(questions) == 2  # one question per blank
    by_id = {q["id"]: q for q in questions}
    sky = by_id["mfb-200_p2"]
    assert sky["question"]["stem"].strip() == "The sky is ____ and fire is *Red*"
    [ans_text] = [c["text"] for c in sky["question"]["choices"] if c["label"] == sky["answerKey"]]
    assert ans_text == "Blue"
    assert sky["objectives"] == ["Name colors"]

    fire = by_id["mfb-200_p1"]
    assert fire["question"]["stem"].strip() == "The sky is *Blue* and fire is ____"
    [ans_text] = [c["text"] for c in fire["question"]["choices"] if c["label"] == fire["answerKey"]]
    assert ans_text == "Red"
