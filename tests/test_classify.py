"""Tests for LO-alignment: the QuestionLO scoring grid and the offline
top-k aggregation over planted score shards."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from kcluster.io.jsonl import dump_questions, load_questions
from kcluster.tasks.classify import QuestionLO, classify_from_pmi

LOS = ["Explain how levers work.", "Identify the states of matter."]


@pytest.fixture
def lo_questions(make_questions):
    """Factory for a bank whose q-0/q-1 belong to lo 0 and q-2 to lo 1."""
    return lambda n=3: make_questions(n, extra=lambda i: {"lo": LOS[0] if i < 2 else LOS[1]})


def test_question_lo_actions_grid(lo_questions):
    questions = lo_questions(2)
    ds = QuestionLO(questions, LOS, "actions")

    assert len(ds) == 2 * 2 + 2
    # Marginals: the question under its bare type header
    context, text = ds[0]
    assert context == "Multiple Choice:\n"
    assert text == str(questions[0])
    # Grid is LO-major; the action LO is decapitalized and stripped of its period
    context, text = ds[2 + 1 * 2 + 0]  # lo 1, question 0
    assert context == ("The exercise below is designed to test whether a student can "
                       "identify the states of matter.\n\nMultiple Choice:\n")
    assert text == str(questions[0])


def test_question_lo_facts_grid(lo_questions):
    questions = lo_questions(1)
    ds = QuestionLO(questions, ["Water boils at 100 C.."], "facts")
    context, _ = ds[1]
    # Facts keep their capitalization and gain a line break; trailing periods collapse to one
    assert context == ("The exercise below is designed to test whether a student knows:\n"
                       "Water boils at 100 C.\n\nMultiple Choice:\n")


def test_question_lo_rejects_unknown_type(lo_questions):
    with pytest.raises(ValueError, match="lo_type"):
        QuestionLO(lo_questions(1), LOS, "verbs")


def test_classify_from_pmi(tmp_path, lo_questions, write_shards):
    questions = lo_questions(3)
    data_path = tmp_path / "questions.jsonl"
    dump_questions(questions, str(data_path))

    run_dir = tmp_path / "classify"
    run_dir.mkdir()
    (run_dir / "args-pmi-questions.json").write_text(json.dumps({"data_path": str(data_path)}))
    (run_dir / "los-questions.json").write_text(json.dumps(LOS))

    # Shards: rows = LOs (conditioning), cols = questions. Plant the true
    # alignment 5 nats above the marginal, the wrong one 5 below.
    marginals = np.full(3, -50.0)
    conds = np.array([[-45.0, -45.0, -55.0],
                      [-55.0, -55.0, -45.0]])
    write_shards(run_dir, np.vstack([marginals, conds]))

    res_df = classify_from_pmi(str(run_dir), topk=1)

    assert res_df["pred_lo_1"].tolist() == [LOS[0], LOS[0], LOS[1]]
    # Every question's true LO is its top-1 here, so all three are "matched",
    # and the output is canonical JSONL (the legacy writer emitted repr)
    matched = load_questions(str(run_dir / "matched-top1.jsonl"))
    assert [q["id"] for q in matched] == ["q-0", "q-1", "q-2"]

    saved = pd.read_csv(run_dir / "classified-top1.csv")
    assert saved["pred_lo_1"].tolist() == [LOS[0], LOS[0], LOS[1]]


def test_classify_from_pmi_records_misses(tmp_path, lo_questions, write_shards):
    # Flip the planted alignment for q-2 so its true LO is not predicted
    questions = lo_questions(3)
    data_path = tmp_path / "questions.jsonl"
    dump_questions(questions, str(data_path))

    run_dir = tmp_path / "classify"
    run_dir.mkdir()
    (run_dir / "args-pmi-questions.json").write_text(json.dumps({"data_path": str(data_path)}))
    (run_dir / "los-questions.json").write_text(json.dumps(LOS))

    marginals = np.full(3, -50.0)
    conds = np.array([[-45.0, -45.0, -45.0],
                      [-55.0, -55.0, -55.0]])
    write_shards(run_dir, np.vstack([marginals, conds]))

    classify_from_pmi(str(run_dir), topk=1)
    matched = load_questions(str(run_dir / "matched-top1.jsonl"))
    assert [q["id"] for q in matched] == ["q-0", "q-1"]  # q-2's true LO was not in the top-1
