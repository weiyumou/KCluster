"""Tests for LO-alignment: the QuestionLO scoring grid and the offline
top-k aggregation over planted score shards."""

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from kcluster.core.question import Question  # noqa: E402
from kcluster.io.jsonl import dump_questions, load_questions  # noqa: E402
from kcluster.tasks.classify import QuestionLO, classify_from_pmi  # noqa: E402

LOS = ["Explain how levers work.", "Identify the states of matter."]


def _questions(n=3) -> list[Question]:
    return [
        Question(
            {
                "id": f"q-{i}",
                "type": "Multiple Choice",
                "question": {"stem": f"Stem {i}?", "choices": [{"label": "a", "text": "x"}]},
                "answerKey": "a",
                "lo": LOS[0] if i < 2 else LOS[1],  # q-0/q-1 belong to lo 0; q-2 to lo 1
            }
        )
        for i in range(n)
    ]


def test_question_lo_actions_grid():
    questions = _questions(2)
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


def test_question_lo_facts_grid():
    questions = _questions(1)
    ds = QuestionLO(questions, ["Water boils at 100 C.."], "facts")
    context, _ = ds[1]
    # Facts keep their capitalization and gain a line break; trailing periods collapse to one
    assert context == ("The exercise below is designed to test whether a student knows:\n"
                       "Water boils at 100 C.\n\nMultiple Choice:\n")


def test_question_lo_rejects_unknown_type():
    with pytest.raises(ValueError, match="lo_type"):
        QuestionLO(_questions(1), LOS, "verbs")


def test_classify_from_pmi(tmp_path):
    questions = _questions(3)
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
    flat = np.concatenate([marginals, conds.ravel()])
    torch.save([[torch.arange(len(flat))]], run_dir / "batch_indices_0.pt")
    torch.save([torch.tensor(flat, dtype=torch.float32)], run_dir / "predictions_0.pt")

    res_df = classify_from_pmi(str(run_dir), topk=1)

    assert res_df["pred_lo_1"].tolist() == [LOS[0], LOS[0], LOS[1]]
    # Every question's true LO is its top-1 here, so all three are "matched",
    # and the output is canonical JSONL (the legacy writer emitted repr)
    matched = load_questions(str(run_dir / "matched-top1.jsonl"))
    assert [q["id"] for q in matched] == ["q-0", "q-1", "q-2"]

    saved = pd.read_csv(run_dir / "classified-top1.csv")
    assert saved["pred_lo_1"].tolist() == [LOS[0], LOS[0], LOS[1]]


def test_classify_from_pmi_records_misses(tmp_path):
    # Flip the planted alignment for q-2 so its true LO is not predicted
    questions = _questions(3)
    data_path = tmp_path / "questions.jsonl"
    dump_questions(questions, str(data_path))

    run_dir = tmp_path / "classify"
    run_dir.mkdir()
    (run_dir / "args-pmi-questions.json").write_text(json.dumps({"data_path": str(data_path)}))
    (run_dir / "los-questions.json").write_text(json.dumps(LOS))

    marginals = np.full(3, -50.0)
    conds = np.array([[-45.0, -45.0, -45.0],
                      [-55.0, -55.0, -55.0]])
    flat = np.concatenate([marginals, conds.ravel()])
    torch.save([[torch.arange(len(flat))]], run_dir / "batch_indices_0.pt")
    torch.save([torch.tensor(flat, dtype=torch.float32)], run_dir / "predictions_0.pt")

    classify_from_pmi(str(run_dir), topk=1)
    matched = load_questions(str(run_dir / "matched-top1.jsonl"))
    assert [q["id"] for q in matched] == ["q-0", "q-1"]  # q-2's true LO was not in the top-1
