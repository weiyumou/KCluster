"""Tests for KC-coverage MCQ selection and the standards report."""

import numpy as np
import pandas as pd

from kcluster.core.question import Question
from kcluster.tasks.qgen.select import build_report, select_mcq


def _questions(n=4):
    return [
        Question(
            {
                "id": f"q-{i}",
                "type": "Multiple Choice",
                "question": {"stem": f"Stem {i}?",
                             "choices": [{"label": "a", "text": "Right"},
                                         {"label": "b", "text": "Wrong one"},
                                         {"label": "c", "text": "Wrong two"}]},
                "answerKey": "a",
                "explanation": "Because of reasons.",
                "lo": "identify the states of matter",
                "kc": f"kc-{i % 2}",
            }
        )
        for i in range(n)
    ]


def _kc_df():
    # Two clusters with exemplars at indices 0 and 2
    return pd.DataFrame({"KC-raw": ["KC-0", "KC-0", "KC-2", "KC-2"]})


def test_select_mcq_takes_all_exemplars_first():
    questions = _questions()
    sel = select_mcq(_kc_df(), questions, mcq_per_lo=2)
    assert [q["id"] for q in sel] == ["q-0", "q-2"]


def test_select_mcq_tops_up_with_non_exemplars():
    questions = _questions()
    sel = select_mcq(_kc_df(), questions, mcq_per_lo=3, rng=np.random.default_rng(42))
    ids = [q["id"] for q in sel]
    assert ids[:2] == ["q-0", "q-2"]
    assert len(ids) == 3
    assert ids[2] in {"q-1", "q-3"}  # sampled from the non-exemplars only


def test_build_report():
    [q] = _questions(1)
    report = build_report([q], {"Identify the states of matter": "STD-7"})

    [row] = report.to_dict(orient="records")
    assert row["Standard Code"] == "STD-7"
    assert row["Standard Text"] == "Identify the states of matter."
    assert (row["Answer"], row["Key"]) == ("Right", "a")
    assert (row["Distractor 1"], row["Distractor 2"]) == ("Wrong one", "Wrong two")
    assert (row["Explanation"], row["KC"]) == ("Because of reasons.", "kc-0")
