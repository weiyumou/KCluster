"""Guard the shipped example data: it must stay loadable, synthetic-shaped,
and consistent with the example standards file."""

from collections import Counter
from pathlib import Path

import pytest

from kcluster.io.jsonl import load_questions

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "data"


def test_sample_questions_load_through_the_validated_reader():
    questions = load_questions(str(EXAMPLES / "sample-mcq.jsonl"))

    assert len(questions) == 15
    for q in questions:
        labels = [c["label"] for c in q["question"]["choices"]]
        assert q["answerKey"] in labels
        assert len(labels) == 4

    # Three planted topic clusters of five questions each
    assert sorted(Counter(q["lo"] for q in questions).values()) == [5, 5, 5]


def test_sample_los_match_the_example_standards():
    pytest.importorskip("torch")
    from kcluster.tasks.qgen.generate import read_standards

    questions = load_questions(str(EXAMPLES / "sample-mcq.jsonl"))
    standards = read_standards(str(EXAMPLES / "standards" / "actions" / "sample.txt"), "actions")

    assert set(q["lo"] for q in questions) == set(standards)
