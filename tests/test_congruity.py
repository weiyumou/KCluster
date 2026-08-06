"""Golden tests for the congruity scoring grid.

The prompt strings assert the exact templates from the EDM 2025 paper
(conditional: Table 3; marginal: Table 4), using the paper's own example
questions. The index-layout tests pin the row-as-conditioning orientation
that the canonical PointwiseMutualInfo relies on.
"""

import pytest

pytest.importorskip("torch")

from kcluster.core.question import Question  # noqa: E402
from kcluster.tasks.congruity import PairQuestion  # noqa: E402


def _mcq(stem: str, choices: list[str], answer: str) -> Question:
    labels = "abcdefghijklmnopqrstuvwxyz"
    return Question(
        {
            "type": "Multiple Choice",
            "question": {
                "stem": stem,
                "choices": [
                    {"label": labels[i], "text": text} for i, text in enumerate(choices)
                ],
            },
            "answerKey": answer,
        }
    )


@pytest.fixture
def questions() -> list[Question]:
    # q_plus and q_star are the example pair from the EDM 2025 paper.
    q_plus = _mcq("Which is the most flexible?", ["bone", "glass jar", "rubber band"], "c")
    q_star = _mcq("Which is the most flexible?", ["paper", "ceramic tea cup", "clay tile"], "a")
    q_other = _mcq("What is 2 + 2?", ["3", "4", "5"], "b")
    return [q_plus, q_star, q_other]


def test_len_is_conditional_grid_plus_marginals(questions):
    assert len(PairQuestion(questions)) == 3 * 3 + 3


def test_marginal_uses_bare_exercise_2_header(questions):
    # The marginal template from the paper: "Exercise 2:" alone as context.
    context, text = PairQuestion(questions)[1]
    assert context == "Exercise 2:\nMultiple Choice:\n"
    assert text == str(questions[1])


def test_conditional_matches_paper_template(questions):
    # Grid item n + 0*n + 1 scores q_star conditioned on q_plus — the paper's
    # worked example of a congruent pair.
    context, text = PairQuestion(questions)[3 + 0 * 3 + 1]
    assert context == (
        "Exercise 1:\n"
        "Multiple Choice:\n"
        "Which is the most flexible?\n"
        "a) bone\n"
        "b) glass jar\n"
        "c) rubber band\n"
        "Answer: c\n"
        "\n"
        "Exercise 2:\n"
        "Multiple Choice:\n"
    )
    assert text == str(questions[1])


def test_grid_is_row_major_with_row_as_conditioning(questions):
    ds, n = PairQuestion(questions), len(questions)
    for i in range(n):
        for j in range(n):
            context, text = ds[n + i * n + j]
            # Context carries question i (the conditioning variable) ...
            assert context.startswith(f"{questions[i].header(1)}\n{questions[i]}")
            # ... and the scored text is question j.
            assert text == str(questions[j])
