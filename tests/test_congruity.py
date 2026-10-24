"""Golden tests for the congruity scoring grid.

The prompt strings assert the exact templates from the EDM 2025 paper
(conditional: Table 3; marginal: Table 4), using the paper's own example
questions. The index-layout tests pin the row-as-conditioning orientation
that the canonical PointwiseMutualInfo relies on.
"""

import pytest

pytest.importorskip("torch")

from kcluster.core.question import Question
from kcluster.tasks.congruity import PairQuestion


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
    context, text, context_figures, text_figures = PairQuestion(questions)[1]
    assert context == "Exercise 2:\nMultiple Choice:\n"
    assert text == str(questions[1])
    assert context_figures == [] and text_figures == []


def test_conditional_matches_paper_template(questions):
    # Grid item n + 0*n + 1 scores q_star conditioned on q_plus — the paper's
    # worked example of a congruent pair.
    context, text, _, _ = PairQuestion(questions)[3 + 0 * 3 + 1]
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
            context, text, _, _ = ds[n + i * n + j]
            # Context carries question i (the conditioning variable) ...
            assert context.startswith(f"{questions[i].header(1)}\n{questions[i]}")
            # ... and the scored text is question j.
            assert text == str(questions[j])


def test_items_carry_each_sides_figures_under_the_root(questions):
    questions[0]["question"]["stem"] = "[figure]\n" + questions[0].stem
    questions[0]["figures"] = ["figs/plus.png"]
    ds = PairQuestion(questions, figure_dir="/data")
    # Marginal of q_plus: its figure is in the scored text (never scored itself)
    _, _, context_figures, text_figures = ds[0]
    assert (context_figures, text_figures) == ([], ["/data/figs/plus.png"])
    # q_star given q_plus: the figure is in the context; q_plus given q_star: in the text
    assert ds[3 + 0 * 3 + 1][2:] == (["/data/figs/plus.png"], [])
    assert ds[3 + 1 * 3 + 0][2:] == ([], ["/data/figs/plus.png"])
    # Without a root the markers stay literal text and no figure is shown
    assert PairQuestion(questions)[0][2:] == ([], [])
    assert "[figure]" in PairQuestion(questions)[0][1]
