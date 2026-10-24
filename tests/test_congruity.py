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


def test_a_prompt_style_writes_the_contexts(questions):
    from kcluster.core.prompts import ChatPrompts

    def render(messages):
        return " | ".join(f"{m['role']}: {m['content']}" for m in messages) + " | assistant:"

    style = ChatPrompts(render, concept_request="?", first_request="Write an exercise.",
                        second_request="Write another one.")
    ds, n = PairQuestion(questions, style), len(questions)
    assert ds[1] == ("user: Write an exercise. | assistant:", str(questions[1]))
    assert ds[n + 0 * n + 1] == (
        f"user: Write an exercise. | assistant: {questions[0]} | user: Write another one. | assistant:",
        str(questions[1]),
    )


def test_a_grid_of_other_texts_shares_the_layout(questions):
    from kcluster.tasks.congruity import ScoringGrid

    class NameGrid(ScoringGrid):
        """Concept names after each question, and after no question: n rows, m columns."""

        def __init__(self, questions, names):
            super().__init__([f" {name}." for name in names], len(questions))
            self.questions = questions

        def marginal(self, col):
            return "Remark:\nThe above exercise tests"

        def context(self, row, col):
            return f"{self.questions[row]}\n\nRemark:\nThe above exercise tests"

    grid = NameGrid(questions, ["flexibility", "addition"])
    assert len(grid) == (3 + 1) * 2
    assert grid[1] == ("Remark:\nThe above exercise tests", " addition.")
    assert grid[2 + 2 * 2 + 0] == (f"{questions[2]}\n\nRemark:\nThe above exercise tests", " flexibility.")

    # PairQuestion is the square case, and its parts read the same way
    ds = PairQuestion(questions)
    assert (ds.n_rows, ds.texts) == (3, [str(q) for q in questions])
    assert ds[3 + 1 * 3 + 2] == (ds.context(1, 2), ds.texts[2])


def test_longest_input_takes_the_longest_row_with_the_longest_text(questions):
    class WordTokenizer:
        def __call__(self, text, text_pair=None):
            return {"input_ids": (text + " " + (text_pair or "")).split()}

    ds = PairQuestion(questions)
    tokenizer = WordTokenizer()
    longest = max(len(tokenizer(*ds[i])["input_ids"]) for i in range(len(ds)))
    assert ds.longest_input(tokenizer) == longest
