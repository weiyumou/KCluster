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

    style = ChatPrompts(concept_request="?", first_request="Write an exercise.", second_request="Write another one.")
    ds, n = PairQuestion(questions, style), len(questions)
    # A chat style's contexts are conversations, which the engine writes in the model's template
    assert ds[1] == ([{"role": "user", "content": "Write an exercise."}], str(questions[1]))
    assert ds[n + 0 * n + 1] == (
        [{"role": "user", "content": "Write an exercise."}, {"role": "assistant", "content": str(questions[0])},
         {"role": "user", "content": "Write another one."}],
        str(questions[1]),
    )

    def render(messages):
        return " | ".join(f"{m['role']}: {m['content']}" for m in messages) + " | assistant:"

    rendered = PairQuestion(questions, ChatPrompts(render, concept_request="?", first_request="Write an exercise.",
                                                   second_request="Write another one."))
    assert rendered[1] == ("user: Write an exercise. | assistant:", str(questions[1]))


def test_a_grid_of_other_texts_shares_the_layout(questions):
    from kcluster.core.grid import ScoringGrid

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
    def count(context, text):
        return len((context + " " + text).split())

    ds = PairQuestion(questions)
    longest = max(count(*ds[i]) for i in range(len(ds)))
    assert ds.longest_input(count) == longest


def test_items_carry_each_sides_figures_at_the_models_image_token(questions):
    questions[0]["question"]["stem"] = "[figure]\n" + questions[0].stem
    questions[0]["figures"] = ["figs/plus.png"]
    ds = PairQuestion(questions, figure_dir="/data", image_token="<|image|>")
    assert ds.shows_images and len(ds[0]) == 4
    # Marginal of q_plus: its figure is in the scored text, at the model's token (its tokens are never scored)
    context, text, context_figures, text_figures = ds[0]
    assert text.startswith("<|image|>\nWhich") and "[figure]" not in text
    assert (context_figures, text_figures) == ([], ["/data/figs/plus.png"])
    # q_star given q_plus: the figure is in the context; q_plus given q_star: in the text
    context, text, context_figures, text_figures = ds[3 + 0 * 3 + 1]
    assert "<|image|>" in context and (context_figures, text_figures) == (["/data/figs/plus.png"], [])
    assert ds[3 + 1 * 3 + 0][2:] == ([], ["/data/figs/plus.png"])
    # Without a root the markers stay literal text, no figure is shown and the items are the pairs
    plain = PairQuestion(questions)
    assert not plain.shows_images and len(plain[0]) == 2 and "[figure]" in plain[0][1]
    with pytest.raises(ValueError, match="image token"):
        PairQuestion(questions, figure_dir="/data")
