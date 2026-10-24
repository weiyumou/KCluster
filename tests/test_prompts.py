"""Golden tests for the prompt registry.

Every assertion here is an exact published string (EDM 2025).
If one of these fails, a prompt changed — that is a scientific decision
requiring a PROMPT_VERSION bump, not a wording cleanup.
"""

from kcluster.core import prompts
from kcluster.core.question import Question


def _mcq():
    return Question(
        {
            "id": "q-0",
            "type": "Multiple Choice",
            "question": {"stem": "Which is the most flexible?",
                         "choices": [{"label": "a", "text": "paper"}, {"label": "b", "text": "clay tile"}]},
            "answerKey": "a",
        }
    )


def test_question_rendering_templates():
    assert prompts.EXERCISE_HEADER == "Exercise {q_num}:"
    assert prompts.QUESTION_TYPE_LINE == "{q_type}:"
    assert prompts.CHOICE_LINE == "{label}) {text}"
    assert prompts.ANSWER_TRAILER == "Answer:"


def test_congruity_contexts_match_the_paper_grid():
    q1, q2 = _mcq(), _mcq()
    assert prompts.congruity_marginal_context(q1) == "Exercise 2:\nMultiple Choice:\n"
    assert prompts.congruity_pair_context(q1, q2) == (
        "Exercise 1:\nMultiple Choice:\n"
        "Which is the most flexible?\na) paper\nb) clay tile\nAnswer: a"
        "\n\n"
        "Exercise 2:\nMultiple Choice:\n"
    )


def test_concept_prompt_noun_and_verbal_variants():
    q = _mcq()
    assert prompts.concept_prompt(q) == (
        "Exercise 1:\nMultiple Choice:\n"
        "Which is the most flexible?\na) paper\nb) clay tile\nAnswer: a"
        "\n\nRemark:\nThe above exercise is a multiple-choice question "
        "that tests whether the student understands the concept of"
    )
    assert prompts.concept_prompt(q, verbal=True).endswith("that tests whether the student can")


def test_concept_decoding_is_shared_by_both_engines():
    # The local concept step and the Vertex job both read these; a concept is
    # the completion up to its first period or comma, cleaned of that token.
    assert prompts.CONCEPT_STOP_TOKENS == (".", ",")
    assert (prompts.CONCEPT_NUM_BEAMS, prompts.CONCEPT_LENGTH_PENALTY) == (5, -0.1)
    assert prompts.clean_concept(" flexibility. ") == "flexibility"
    assert prompts.clean_concept("states of matter,") == "states of matter"
    assert prompts.clean_concept(" Ohm's law") == "Ohm's law"


# --- Prompt styles: the stock strings above, and the same requests written for a chat model ---

def _render(messages) -> str:
    """A stand-in chat template: one line per turn, then the opened assistant turn."""
    return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"


def test_the_stock_style_writes_the_published_prompts():
    q1, q2 = _mcq(), _mcq()
    assert prompts.STOCK.concept(q1) == prompts.concept_prompt(q1)
    assert prompts.STOCK.concept(q1, verbal=True) == prompts.concept_prompt(q1, verbal=True)
    assert prompts.STOCK.pair(q1, q2) == prompts.congruity_pair_context(q1, q2)
    assert prompts.STOCK.marginal(q2) == prompts.congruity_marginal_context(q2)
    # The concept marginal is the concept prompt with the question left out
    assert prompts.STOCK.concept_marginal(q1) == (
        "Exercise 1:\nMultiple Choice:\n"
        "\nRemark:\nThe above exercise is a multiple-choice question "
        "that tests whether the student understands the concept of"
    )


def test_a_style_can_change_the_remark_a_name_completes():
    style = prompts.StockPrompts(remark="The above exercise tests {trailer}")
    assert style.concept(_mcq()).endswith("\n\nRemark:\nThe above exercise tests whether the student understands the concept of")
    assert style.concept(_mcq(), verbal=True).endswith("Remark:\nThe above exercise tests whether the student can")
    assert style.concept_marginal(_mcq()) == (
        "Exercise 1:\nMultiple Choice:\n\nRemark:\nThe above exercise tests whether the student understands the concept of"
    )


def test_the_chat_style_puts_the_requests_in_turns():
    style = prompts.ChatPrompts(_render, concept_request="What does it test?", first_request="Write an exercise.",
                                second_request="Write another one.")
    q1, q2 = _mcq(), _mcq()
    exercise = "Exercise 1:\nMultiple Choice:\nWhich is the most flexible?\na) paper\nb) clay tile\nAnswer: a"
    assert style.concept(q1) == (
        f"<user>{exercise}\n\nWhat does it test?</user><assistant>"
        "The above exercise is a multiple-choice question that tests whether the student understands the concept of"
    )
    assert style.concept(q1, verbal=True).endswith("<assistant>The above exercise is a multiple-choice question "
                                                   "that tests whether the student can")
    assert style.concept_marginal(q1) == (
        "<user>Exercise 1:\nMultiple Choice:\n\nWhat does it test?</user><assistant>"
        "The above exercise is a multiple-choice question that tests whether the student understands the concept of"
    )
    assert style.pair(q1, q2) == (
        f"<user>Write an exercise.</user><assistant>{q1}</assistant><user>Write another one.</user><assistant>"
    )
    assert style.marginal(q2) == "<user>Write an exercise.</user><assistant>"
    # The remark is a parameter here too
    plain = prompts.ChatPrompts(_render, concept_request="?", first_request="", second_request="",
                                remark="The above exercise tests {trailer}")
    assert plain.concept(q1).endswith("<assistant>The above exercise tests whether the student understands the concept of")
