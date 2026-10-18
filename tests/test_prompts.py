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
