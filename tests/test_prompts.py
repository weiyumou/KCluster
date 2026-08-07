"""Golden tests for the prompt registry.

Every assertion here is an exact published string (EDM 2025 / LAK 2026).
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


def test_lo_alignment_scaffolds():
    assert prompts.LO_ALIGNMENT_MARGINAL_CONTEXT == "{q_type}:\n"
    assert prompts.LO_ACTIONS_HEADER == "The exercise below is designed to test whether a student can {lo}."
    assert prompts.LO_FACTS_HEADER == "The exercise below is designed to test whether a student knows:\n{lo}."


def test_qgen_scaffolds():
    assert prompts.QGEN_SEED_ACTIONS == (
        "The exercises below are designed to test whether a student can {std}.\n\n{header}")
    assert prompts.QGEN_SEED_FACTS == (
        'The exercises below are designed to test whether a student understands the following facts:\n'
        '"{std}."\n\n{header}')
    assert prompts.QGEN_MCQ_HEADER == "Multiple Choice (best out of {num_choices} options):\n1."
    assert prompts.QGEN_SOLUTION_PREFIX == "\n\nSolution:\nThe correct answer is"
    assert prompts.QGEN_EXPLANATION_PREFIX == "\n\nExplanation:\n"


def test_judge_wordings():
    assert prompts.GPT_JUDGE_SYSTEM_PROMPT == (
        "You are an expert at answering multiple choice questions. "
        "If none of the options a-d are correct, choose e for 'None of the above'. "
        "Provide your answer (letter a-e) and explanation in the JSON format specified."
    )
    assert prompts.JUDGE_Q1_PAIRED == "Answer the following two questions:\n\nQ1. {question}"
    assert prompts.JUDGE_Q1_SINGLE == "Answer the following question:\n\n{question}"
    assert prompts.JUDGE_Q2_LOGPROB == (
        "Does the above question help teachers test whether a student can {lo}?\na) Yes\nb) No")
    assert prompts.JUDGE_Q2_TEXT == "Does the following question test whether a student can **{lo}**?\n\n{question}"
    assert prompts.JUDGE_PREFILL_Q1 == "The answer to Q1 is **"
    assert prompts.JUDGE_PREFILL_Q2 == "The answer to Q2 is **"


def test_prompt_version_is_declared():
    assert prompts.PROMPT_VERSION == 1
