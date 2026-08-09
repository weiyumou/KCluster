"""Golden tests pinning Question's prompt rendering to the EDM 2025 paper templates.

These assertions encode the exact strings shown in the paper's prompt tables;
any refactoring (e.g., extracting rendering into a prompt registry) must keep
them passing byte-for-byte.
"""

import pytest

from kcluster.core.question import Question


@pytest.fixture
def mcq() -> Question:
    # The example question from the EDM 2025 paper's prompt tables.
    return Question(
        {
            "id": "sciqa-0",
            "type": "Multiple Choice",
            "question": {
                "stem": "Which is the most flexible?",
                "choices": [
                    {"label": "a", "text": "paper"},
                    {"label": "b", "text": "ceramic tea cup"},
                    {"label": "c", "text": "clay tile"},
                ],
            },
            "answerKey": "a",
            "skill": ["identify properties of an object"],
        }
    )


def test_str_renders_body_and_answer(mcq):
    assert str(mcq) == (
        "Which is the most flexible?\n"
        "a) paper\n"
        "b) ceramic tea cup\n"
        "c) clay tile\n"
        "Answer: a"
    )


def test_str_without_answer_key_is_body_only(mcq):
    del mcq["answerKey"]
    assert str(mcq) == (
        "Which is the most flexible?\na) paper\nb) ceramic tea cup\nc) clay tile"
    )


def test_header_numbering_matches_paper_templates(mcq):
    # "Exercise 1:" introduces a conditioning question; "Exercise 2:" is the
    # marginal-prompt header (design decision D2).
    assert mcq.header(1) == "Exercise 1:\nMultiple Choice:"
    assert mcq.header(2) == "Exercise 2:\nMultiple Choice:"


def test_header_without_type_omits_type_line():
    q = Question({"question": {"stem": "What is 2 + 2?"}})
    assert q.header(1) == "Exercise 1:"


def test_prompt_is_header_body_trailer(mcq):
    assert mcq.prompt() == (
        "Exercise 1:\n"
        "Multiple Choice:\n"
        "Which is the most flexible?\n"
        "a) paper\n"
        "b) ceramic tea cup\n"
        "c) clay tile\n"
        "Answer:"
    )


def test_non_mcq_body_is_stem_only():
    q = Question(
        {
            "type": "Short Answer",
            "question": {"stem": "Name a flexible material."},
            "answerKey": "rubber",
        }
    )
    assert q.body == "Name a flexible material."
    assert str(q) == "Name a flexible material.\nAnswer: rubber"


def test_choices_render_for_any_choice_bearing_type(mcq):
    # Rendering keys on the presence of choices, not on the exact type string,
    # so select-all types render their choices too.
    mcq["type"] = "Multiple Choice (select all)"
    mcq["answerKey"] = "a, c"
    assert mcq.prompt() == (
        "Exercise 1:\n"
        "Multiple Choice (select all):\n"
        "Which is the most flexible?\n"
        "a) paper\n"
        "b) ceramic tea cup\n"
        "c) clay tile\n"
        "Answer:"
    )
    assert str(mcq).endswith("Answer: a, c")


def test_flat_dict_flattens_question_and_joins_string_lists(mcq):
    flat = mcq.flat_dict
    assert flat["question"] == str(mcq)
    assert flat["skill"] == "identify properties of an object"
    assert flat["id"] == "sciqa-0"
    assert flat["answerKey"] == "a"


def test_flat_dict_joins_multi_item_string_lists():
    q = Question({"question": {"stem": "s"}, "ds-step-name": ["step one", "step two"]})
    assert q.flat_dict["ds-step-name"] == "step one~step two"
