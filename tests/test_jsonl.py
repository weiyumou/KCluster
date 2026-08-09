import json

import pytest

from kcluster.core.question import Question
from kcluster.io.jsonl import dump_questions, load_questions, validate_question


def _mcq_dict(qid: str = "q-1") -> dict:
    return {
        "id": qid,
        "type": "Multiple Choice",
        "question": {
            "stem": "Which is the most flexible?",
            "choices": [{"label": "a", "text": "paper"}, {"label": "b", "text": "clay tile"}],
        },
        "answerKey": "a",
        "lo": "identify properties of an object",
    }


def test_dump_then_load_round_trips(tmp_path):
    path = str(tmp_path / "questions.jsonl")
    dump_questions([Question(_mcq_dict("q-1")), Question(_mcq_dict("q-2"))], path)
    loaded = load_questions(path)
    assert [q["id"] for q in loaded] == ["q-1", "q-2"]
    assert loaded[0].data == _mcq_dict("q-1")


def test_blank_lines_are_tolerated(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(_mcq_dict()) + "\n\n")
    assert len(load_questions(str(path))) == 1


def test_missing_required_field_reports_path_and_line(tmp_path):
    bad = _mcq_dict()
    del bad["answerKey"]
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(_mcq_dict()) + "\n" + json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2: .*answerKey"):
        load_questions(str(path))


def test_validate_false_skips_schema_checks(tmp_path):
    path = tmp_path / "partial.jsonl"
    path.write_text(json.dumps({"id": "q-1"}) + "\n")
    assert load_questions(str(path), validate=False)[0]["id"] == "q-1"


def test_mcq_without_choices_is_rejected():
    bad = _mcq_dict()
    bad["question"].pop("choices")
    with pytest.raises(ValueError, match="no choices"):
        validate_question(Question(bad))


def test_select_all_without_choices_is_rejected():
    bad = _mcq_dict()
    bad["type"] = "Multiple Choice (select all)"
    bad["question"].pop("choices")
    with pytest.raises(ValueError, match="no choices"):
        validate_question(Question(bad))


def test_choiceless_type_without_choices_is_accepted():
    q = {"id": "q-1", "type": "Fill-in-the-blank(s)",
         "question": {"stem": "28 is 7 times what number? ____"}, "answerKey": "4"}
    validate_question(Question(q))


def test_malformed_choice_is_rejected():
    bad = _mcq_dict()
    bad["question"]["choices"].append({"label": "c"})
    with pytest.raises(ValueError, match="malformed choice"):
        validate_question(Question(bad))


def test_invalid_json_reports_line_number(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text("{'single': 'quotes'}\n")
    with pytest.raises(ValueError, match=r"broken\.jsonl:1: not a valid JSON line"):
        load_questions(str(path))


def test_non_object_line_is_rejected(tmp_path):
    path = tmp_path / "list.jsonl"
    path.write_text("[1, 2, 3]\n")
    with pytest.raises(ValueError, match="expected an object per line"):
        load_questions(str(path))


def test_repr_format_lines_are_rejected(tmp_path):
    # The retired legacy repr format (single-quoted Python dicts) must not
    # silently load as JSON.
    path = tmp_path / "legacy.jsonl"
    path.write_text(repr(_mcq_dict()) + "\n")
    with pytest.raises(ValueError, match="not a valid JSON line"):
        load_questions(str(path))
