import pandas as pd
import pytest

from kcluster.core.question import Question
from kcluster.io.student_step import (
    MINIMAL_SUFFIX,
    check_coverage,
    load_student_step,
    save_student_step,
    validate_student_step,
)


def _frame(**overrides) -> pd.DataFrame:
    data = {
        "Anon Student Id": ["s1", "s1", "s2"],
        "Problem Name": ["fa-1", "fa-2", "fa-1"],
        "Step Name": ["fa-1", "fa-2", "fa-1"],
        "First Attempt": ["correct", "incorrect", "hint"],
        "First Transaction Time": ["2019-08-25 22:52:54.873+00",
                                   "2019-08-26 02:07:21.189+00",
                                   "2019-09-05 00:54:03.194+00"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _q(qid: str, **extra) -> Question:
    # ds-step-name defaults to the id, which is what a non-DataShop driver
    # writes; DataShop-derived cases override it below.
    data = {"id": qid, "ds-step-name": qid,
            "type": "Short Answer", "question": {"stem": "?"}, "answerKey": "x"}
    data.update(extra)
    return Question(data)


# --- load / save ---

def test_round_trip_keeps_strings_and_empty_cells(tmp_path):
    df = _frame()
    df["KC (CCSS)"] = ["4.OA.A.1~~4.NF.B.4b", None, "4.OA.A.1"]
    path = str(tmp_path / f"toy{MINIMAL_SUFFIX}")
    save_student_step(df, path)
    loaded = load_student_step(path)
    assert loaded["KC (CCSS)"].tolist() == ["4.OA.A.1~~4.NF.B.4b", "", "4.OA.A.1"]
    assert pd.api.types.is_string_dtype(loaded["Anon Student Id"])
    assert loaded[df.columns[:-1]].equals(df[df.columns[:-1]])


# --- validate_student_step ---

def test_validate_accepts_minimal_frame_with_optional_columns():
    df = _frame()
    df["Problem Hierarchy"] = ["Unit A", "Unit A", "Unit B"]
    df["KC (CCSS)"] = ["4.OA.A.1", "", "4.OA.A.1~~4.NF.B.4b"]  # empty cells allowed
    df["KC (Teacher)"] = ["adding", "", "fractions"]  # same untagged rows as CCSS
    validate_student_step(df)


def test_expert_kc_columns_must_tag_the_same_rows():
    df = _frame()
    df["KC (CCSS)"] = ["4.OA.A.1", "", "4.NF.B.4b"]
    df["KC (Teacher)"] = ["adding", "multiplying", ""]
    with pytest.raises(ValueError, match=r"'KC \(CCSS\)' and 'KC \(Teacher\)' tag different rows \(2"):
        validate_student_step(df)


def test_missing_required_column_is_rejected():
    with pytest.raises(ValueError, match=r"missing required column\(s\).*First Attempt"):
        validate_student_step(_frame().drop(columns=["First Attempt"]))


def test_opportunity_columns_are_forbidden():
    df = _frame()
    df["Opportunity (CCSS)"] = ["1", "1", "2"]
    with pytest.raises(ValueError, match="tagger owns opportunity counting"):
        validate_student_step(df)


def test_empty_required_cell_is_rejected():
    df = _frame()
    df.loc[1, "Step Name"] = "  "
    with pytest.raises(ValueError, match="'Step Name' has 1 empty cell"):
        validate_student_step(df)


def test_first_attempt_vocabulary_is_exact():
    df = _frame(**{"First Attempt": ["correct", "Correct", "1"]})
    with pytest.raises(ValueError, match=r"unrecognized 'First Attempt' value\(s\) \['1', 'Correct'\]"):
        validate_student_step(df)


def test_time_text_order_must_match_time_order():
    # "2019-10-01" sorts before "2019-9-01" as text but is later in time
    df = _frame(**{"First Transaction Time": ["2019-9-01 00:00:00",
                                              "2019-10-01 00:00:00",
                                              "2019-9-02 00:00:00"]})
    with pytest.raises(ValueError, match="sorts differently as text than as time"):
        validate_student_step(df)


def test_mixed_timezone_offsets_are_rejected():
    # 20:52-05:00 is 01:52 UTC the next day: later than 22:52+00 in time,
    # earlier as text
    df = _frame(**{"First Transaction Time": ["2019-08-25 22:52:54+00",
                                              "2019-08-25 20:52:54-05",
                                              "2019-08-25 23:00:00+00"]})
    with pytest.raises(ValueError, match="sorts differently as text than as time"):
        validate_student_step(df)


def test_unparseable_time_is_rejected():
    df = _frame(**{"First Transaction Time": ["yesterday-ish", "2019-08-26 02:07:21+00",
                                              "2019-09-05 00:54:03+00"]})
    with pytest.raises(ValueError, match="unparseable 'First Transaction Time'"):
        validate_student_step(df)


# --- check_coverage, non-DataShop keying (ds-step-name = id <-> Step Name) ---

def test_question_without_the_key_field_raises():
    questions = [_q("fa-1"), Question({"id": "fa-2", "question": {"stem": "?"}})]
    with pytest.raises(ValueError, match="lack 'ds-step-name'.*'fa-2'"):
        check_coverage(questions, _frame())

def test_coverage_reports_questions_with_no_rows():
    questions = [_q("fa-1"), _q("fa-2"), _q("fa-3")]
    assert check_coverage(questions, _frame()) == ["fa-3"]


def test_coverage_returns_empty_when_every_question_has_rows():
    assert check_coverage([_q("fa-1"), _q("fa-2")], _frame()) == []


def test_unknown_step_raises():
    with pytest.raises(ValueError, match="match no question.*'fa-2'"):
        check_coverage([_q("fa-1")], _frame())


# --- check_coverage, DataShop keying (ds-* passthrough fields) ---

def _ds_frame() -> pd.DataFrame:
    return _frame(**{"Problem Name": ["quiz1", "quiz1", "quiz2"],
                     "Step Name": ["q1s1", "q1s2", "q2s1"]})


def test_coverage_expands_multi_step_questions():
    # One question spans two DataShop steps, joined by "~"
    questions = [_q("e-1", **{"ds-problem-name": "quiz1", "ds-step-name": "q1s1~q1s2"}),
                 _q("e-2", **{"ds-problem-name": "quiz2", "ds-step-name": "q2s1"})]
    assert check_coverage(questions, _ds_frame()) == []


def test_coverage_accepts_list_valued_keys():
    # Questions loaded from JSONL hold multiple steps as a list; flat_dict
    # "~"-joins them on the way into a KC CSV. Both shapes reach check_coverage.
    questions = [_q("e-1", **{"ds-problem-name": "quiz1", "ds-step-name": ["q1s1", "q1s2"]}),
                 _q("e-2", **{"ds-problem-name": ["quiz2"], "ds-step-name": ["q2s1"]})]
    assert check_coverage(questions, _ds_frame()) == []


def test_coverage_reports_uncovered_datashop_question():
    questions = [_q("e-1", **{"ds-problem-name": "quiz1", "ds-step-name": "q1s1~q1s2"}),
                 _q("e-2", **{"ds-problem-name": "quiz2", "ds-step-name": "q2s1"}),
                 _q("e-3", **{"ds-problem-name": "quiz3", "ds-step-name": "q3s1"})]
    assert check_coverage(questions, _ds_frame()) == ["e-3"]


def test_ambiguous_key_raises():
    questions = [_q("e-1", **{"ds-problem-name": "quiz1", "ds-step-name": "q1s1"}),
                 _q("e-1b", **{"ds-problem-name": "quiz1", "ds-step-name": "q1s1~q1s2"}),
                 _q("e-2", **{"ds-problem-name": "quiz2", "ds-step-name": "q2s1"})]
    with pytest.raises(ValueError, match="more than one question.*'e-1', 'e-1b'"):
        check_coverage(questions, _ds_frame())


def test_inconsistent_passthrough_fields_raise():
    # ds-problem-name on one question but not the other leaves the key ill-defined
    questions = [_q("e-1", **{"ds-problem-name": "quiz1", "ds-step-name": "q1s1"}),
                 _q("e-2", **{"ds-step-name": "q2s1"})]
    with pytest.raises(ValueError, match="present on 1 of 2 questions"):
        check_coverage(questions, _ds_frame())


def test_ds_fields_without_matching_column_are_dropped_from_key():
    # Questions carry ds-problem-hierarchy but the frame has no Problem
    # Hierarchy column; the key falls back to the remaining fields.
    questions = [_q("e-1", **{"ds-problem-hierarchy": "Unit A",
                              "ds-problem-name": "quiz1", "ds-step-name": "q1s1~q1s2"}),
                 _q("e-2", **{"ds-problem-hierarchy": "Unit B",
                              "ds-problem-name": "quiz2", "ds-step-name": "q2s1"})]
    assert check_coverage(questions, _ds_frame()) == []
