"""Tests for the KC tagger (kcluster.tasks.tag)."""

import pandas as pd
import pytest

from kcluster.tasks.tag import (
    SINGLE_KC_NAME,
    UNIQUE_STEP_NAME,
    add_opportunities,
    model_name,
    tag_student_step,
)


def _frame(**overrides) -> pd.DataFrame:
    """A minimal student-step frame that exercises the ordering rules.

    Row 1 (10:00) comes *after* row 0 (10:05) in the file, so practice order is
    not file order; rows 0 and 3 share student A's 10:05, so their tie breaks
    on file order. Row 4 is untagged by the expert model.
    """
    frame = pd.DataFrame({
        "Anon Student Id": ["A", "A", "B", "A", "A"],
        "Problem Name": ["s2", "s1", "s1", "s1", "s3"],
        "Step Name": ["s2", "s1", "s1", "s1", "s3"],
        "First Attempt": ["correct", "incorrect", "correct", "hint", "correct"],
        "First Transaction Time": ["2024-01-01 10:05:00", "2024-01-01 10:00:00", "2024-01-01 10:00:00",
                                   "2024-01-01 10:05:00", "2024-01-01 10:10:00"],
        "KC (Expert)": ["E1~~E2", "E1", "E1", "E1", ""],
    })
    for col, values in overrides.items():
        frame[col] = values
    return frame


def _kc_csv(rows: dict[str, str]) -> pd.DataFrame:
    """A flattened KC CSV: ``ds-step-name`` -> KC label."""
    return pd.DataFrame({"id": [f"q{i}" for i in range(len(rows))], "question": "what?",
                         "ds-step-name": list(rows), "KC": list(rows.values())})


def test_generated_model_is_joined_and_masked_to_expert_rows():
    tagged = tag_student_step(_frame(), {"concept": _kc_csv({"s1": "alpha", "s2": "beta", "s3": "gamma"})})
    # s3 only occurs on the expert-untagged row, so its label is blanked
    assert tagged["KC (concept)"].tolist() == ["beta", "alpha", "alpha", "alpha", ""]


def test_opportunities_follow_time_order_with_file_order_tiebreak():
    tagged = tag_student_step(_frame(), {"concept": _kc_csv({"s1": "alpha", "s2": "beta"})})
    # student A's practice order is row 1 (10:00), row 0 (10:05, first in
    # file), row 3 (10:05), row 4 (10:10); multi-KC counts align by position
    assert tagged["Opportunity (Expert)"].tolist() == ["2~~1", "1", "1", "3", ""]
    assert tagged["Opportunity (concept)"].tolist() == ["1", "1", "1", "2", ""]


def test_default_models_are_added_and_masked():
    tagged = tag_student_step(_frame(), {})
    assert tagged[f"KC ({SINGLE_KC_NAME})"].tolist() == ["Single-KC"] * 4 + [""]
    assert tagged[f"Opportunity ({SINGLE_KC_NAME})"].tolist() == ["2", "1", "1", "3", ""]
    # one KC per distinct Problem Hierarchy + Problem Name + Step Name value
    unique = tagged[f"KC ({UNIQUE_STEP_NAME})"]
    assert unique.tolist() == ["KC-1", "KC-2", "KC-2", "KC-2", ""]
    assert tagged[f"Opportunity ({UNIQUE_STEP_NAME})"].tolist() == ["1", "1", "1", "2", ""]


def test_every_kc_column_gets_an_adjacent_opportunity_column():
    tagged = tag_student_step(_frame(), {"concept": _kc_csv({"s1": "alpha", "s2": "beta"})})
    columns = tagged.columns.tolist()
    for name in ("Expert", "concept", SINGLE_KC_NAME, UNIQUE_STEP_NAME):
        assert columns.index(f"Opportunity ({name})") == columns.index(f"KC ({name})") + 1


def test_ds_step_name_key_join_expands_and_dedupes():
    # one multi-step question and one listing the same step twice ("s2~s2"):
    # the self-duplicate must not duplicate that step's rows
    kc = _kc_csv({"s1~s3": "alpha", "s2~s2": "beta"})
    tagged = tag_student_step(_frame(), {"concept": kc})
    assert len(tagged) == 5
    assert tagged["KC (concept)"].tolist() == ["beta", "alpha", "alpha", "alpha", ""]


def test_without_expert_columns_every_row_must_be_covered():
    frame = _frame().drop(columns="KC (Expert)")
    with pytest.raises(ValueError, match="match no question"):
        tag_student_step(frame, {"concept": _kc_csv({"s1": "alpha", "s2": "beta"})})
    tagged = tag_student_step(frame, {"concept": _kc_csv({"s1": "alpha", "s2": "beta", "s3": "gamma"})})
    assert tagged["KC (concept)"].tolist() == ["beta", "alpha", "alpha", "alpha", "gamma"]


def test_model_that_misses_an_expert_tagged_step_raises():
    with pytest.raises(ValueError, match="match no question"):
        tag_student_step(_frame(), {"concept": _kc_csv({"s1": "alpha"})})


def test_model_with_an_empty_label_raises():
    with pytest.raises(ValueError, match="without a label"):
        tag_student_step(_frame(), {"concept": _kc_csv({"s1": "alpha", "s2": ""})})


def test_ambiguous_step_to_question_mapping_raises():
    kc = pd.DataFrame({"id": ["q1", "q2"], "ds-step-name": ["s1~s2", "s2"], "KC": ["alpha", "beta"]})
    with pytest.raises(ValueError, match="more than one"):
        tag_student_step(_frame(), {"concept": kc})


def test_model_name_collision_raises():
    with pytest.raises(ValueError, match="collides"):
        tag_student_step(_frame(), {"Expert": _kc_csv({"s1": "alpha", "s2": "beta"})})


def test_unkeyed_kc_csv_raises():
    kc = pd.DataFrame({"id": ["q0"], "question": ["what?"], "KC": ["alpha"]})
    with pytest.raises(ValueError, match="lack 'ds-step-name'"):
        tag_student_step(_frame(), {"concept": kc})


def test_duplicate_kc_on_one_step_raises():
    with pytest.raises(ValueError, match="twice on one step"):
        add_opportunities(_frame(**{"KC (Expert)": ["E1~~E1", "E1", "E1", "E1", ""]}))


def test_model_name_strips_dataset_prefix_and_kc_suffix():
    assert model_name("kc/foundational-assist_concept-kc.csv") == "concept"
    assert model_name("elearning22-mcq_kcluster-unnorm-residfull-kc.csv") == "kcluster-unnorm-residfull"
    assert model_name("questions_sbert-cosine-kc.csv") == "sbert-cosine"
