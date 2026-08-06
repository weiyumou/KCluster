import json
import re

import pandas as pd
import pytest

from kcluster.io.datashop import (
    KC_PAT,
    create_datashop_kc,
    create_default_kc,
    create_kc_from_questions,
    load_datashop_temp,
    merge_student_step_with_kc,
)


def _write_tsv(df: pd.DataFrame, path) -> str:
    df.to_csv(path, sep="\t", index=False)
    return str(path)


@pytest.fixture
def template(tmp_path) -> str:
    return _write_tsv(
        pd.DataFrame(
            {
                "Step ID": [1, 2, 3],
                "Problem Hierarchy": ["(Unit A)", "(Unit A)", "(Unit B)"],
                "Problem Name": ["P1", "P1", "P2"],
                "Step Name": ["s1", "s2", "s3"],
                "KC (model name)": [None, None, None],
            }
        ),
        tmp_path / "empty-template.txt",
    )


def test_kc_pat_strips_cv_replica_suffix_but_keeps_hyphenated_names():
    assert re.match(KC_PAT, "KC (KCluster-3)").group("name") == "KCluster"
    assert re.match(KC_PAT, "KC (LOs-new)").group("name") == "LOs-new"


def test_load_datashop_temp_drops_empty_columns_and_fills_keys(template):
    df = load_datashop_temp(template)
    assert "KC (model name)" not in df.columns  # all-NaN placeholder dropped
    assert (df["Problem Hierarchy"] != "").all()


def test_create_default_kc_builds_single_and_unique_step(template):
    df = create_default_kc(template, "Single-KC", "Unique-step")
    assert (df["KC (Single-KC)"] == "Single-KC").all()
    assert df["KC (Unique-step)"].tolist() == ["KC-1", "KC-2", "KC-3"]


def test_create_datashop_kc_explodes_tilde_joined_steps(template):
    # At most one key column is ever multi-valued in real data (the ~-joined
    # step list of a merged question); parallel multi-valued keys would
    # cross-product under the sequential explode.
    kc = pd.DataFrame(
        {
            "ds-problem-name": ["P1", "P2"],
            "ds-step-name": ["s1~s2", "s3"],
            "concept": ["alpha", "beta"],
        }
    )
    out = create_datashop_kc(kc, template, kc_cols=["concept"], new_kc_names=["MyModel"])
    assert out["KC (MyModel)"].tolist() == ["alpha", "alpha", "beta"]


def test_create_datashop_kc_match_other_kc_masks_uncovered_steps(tmp_path):
    template = _write_tsv(
        pd.DataFrame(
            {
                "Problem Hierarchy": ["Unit A", "Unit A", "Unit B"],
                "Problem Name": ["P1", "P1", "P2"],
                "Step Name": ["s1", "s2", "s3"],
                "KC (expert)": ["e1", "e2", None],  # expert model does not cover s3
            }
        ),
        tmp_path / "expert-template.txt",
    )
    kc = pd.DataFrame({"ds-step-name": ["s1", "s2", "s3"], "concept": ["a", "b", "c"]})
    out = create_datashop_kc(kc, template, kc_cols=["concept"], match_other_kc=True)
    assert out["KC (concept)"].tolist()[:2] == ["a", "b"]
    assert pd.isna(out["KC (concept)"].iloc[2])


def test_create_kc_from_questions_promotes_question_fields(tmp_path, template):
    questions = [
        {"id": "q-1", "type": "Short Answer", "question": {"stem": "s?"}, "answerKey": "x",
         "standard": "STD-1", "ds-problem-name": "P1", "ds-step-name": ["s1", "s2"]},
        {"id": "q-2", "type": "Short Answer", "question": {"stem": "t?"}, "answerKey": "y",
         "standard": "STD-2", "ds-problem-name": "P2", "ds-step-name": ["s3"]},
    ]
    data_path = tmp_path / "questions.jsonl"
    data_path.write_text("\n".join(json.dumps(q) for q in questions) + "\n")

    out = create_kc_from_questions(str(data_path), template, kc_fields=["standard"], kc_names=["Standard"])
    assert out["KC (Standard)"].tolist() == ["STD-1", "STD-1", "STD-2"]


@pytest.fixture
def filled_kc_and_ss(tmp_path):
    kc = pd.DataFrame(
        {
            "Problem Hierarchy": ["(Unit A)", "(Unit A)", "(Unit B)"],
            "Problem Name": ["P1", "P1", "P2"],
            "Step Name": ["s1", "s2", "s3"],
            "KC (m)": ["alpha", "alpha", "beta"],
        }
    )
    ss = _write_tsv(
        pd.DataFrame(
            {
                "Anon Student Id": ["ST1", "ST1", "ST1", "ST2"],
                "Problem Hierarchy": ["Unit A", "Unit A", "Unit B", "Unit A"],
                "Problem Name": ["P1", "P1", "P2", "P1"],
                "Step Name": ["s1", "s2", "s3", "s1"],
                "First Transaction Time": ["t1", "t2", "t3", "t4"],
                "First Attempt": ["correct", "incorrect", "correct", "correct"],
            }
        ),
        tmp_path / "ss.txt",
    )
    return kc, ss


def test_merge_student_step_counts_opportunities_per_student(filled_kc_and_ss):
    kc, ss_path = filled_kc_and_ss
    ss = merge_student_step_with_kc(ss_path, kc)
    # ST1 practices KC alpha on s1 then s2 (opportunities 1, 2), beta once;
    # ST2's count starts fresh.
    assert ss["KC (m)"].tolist() == ["alpha", "alpha", "beta", "alpha"]
    assert ss["Opportunity (m)"].tolist() == [1, 2, 1, 1]


def test_merge_student_step_multiplier_duplicates_kc_columns(filled_kc_and_ss):
    kc, ss_path = filled_kc_and_ss
    ss = merge_student_step_with_kc(ss_path, kc, multiplier=3)
    for i in (1, 2):
        assert ss[f"KC (m-{i})"].tolist() == ss["KC (m)"].tolist()
        assert ss[f"Opportunity (m-{i})"].tolist() == ss["Opportunity (m)"].tolist()


def test_merge_student_step_minimal_encodes_identifiers(filled_kc_and_ss):
    kc, ss_path = filled_kc_and_ss
    ss = merge_student_step_with_kc(ss_path, kc, minimal=True)
    assert ss["Anon Student Id"].str.fullmatch(r"ST-\d+").all()
    assert ss["Step Name"].str.fullmatch(r"SN-\d+").all()
    assert ss["KC (m)"].str.fullmatch(r"KC-\d+").all()
