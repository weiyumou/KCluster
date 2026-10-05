import ast
import json
import pathlib
import re

import pandas as pd
import pytest

import kcluster.datashop
from kcluster.datashop.kc_model import (
    KC_PAT,
    create_datashop_kc,
    create_default_kc,
    create_kc_from_questions,
    load_datashop_temp,
    load_kc_template,
    merge_student_step_with_kc,
    save_kc_template,
)


@pytest.fixture
def template(tmp_path, write_tsv) -> str:
    return write_tsv(
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


def test_create_datashop_kc_leaves_the_callers_frames_alone(template):
    # The first key column used to be split into lists in place, so a caller
    # reusing its KC frame (for another model or template) joined on lists
    kc = pd.DataFrame({"ds-problem-name": ["P1", "P2"], "ds-step-name": ["s1~s2", "s3"], "concept": ["a", "b"]})
    kc_temp = load_datashop_temp(template)
    kc_before, kc_temp_before = kc.copy(), kc_temp.copy()

    create_datashop_kc(kc, kc_temp, kc_cols=["concept"], match_other_kc=True, drop_other_kc=True)
    pd.testing.assert_frame_equal(kc, kc_before)
    pd.testing.assert_frame_equal(kc_temp, kc_temp_before)


def test_create_datashop_kc_match_other_kc_masks_uncovered_steps(tmp_path, write_tsv):
    template = write_tsv(
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
def filled_kc_and_ss(tmp_path, write_tsv):
    kc = pd.DataFrame(
        {
            "Problem Hierarchy": ["(Unit A)", "(Unit A)", "(Unit B)"],
            "Problem Name": ["P1", "P1", "P2"],
            "Step Name": ["s1", "s2", "s3"],
            "KC (m)": ["alpha", "alpha", "beta"],
        }
    )
    ss = write_tsv(
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


def test_kc_template_round_trips_cell_for_cell(tmp_path):
    # DataShop's import wants the export's columns back as they were: nothing
    # parsed, unquoted or dropped, bar the column the header's trailing tab makes
    path = tmp_path / "kcm.txt"
    path.write_text('Step ID\tProblem Name\tMax Problem View\tAvg Error Step Duration (sec)\tKC (m)\t\n'
                    'id1\t"Intro" quiz\t2.0\t\tthe "carrots" approach\n')
    save_kc_template(load_kc_template(str(path)), str(tmp_path / "out.txt"))
    assert (tmp_path / "out.txt").read_text() == path.read_text().replace("(m)\t\n", "(m)\n")


def test_the_toolbox_imports_only_the_core_and_the_question_reader():
    """The promise that lets another project use it on the base install: no
    engine, no task, no command behind any of its modules."""
    imported = set()
    for path in pathlib.Path(kcluster.datashop.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
    own = {name for name in imported if name.split(".")[0] == "kcluster"}
    allowed = ("kcluster.datashop.", "kcluster.core.", "kcluster.io.jsonl")
    assert {name for name in own if not name.startswith(allowed)} == set()
