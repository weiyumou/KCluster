"""End-to-end tests of the elearning dataset drivers.

The drivers live outside the package on purpose (workspace scripts, not wheel
code), so they are loaded from the repo path. The two offerings share the
course HTML and the whole procedure; what is tested here is what differs —
each one's step-name notation and the KC models it carries.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("bs4")

from test_oli_html import _question_div, _write_html  # noqa: E402

from kcluster.io.jsonl import load_questions  # noqa: E402
from kcluster.io.student_step import (  # noqa: E402
    MINIMAL_SUFFIX,
    check_coverage,
    load_student_step,
    validate_student_step,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_driver(workspace: str):
    path = REPO_ROOT / "datasets" / workspace / "build.py"
    spec = importlib.util.spec_from_file_location(f"{workspace}_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver22():
    return _load_driver("elearning22")


@pytest.fixture(scope="module")
def driver23():
    return _load_driver("elearning23")


@pytest.fixture()
def html_dir(tmp_path):
    # Two parseable questions; only mcq-one is inside the universe below
    root = tmp_path / "html"
    root.mkdir()
    _write_html(root / "page.html",
                _question_div(q_id="mcq-one", part_id="p1"),
                _question_div(q_id="mcq-two", part_id="p2", stem_ps=("A different stem?",)))
    return str(root)


def _write_export(path, rows, kc_model="expert") -> str:
    """A miniature DataShop student-step export: (step, KC, outcome) per row."""
    pd.DataFrame({
        "Anon Student Id": [f"stu{i % 2}" for i in range(len(rows))],
        "Problem Hierarchy": "unit U",
        "Problem Name": "P",
        "Step Name": [step for step, _, _ in rows],
        "First Transaction Time": [f"2022-08-30 13:{i:02d}:00" for i in range(len(rows))],
        "First Attempt": [outcome for _, _, outcome in rows],
        f"KC ({kc_model})": [kc for _, kc, _ in rows],
        f"Opportunity ({kc_model})": 1,     # dropped: the tagger recomputes these
        "Step Duration (sec)": 3.5,         # dropped: not part of the contract
    }).to_csv(path, sep="\t", index=False)
    return str(path)


@pytest.fixture()
def tiers(tmp_path):
    """The two tiers a driver writes into: the JSONL and the minimal file are split."""
    out_dir, interim_dir = tmp_path / "processed", tmp_path / "interim"
    out_dir.mkdir()
    interim_dir.mkdir()
    return out_dir, interim_dir


def _assert_pair_is_consistent(tiers, ds: str) -> pd.DataFrame:
    """The invariant both drivers exist to hold: the two halves agree."""
    out_dir, interim_dir = tiers
    questions = load_questions(str(out_dir / f"{ds}.jsonl"))
    ss = load_student_step(str(interim_dir / f"{ds}{MINIMAL_SUFFIX}"))
    validate_student_step(ss)
    assert check_coverage(questions, ss) == []
    return ss


# --- 2022 offering (ds5426): steps lead with the OLI step name ---

def test_write_elearning22_pair(driver22, html_dir, tmp_path, tiers):
    # mcq-two's step is untagged by the expert model, so it is outside the
    # universe: its question is dropped and so are its rows.
    export = _write_export(tmp_path / "export.txt", [
        ("mcq-one_p1 Row1", "e1", "correct"),
        ("mcq-one_p1 Row2", "e2", "incorrect"),
        ("mcq-one_p1 Row1", "e1", "hint"),
        ("mcq-two_p2 Row3", None, "correct"),
    ])

    out_dir, interim_dir = tiers
    driver22.write_elearning22(html_dir, export, str(out_dir), str(interim_dir), kc_models=("expert",))

    [q] = load_questions(str(out_dir / "elearning22-mcq.jsonl"))
    assert q["step-name"] == ["mcq-one_p1"]
    assert q["skillref"] == ["skill-a"]
    assert q["ds-step-name"] == ["mcq-one_p1 Row1", "mcq-one_p1 Row2"]

    ss = _assert_pair_is_consistent(tiers, "elearning22-mcq")
    # the unresolvable row is dropped; Opportunity and extra columns are gone
    assert list(ss.columns) == ["Anon Student Id", "Problem Hierarchy", "Problem Name", "Step Name",
                                "First Transaction Time", "First Attempt", "KC (expert)"]
    assert ss["Step Name"].tolist() == ["mcq-one_p1 Row1", "mcq-one_p1 Row2", "mcq-one_p1 Row1"]


def test_write_elearning22_keeps_rows_the_experts_left_untagged(driver22, html_dir, tmp_path, tiers):
    # A step inside the universe can still carry rows the experts did not tag
    # (elearning22 has 139). They resolve to a question, so they stay — inert
    # under every model, but preserving the student's history.
    export = _write_export(tmp_path / "export.txt", [
        ("mcq-one_p1 Row1", "e1", "correct"),
        ("mcq-one_p1 Row1", None, "correct"),
    ])
    out_dir, interim_dir = tiers
    driver22.write_elearning22(html_dir, export, str(out_dir), str(interim_dir), kc_models=("expert",))

    ss = load_student_step(str(interim_dir / f"elearning22-mcq{MINIMAL_SUFFIX}"))
    assert len(ss) == 2
    assert ss["KC (expert)"].tolist() == ["e1", ""]


# --- 2023 offering (ds5843): the part id sits after a "part" marker ---

def test_write_elearning23_pair(driver23, html_dir, tmp_path, tiers):
    export = _write_export(tmp_path / "export.txt", [
        ("Activity alpha, part p1 Multiple choice submission", "v1", "correct"),
        ("Activity beta, part p1 Multiple choice submission", "v2", "incorrect"),
        ("Activity gamma, part p2 Multiple choice submission", None, "correct"),
    ], kc_model="expert")

    out_dir, interim_dir = tiers
    driver23.write_elearning23(html_dir, export, str(out_dir), str(interim_dir), kc_models=("expert",))

    [q] = load_questions(str(out_dir / "elearning23-mcq.jsonl"))
    assert q["step-name"] == ["mcq-one_p1"]
    assert q["ds-step-name"] == ["Activity alpha, part p1 Multiple choice submission",
                                 "Activity beta, part p1 Multiple choice submission"]

    ss = _assert_pair_is_consistent(tiers, "elearning23-mcq")
    assert ss["KC (expert)"].tolist() == ["v1", "v2"]


def test_write_elearning23_ignores_steps_without_a_part_marker(driver23, html_dir, tmp_path, tiers):
    # ds5843's step names all carry the marker, but a step without one has no
    # key to join on and must be skipped rather than crash the join.
    export = _write_export(tmp_path / "export.txt", [
        ("Activity alpha, part p1 Multiple choice submission", "v1", "correct"),
        ("A step with no marker", "v2", "correct"),
    ], kc_model="expert")

    out_dir, interim_dir = tiers
    driver23.write_elearning23(html_dir, export, str(out_dir), str(interim_dir), kc_models=("expert",))

    ss = _assert_pair_is_consistent(tiers, "elearning23-mcq")
    assert ss["Step Name"].tolist() == ["Activity alpha, part p1 Multiple choice submission"]
