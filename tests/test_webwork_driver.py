"""End-to-end tests of the WeBWorK dataset driver (datasets/webwork/processing.py)."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def driver():
    path = REPO_ROOT / "datasets" / "webwork" / "processing.py"
    spec = importlib.util.spec_from_file_location("webwork_processing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_data(driver, tmp_path):
    prob_dir = tmp_path / "problems"
    for p in ["Rogawski_Calculus/1.1_Intro/p1.pg", "Rogawski_Calculus/12_Vector_Geometry/p2.pg"]:
        (prob_dir / p).parent.mkdir(parents=True, exist_ok=True)
        (prob_dir / p).write_text(f"content of {p}")

    df = pd.DataFrame(
        {
            "Permission Level": ["student", "student", "student", "professor"],
            "Problem Path": ["Rogawski_Calculus/1.1_Intro/p1.pg"] * 2
            + ["Rogawski_Calculus/12_Vector_Geometry/p2.pg"] * 2,
            "OPL Subject": ["Calculus - single variable"] * 4,
            "Student ID hash": ["S1", "S1", "S2", "S2"],
            "Answer Timestamp": ["200", "100", "150", "150"],
            "OPL Chapter": ["Limits", "Limits", None, None],
            "OPL Section": ["Intro", "Intro", None, None],
        }
    )
    raw_path = tmp_path / "raw.csv"
    df.to_csv(raw_path, index=False)

    students_df, problems_df = driver.extract_data(str(raw_path), str(prob_dir), expected_records=3)

    # The professor row is dropped; rows are time-sorted within each student,
    # with students kept in first-appearance order
    assert students_df["Student ID hash"].tolist() == ["S1", "S1", "S2"]
    assert students_df["Answer Timestamp"].tolist() == [100, 200, 150]
    # The Vector Geometry chapter/section backfill
    assert students_df["OPL Chapter"].tolist() == ["Limits", "Limits", "Vector Geometry"]
    assert problems_df["Problem Content"].str.startswith("content of").all()
    assert len(problems_df) == 2


def test_create_datashop_transactions(driver, tmp_path):
    df = pd.DataFrame(
        {
            "Student ID hash": ["a" * 40, "b" * 40],
            "Answer Date": ["2024-01-01", "2024-01-02"],
            "Answer Timestamp": ["100", "200"],
            # The export's blank count is one more than the answers used
            "Number of Answer Blanks": ["3", "2"],
            "Answer 1 Value": ["x", "z"],
            "Answer 1 Status": ["1", "1"],
            "Answer 2 Value": ["y", "unused"],
            "Answer 2 Status": ["0", "1"],
            "OPL Chapter": ["Limits", "Integrals"],
            "OPL Section": ["Intro", "Net Change"],
            "Problem Path": ["P1", "P2"],
            "Attempt Number": ["1", "1"],
            "Problem Seed": ["7", "8"],
            "OPL Keywords": ["Limits, L'Hopital Rule", "integrals"],
        }
    )
    clean_path = tmp_path / "clean.csv"
    df.to_csv(clean_path, index=False)

    tx = driver.create_datashop_transactions(str(clean_path))

    assert len(tx) == 3  # 2 blanks for the first record + 1 for the second
    assert tx["Step Name"].tolist() == ["Blank-1", "Blank-2", "Blank-1"]
    assert tx["Outcome"].tolist() == ["CORRECT", "INCORRECT", "CORRECT"]
    assert tx["Input"].tolist() == ["x", "y", "z"]
    assert tx["Anon Student Id"].str.len().eq(32).all()
    assert tx["Time"].tolist() == [100000, 100000, 200000]
    # Keywords: apostrophes dropped, lowercased, sorted, "~~"-joined, spaces -> _
    assert tx["CF (Problem Keywords)"].iloc[0] == "lhopital_rule~~limits"


def test_create_chapter_n_section_kc(driver):
    kc_temp = pd.DataFrame(
        {
            "Problem Hierarchy": ["(Subject) Calculus, (Chapter) Limits, (Section) Continuity"] * 2,
            "Problem Name": ["P1", "P2"],
            "Step Name": ["Blank-1", "Blank-1"],
            "KC (Unique-step)": ["KC-1", None],
        }
    )
    kc = driver.create_chapter_n_section_kc(kc_temp)
    assert kc["KC (Chapter)"].iloc[0] == "Limits" and pd.isna(kc["KC (Chapter)"].iloc[1])
    assert kc["KC (Section)"].iloc[0] == "Continuity" and pd.isna(kc["KC (Section)"].iloc[1])
    assert "KC (Unique-step)" not in kc.columns


def test_create_keywords_kc(driver):
    kc_temp = pd.DataFrame(
        {
            "Problem Name": ["P1", "P2"],
            "Step Name": ["Blank-1", "Blank-1"],
            "KC (Unique-step)": ["KC-1", None],
        }
    )
    trans_df = pd.DataFrame(
        {
            "Problem Name": ["P1", "P1", "P2"],
            "CF (Problem Keywords)": ["limits", "limits", "integrals"],
        }
    )
    kc = driver.create_keywords_kc(kc_temp, trans_df)
    assert kc["KC (Keywords)"].iloc[0] == "limits" and pd.isna(kc["KC (Keywords)"].iloc[1])
    assert "KC (Unique-step)" not in kc.columns
