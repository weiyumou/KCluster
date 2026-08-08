"""End-to-end test of the spacing dataset driver (datasets/spacing-exp2/processing.py)."""

import importlib.util
import random
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def driver():
    path = REPO_ROOT / "datasets" / "spacing-exp2" / "processing.py"
    spec = importlib.util.spec_from_file_location("spacing_processing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_tx(tmp_path) -> str:
    # Two questions: one MCQ (three transactions, one in a dropped stage) and
    # one True or False. Duplicated Condition columns mirror the raw export
    # (pandas mangles the second pair to "Condition Name.1"/"Condition Type.1").
    header = ["Anon Student Id", "Session Id", "Time", "Level (Course)", "Problem Name",
              "Problem Start Time", "Step Name", "Outcome", "Input",
              "Condition Name", "Condition Type", "Condition Name", "Condition Type",
              "School", "Class",
              "CF (Question Id)", "CF (Full Problem Name)", "CF (Question Type)",
              "CF (Answer Options)", "CF (Correct Answer Options)",
              "CF (Topic Text)", "KC (question_group)",
              "CF (Stage)", "CF (Assignment Day)",
              "CF (Response Time)", "CF (Completion Time)",
              "CF (Anon Teacher Id)", "CF (Course Name)",
              "CF (Timed Out)", "CF (District Id)"]

    def row(problem, q_type, options, answer, stage, group):
        return ["S1", "sess", "t0", "Biology", problem, "t0", "step-raw", "CORRECT", "resp",
                "cond", "ct", "cond2", "ct2", "school", "classA",
                "old-id", f"What is {problem}?", q_type,
                options, answer,
                "standard-1", group,
                stage, "day1",
                "100", "1700000000000",
                "T1", "Bio",
                "0", "D1"]

    rows = [
        row("P1", "Multiple Choice", "Alpha|Beta|Gamma", "Beta", "learning", "g1"),
        row("P1", "Multiple Choice", "Alpha|Beta|Gamma", "Beta", "post-test", "g1"),
        row("P1", "Multiple Choice", "Alpha|Beta|Gamma", "Beta", "practice", "g1"),  # dropped stage
        row("P2", "True or False", "True|", "True|", "learning", "g2"),
    ]
    path = tmp_path / "raw_tx.txt"
    path.write_text("\n".join("\t".join(r) for r in [header] + rows))
    return str(path)


def test_clean_transactions(driver, tmp_path):
    tx = driver.clean_transactions(_raw_tx(tmp_path))

    assert len(tx) == 3  # the "practice" stage row is dropped
    # Stable per-type question IDs replace the export's IDs and step names
    assert tx["CF (Question ID)"].tolist() == ["mcq-0", "mcq-0", "tof-0"]
    assert tx["Step Name"].tolist() == ["mcq-0", "mcq-0", "tof-0"]
    assert tx["CF (Question Group)"].tolist() == ["QG-g1", "QG-g1", "QG-g2"]
    assert "Condition Name" in tx.columns and "Condition Name.1" not in tx.columns
    assert tx["CF (Completion Time)"].iloc[0] == "2023-11-14 22:13:20"


def test_extract_questions_by_type(driver, tmp_path):
    tx = driver.clean_transactions(_raw_tx(tmp_path))

    [mcq] = driver.extract_questions_by_type(tx, "Multiple Choice", rng=random.Random(42))
    assert mcq["id"] == "mcq-0"
    assert mcq["question"]["stem"] == "What is p1?"  # str.capitalize() lowercases the rest
    labels = [c["label"] for c in mcq["question"]["choices"]]
    assert labels == ["a", "b", "c"]
    assert {c["text"] for c in mcq["question"]["choices"]} == {"Alpha", "Beta", "Gamma"}
    # The answer key is the label of the shuffled choice whose text matches
    [ans_text] = [c["text"] for c in mcq["question"]["choices"] if c["label"] == mcq["answerKey"]]
    assert ans_text == "Beta"
    assert (mcq["ds-course"], mcq["standard"], mcq["q-group"]) == ("Biology", "standard-1", "QG-g1")

    [tof] = driver.extract_questions_by_type(tx, "True or False")
    assert tof["answerKey"] == "True"  # first non-empty option of "True|"
    assert "choices" not in tof["question"]


def test_unknown_question_type_is_rejected(driver, tmp_path):
    tx = driver.clean_transactions(_raw_tx(tmp_path))
    with pytest.raises(AssertionError):
        driver.extract_questions_by_type(tx, "Essay")
