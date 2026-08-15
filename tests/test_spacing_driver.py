"""End-to-end test of the spacing dataset driver (datasets/spacing-exp2/processing.py)."""

import importlib.util
import logging
import random
from pathlib import Path

import pytest

from kcluster.io.student_step import MINIMAL_COLUMNS, check_coverage, validate_student_step

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def driver():
    path = REPO_ROOT / "datasets" / "spacing-exp2" / "processing.py"
    spec = importlib.util.spec_from_file_location("spacing_processing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_tx(tmp_path) -> str:
    # Two questions: one MCQ (five transactions — a retry, a spaced re-encounter
    # in the same stage, a second stage, and one in a dropped stage) and one True
    # or False. Duplicated Condition columns mirror the raw export (pandas
    # mangles the second pair to "Condition Name.1"/"Condition Type.1").
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

    def row(problem, q_type, options, answer, stage, group, start, outcome="CORRECT", session="s1"):
        return ["S1", session, "t0", "Biology", problem, start, "step-raw", outcome, "resp",
                "cond", "ct", "cond2", "ct2", "school", "classA",
                "old-id", f"What is {problem}?", q_type,
                options, answer,
                "standard-1", group,
                stage, "day1",
                "100", "1700000000000",
                "T1", "Bio",
                "0", "D1"]

    mcq = ("Multiple Choice", "Alpha|Beta|Gamma", "Beta")
    rows = [
        row("P1", *mcq, "learning", "g1", "1700000000000", outcome="INCORRECT"),
        # a retry after feedback: same sitting, so the same encounter
        row("P1", *mcq, "learning", "g1", "1700000060000"),
        # the spaced re-presentation the study is about: same stage, new sitting,
        # and this time the student gets it right unaided — the learning an
        # opportunity exists to measure, invisible while this row was collapsed
        row("P1", *mcq, "learning", "g1", "1700259260000", session="s2"),
        row("P1", *mcq, "post-test", "g1", "1700518460000", session="s3"),  # its own encounter
        row("P1", *mcq, "practice", "g1", "1700518520000", session="s3"),  # dropped stage
        row("P2", "True or False", "True|", "True|", "learning", "g2", "1700604860000", session="s4"),
    ]
    path = tmp_path / "raw_tx.txt"
    path.write_text("\n".join("\t".join(r) for r in [header] + rows))
    return str(path)


def test_clean_transactions(driver, tmp_path):
    tx = driver.clean_transactions(_raw_tx(tmp_path))

    assert len(tx) == 5  # the "practice" stage row is dropped
    # Stable per-type question IDs replace the export's IDs and step names
    assert tx["CF (Question ID)"].tolist() == ["mcq-0"] * 4 + ["tof-0"]
    assert tx["Step Name"].tolist() == ["mcq-0"] * 4 + ["tof-0"]
    assert tx["CF (Question Group)"].tolist() == ["QG-g1"] * 4 + ["QG-g2"]
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


def test_expert_kcs_collapse_to_one_label_per_step(driver, tmp_path, caplog):
    # P1's four kept transactions disagree about the question group; a KC model
    # maps steps to KCs, so the majority label wins for all of them.
    raw = Path(_raw_tx(tmp_path))
    rows = raw.read_text().splitlines()
    rows[2] = rows[2].replace("\tg1\t", "\tg-stray\t")  # one of three P1 rows
    raw.write_text("\n".join(rows))

    with caplog.at_level(logging.INFO):
        tx = driver.clean_transactions(str(raw))

    assert tx["CF (Question Group)"].tolist() == ["QG-g1"] * 4 + ["QG-g2"]
    assert "collapsed to 'QG-g1' (3 rows), dropping 'QG-g-stray' (1)" in caplog.text

    # Both halves inherit it: the question's q-group and the student-step column
    [mcq] = driver.extract_questions_by_type(tx, "Multiple Choice", rng=random.Random(42))
    assert mcq["q-group"] == "QG-g1"
    assert driver.build_student_step(tx)["KC (Question Group)"].tolist() == ["QG-g1"] * 3 + ["QG-g2"]


def test_build_student_step(driver, tmp_path):
    tx = driver.clean_transactions(_raw_tx(tmp_path))
    ss = driver.build_student_step(tx)

    validate_student_step(ss)
    assert all(col in ss.columns for col in MINIMAL_COLUMNS)
    assert "Session Id" not in ss.columns  # the session keys the encounter, it is not a column
    assert "CF (Stage)" not in ss.columns

    # One row per (course, step, student, session). The MCQ's three encounters:
    # the retry collapses into the sitting it belongs to, the spaced
    # re-presentation in the same stage is its own row, and so is the post-test.
    assert len(ss) == 4
    assert ss["Step Name"].tolist() == ["mcq-0", "mcq-0", "mcq-0", "tof-0"]
    # first attempt of each encounter, not the best one in it: the MCQ's opening
    # attempt was wrong, the retry that followed the feedback is not a row
    assert ss["First Attempt"].tolist() == ["incorrect", "correct", "correct", "correct"]
    assert ss["Problem Hierarchy"].unique().tolist() == ["Course Biology"]  # as DataShop renders a Level
    assert ss["First Transaction Time"].tolist() == ["2023-11-14 22:13:20", "2023-11-17 22:14:20",
                                                     "2023-11-20 22:14:20", "2023-11-21 22:14:20"]
    assert ss["KC (Topic)"].tolist() == ["standard-1"] * 4
    assert ss["KC (Question Group)"].tolist() == ["QG-g1", "QG-g1", "QG-g1", "QG-g2"]


def test_a_retry_after_feedback_does_not_become_an_opportunity(driver, tmp_path):
    # The distinction the session key exists to make: same step, same stage, two
    # transactions. In one sitting they are one encounter; in two they are two.
    tx = driver.clean_transactions(_raw_tx(tmp_path))
    mcq = tx[tx["Step Name"].eq("mcq-0")]

    one_sitting = mcq[mcq["Session Id"].eq("s1")]
    assert len(one_sitting) == 2 and len(driver.build_student_step(one_sitting)) == 1

    two_sittings = mcq[mcq["Session Id"].isin(["s1", "s2"])].drop_duplicates(subset="Session Id")
    assert len(two_sittings) == 2 and len(driver.build_student_step(two_sittings)) == 2


def test_pair_is_self_consistent(driver, tmp_path):
    # The reason both halves are written by one pass: every student-step row
    # resolves to exactly one question, and no question is left without rows.
    tx = driver.clean_transactions(_raw_tx(tmp_path))
    questions = [q for q_type in ("Multiple Choice", "True or False")
                 for q in driver.extract_questions_by_type(tx, q_type, rng=random.Random(42))]

    assert check_coverage(questions, driver.build_student_step(tx)) == []


def test_unknown_question_type_is_rejected(driver, tmp_path):
    tx = driver.clean_transactions(_raw_tx(tmp_path))
    with pytest.raises(AssertionError):
        driver.extract_questions_by_type(tx, "Essay")
