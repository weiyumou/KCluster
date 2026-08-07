"""End-to-end tests of the elearning dataset drivers (datasets/elearning/).

The drivers live outside the package on purpose (workspace scripts, not
wheel code), so they are loaded from the repo path.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("bs4")

from test_oli_html import _question_div, _write_html  # noqa: E402

from kcluster.io.jsonl import load_questions  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def driver():
    path = REPO_ROOT / "datasets" / "elearning" / "build.py"
    spec = importlib.util.spec_from_file_location("elearning_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def html_dir(tmp_path):
    # Two parseable questions; only mcq-one appears in the KC templates below
    root = tmp_path / "html"
    root.mkdir()
    _write_html(root / "page.html",
                _question_div(q_id="mcq-one", part_id="p1"),
                _question_div(q_id="mcq-two", part_id="p2", stem_ps=("A different stem?",)))
    return str(root)


def _write_template(path, step_names, kcs) -> str:
    pd.DataFrame(
        {
            "Problem Hierarchy": ["U"] * len(step_names),
            "Problem Name": ["P"] * len(step_names),
            "Step Name": step_names,
            "KC (expert)": kcs,
        }
    ).to_csv(path, sep="\t", index=False)
    return str(path)


def test_write_elearning22_mcqs(driver, html_dir, tmp_path):
    # 2022 step names start with "<question id>_<part id>"; the row without a
    # KC value must not admit its step
    temp_path = _write_template(tmp_path / "kcm.txt",
                                ["mcq-one_p1 Row1", "mcq-one_p1 Row2", "mcq-two_p2 Row3"],
                                ["e1", "e2", None])

    driver.write_elearning22_mcqs(html_dir, str(tmp_path), temp_path)

    [q] = load_questions(str(tmp_path / "elearning22-mcq.jsonl"))
    assert q["step-name"] == ["mcq-one_p1"]
    assert q["skillref"] == ["skill-a"]
    assert q["ds-step-name"] == ["mcq-one_p1 Row1", "mcq-one_p1 Row2"]


def test_write_elearning23_mcqs(driver, html_dir, tmp_path):
    # 2023 step names carry the part id after a "part " marker
    temp_path = _write_template(tmp_path / "kcm.txt",
                                ["Quiz1 part p1 Row1", "Quiz2 part p9 Row2"],
                                ["e1", "e2"])

    driver.write_elearning23_mcqs(html_dir, str(tmp_path), temp_path)

    [q] = load_questions(str(tmp_path / "elearning23-mcq.jsonl"))
    assert q["step-name"] == ["mcq-one_p1"]
    assert q["ds-step-name"] == ["Quiz1 part p1 Row1"]
