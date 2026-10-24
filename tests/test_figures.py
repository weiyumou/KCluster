"""Figure paths: the root a bank's relative paths resolve against, and the
pre-flight check that every file is there."""

import os

import pytest

from kcluster.core.question import Question
from kcluster.io.figures import check_figures, figure_dir, figure_paths, has_figures


def _with_figures(files: list[str]) -> Question:
    return Question({"id": "q", "type": "Short Answer", "answerKey": "0",
                     "question": {"stem": "\n".join(["[figure]"] * len(files) + ["f'(1) = ___"])},
                     "figures": files})


def test_figure_dir_defaults_to_the_question_files_folder(tmp_path):
    bank = tmp_path / "processed" / "bank.jsonl"
    assert figure_dir(str(bank)) == str(tmp_path / "processed")
    assert figure_dir(str(bank), str(tmp_path / "elsewhere")) == str(tmp_path / "elsewhere")


def test_figure_paths_join_the_root_in_marker_order(tmp_path):
    q = _with_figures(["raw/figures/a.png", "lib/b.png"])
    assert figure_paths(q, str(tmp_path)) == [str(tmp_path / "raw" / "figures" / "a.png"),
                                              str(tmp_path / "lib" / "b.png")]
    assert has_figures([q]) and not has_figures([_with_figures([])])


def test_check_figures_counts_questions_and_names_missing_files(tmp_path):
    os.makedirs(tmp_path / "raw" / "figures")
    (tmp_path / "raw" / "figures" / "a.png").write_bytes(b"png")
    shown = check_figures([_with_figures(["raw/figures/a.png"]), _with_figures([])], str(tmp_path))
    assert shown == 1
    with pytest.raises(FileNotFoundError, match=r"lib/b\.png.*--figure_dir"):
        check_figures([_with_figures(["raw/figures/a.png", "lib/b.png"])], str(tmp_path))
