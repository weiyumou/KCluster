import json
import os
import re

import pytest

from kcluster.paths import (
    RESULTS_DIR_ENV,
    RUN_DIR_ENV,
    dataset_name,
    default_output_dir,
    prepare_output_dir,
    question_path,
    required_result_dir,
    result_dataset,
    results_root,
    save_args,
    timestamp,
)


@pytest.fixture(autouse=True)
def _no_ambient_run_dir(monkeypatch):
    monkeypatch.delenv(RUN_DIR_ENV, raising=False)


def test_results_root_defaults_and_honors_env(monkeypatch):
    monkeypatch.delenv(RESULTS_DIR_ENV, raising=False)
    assert results_root() == "results"
    monkeypatch.setenv(RESULTS_DIR_ENV, "/somewhere/else")
    assert results_root() == "/somewhere/else"


def test_timestamp_is_sortable_and_filesystem_safe():
    ts = timestamp()
    assert re.fullmatch(r"\d{8}-\d{6}", ts)


def test_default_output_dir_mints_a_run_folder_per_invocation(monkeypatch):
    # Run-major layout: <results>/<run>/<step>, so one run's steps sit together
    monkeypatch.setenv(RESULTS_DIR_ENV, "/tmp/results")
    path = default_output_dir("concept")
    root, ts, step = path.rsplit(os.sep, 2)
    assert root == "/tmp/results"
    assert step == "concept"
    assert re.fullmatch(r"\d{8}-\d{6}", ts)


def test_run_dir_keeps_every_step_of_a_run_together(monkeypatch):
    monkeypatch.setenv(RESULTS_DIR_ENV, "/tmp/results")
    assert default_output_dir("concept", "/runs/exp1") == os.path.join("/runs/exp1", "concept")
    assert default_output_dir("pmi", "/runs/exp1") == os.path.join("/runs/exp1", "pmi")

    # ...and the env var does the same without touching call sites
    monkeypatch.setenv(RUN_DIR_ENV, "/runs/exp2")
    assert default_output_dir("kc") == os.path.join("/runs/exp2", "kc")
    # an explicit flag still wins over the environment
    assert default_output_dir("kc", "/runs/exp3") == os.path.join("/runs/exp3", "kc")


def test_prepare_output_dir_creates_and_returns_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = prepare_output_dir("nested/run")
    assert os.path.isabs(out)
    assert os.path.isdir(out)


def test_prepare_output_dir_can_refuse_existing(tmp_path):
    target = tmp_path / "run"
    target.mkdir()
    with pytest.raises(FileExistsError):
        prepare_output_dir(str(target), exist_ok=False)


def test_required_result_dir_takes_the_flag_then_the_run_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert required_result_dir("rd") == str(tmp_path / "rd")  # absolute, wherever it came from
    assert required_result_dir(None, "/runs/exp1") == "/runs/exp1"
    monkeypatch.setenv(RUN_DIR_ENV, "/runs/exp2")
    assert required_result_dir() == "/runs/exp2"
    assert required_result_dir("/explicit") == "/explicit"  # the flag wins over the run folder


def test_required_result_dir_never_mints_a_folder():
    with pytest.raises(SystemExit, match="--result_dir is required"):
        required_result_dir()


def test_dataset_name_is_the_stem_with_spaces_hyphenated():
    assert dataset_name("/data/EPLA Physics.jsonl") == "EPLA-Physics"
    assert dataset_name("elearning22-mcq.jsonl") == "elearning22-mcq"


def test_result_dataset_is_read_off_the_concept_kc(tmp_path):
    (tmp_path / "kc" / "concept").mkdir(parents=True)
    (tmp_path / "kc" / "concept" / "my_course_concept-kc.csv").write_text("")
    assert result_dataset(str(tmp_path)) == "my_course"


def test_question_path_reads_any_steps_breadcrumb(tmp_path):
    """A Vertex course dir has no args-concept; its args-kc records the path just as well."""
    data = tmp_path / "questions.jsonl"
    data.write_text("")
    rd = str(tmp_path)
    save_args(rd, "embed", "ds", {"sent_path": "st"})  # sorts first, records no data_path
    save_args(rd, "kc", "ds", {"data_path": str(data)})
    save_args(rd, "kc", "other", {"data_path": "/another/dataset.jsonl"})

    assert json.loads((tmp_path / "args-kc-ds.json").read_text()) == {"data_path": str(data)}
    assert question_path(rd, "ds") == str(data)


def test_question_path_prefers_the_flag_and_names_what_is_missing(tmp_path):
    data = tmp_path / "here.jsonl"
    data.write_text("")
    rd = str(tmp_path)
    with pytest.raises(SystemExit, match="No data_path recorded"):
        question_path(rd, "ds")

    save_args(rd, "concept", "ds", {"data_path": "/on/the/cluster.jsonl"})
    with pytest.raises(SystemExit, match="Question file not found: /on/the/cluster.jsonl"):
        question_path(rd, "ds")
    assert question_path(rd, "ds", str(data)) == str(data)
