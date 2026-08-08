import os
import re

import pytest

from kcluster.paths import (
    RESULTS_DIR_ENV,
    RUN_DIR_ENV,
    default_output_dir,
    prepare_output_dir,
    results_root,
    step_dir,
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


def test_step_dir_locates_earlier_steps_only_within_a_run(monkeypatch):
    assert step_dir("concept") is None  # no run folder: caller must be explicit
    assert step_dir("concept", "/runs/exp1") == os.path.join("/runs/exp1", "concept")
    monkeypatch.setenv(RUN_DIR_ENV, "/runs/exp2")
    assert step_dir("pmi") == os.path.join("/runs/exp2", "pmi")


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
