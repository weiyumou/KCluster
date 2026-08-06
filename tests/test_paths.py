import os
import re

import pytest

from kcluster.paths import (
    RESULTS_DIR_ENV,
    default_output_dir,
    prepare_output_dir,
    results_root,
    timestamp,
)


def test_results_root_defaults_and_honors_env(monkeypatch):
    monkeypatch.delenv(RESULTS_DIR_ENV, raising=False)
    assert results_root() == "results"
    monkeypatch.setenv(RESULTS_DIR_ENV, "/somewhere/else")
    assert results_root() == "/somewhere/else"


def test_timestamp_is_sortable_and_filesystem_safe():
    ts = timestamp()
    assert re.fullmatch(r"\d{8}-\d{6}", ts)


def test_default_output_dir_composes_root_step_timestamp(monkeypatch):
    monkeypatch.setenv(RESULTS_DIR_ENV, "/tmp/results")
    path = default_output_dir("concept")
    root, step, ts = path.rsplit(os.sep, 2)
    assert root == "/tmp/results"
    assert step == "concept"
    assert re.fullmatch(r"\d{8}-\d{6}", ts)


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
