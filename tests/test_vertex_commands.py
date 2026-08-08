"""Offline end-to-end tests of the vertex-launch / vertex-retrieve /
vertex-build-kc commands, sharing the engine tests' in-memory GCS fake."""

import argparse
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("google.cloud.aiplatform")

from google.cloud.aiplatform_v1 import JobState  # noqa: E402
from test_vertex_engine import FakeStorageClient  # noqa: E402

from kcluster.core.question import Question  # noqa: E402
from kcluster.engine import vertex  # noqa: E402
from kcluster.io.jsonl import dump_questions  # noqa: E402

GROUPS = ["alpha"] * 3 + ["beta"] * 3
CONCEPTS = GROUPS


@pytest.fixture()
def gcs(monkeypatch):
    FakeStorageClient._buckets = {}
    monkeypatch.setattr(vertex.storage, "Client", FakeStorageClient)
    return FakeStorageClient()


@pytest.fixture()
def config_path(tmp_path):
    path = tmp_path / "vertex.toml"
    path.write_text('[vertex]\nproject = "p"\nbucket = "my-bucket"\nmodel_id = "m"\n')
    return str(path)


def _questions(n=6) -> list[Question]:
    return [
        Question(
            {
                "id": f"q-{i}",
                "type": "Multiple Choice",
                "question": {"stem": f"Stem {i}?", "choices": [{"label": "a", "text": "x"}]},
                "answerKey": "a",
            }
        )
        for i in range(n)
    ]


def _collected_pmi(n=6):
    """A stacked [marginals; conditionals] array with two planted clusters per concept."""
    marginals = np.full(n, -50.0)
    same = np.array([[g1 == g2 for g2 in GROUPS] for g1 in GROUPS])
    conds = np.where(same, -45.0, -55.0)
    return np.vstack([marginals, conds])


def _store_collected(store, job_id, n=6):
    buf = __import__("io").BytesIO()
    np.save(buf, _collected_pmi(n))
    store[f"batch-output/{job_id}/pmi.npy"] = buf.getvalue()
    store[f"batch-output/{job_id}/concepts.jsonl"] = "\n".join(
        json.dumps({"concept": c}) for c in CONCEPTS[:n])


def test_vertex_build_kc(gcs, config_path, tmp_path):
    from kcluster.commands.vertex_build_kc import main

    questions = _questions()
    data_path = tmp_path / "my questions.jsonl"
    dump_questions(questions, str(data_path))

    work_dir = tmp_path / "run"
    work_dir.mkdir()
    (work_dir / "launched_jobs.jsonl").write_text(
        json.dumps({"job_id": "job1", "data_path": str(data_path), "resource_name": "jobs/1"}) + "\n")
    _store_collected(gcs.bucket("my-bucket").store, "job1")

    main(argparse.Namespace(work_dir=str(work_dir), normalize=False, config=config_path))

    out_dir = work_dir / "kc" / "my-questions"
    assert np.load(out_dir / "my-questions_pmi-unnorm.npy").shape == (6, 6)

    import pandas as pd
    concept_kc = pd.read_csv(out_dir / "my-questions_concept-kc.csv")
    assert concept_kc["KC"].tolist() == CONCEPTS

    kcluster_kc = pd.read_csv(out_dir / "my-questions_kcluster-unnorm-kc.csv")
    # Clusters follow the planted blocks; labels come from exemplar concepts
    assert kcluster_kc["KC"].tolist() == CONCEPTS
    assert kcluster_kc["KC-raw"].str.match(r"KC-\d+").all()


def test_vertex_build_kc_requires_collected_results(gcs, config_path, tmp_path):
    from kcluster.commands.vertex_build_kc import main

    data_path = tmp_path / "qs.jsonl"
    dump_questions(_questions(), str(data_path))
    work_dir = tmp_path / "run"
    work_dir.mkdir()
    (work_dir / "launched_jobs.jsonl").write_text(
        json.dumps({"job_id": "missing", "data_path": str(data_path), "resource_name": "jobs/1"}) + "\n")

    with pytest.raises(AssertionError, match="vertex-retrieve"):
        main(argparse.Namespace(work_dir=str(work_dir), normalize=False, config=config_path))


def test_vertex_retrieve(gcs, config_path, tmp_path, monkeypatch):
    from kcluster.commands.vertex_retrieve import main

    data_path = tmp_path / "qs.jsonl"
    dump_questions(_questions(), str(data_path))

    # job1 is collected; job2 is not and its job failed on Vertex
    _store_collected(gcs.bucket("my-bucket").store, "job1")
    jobs_path = tmp_path / "launched_jobs.jsonl"
    jobs_path.write_text("\n".join(
        json.dumps({"job_id": jid, "data_path": str(data_path), "resource_name": f"jobs/{jid}"})
        for jid in ("job1", "job2")))

    monkeypatch.setattr(vertex.aiplatform, "init", lambda **kw: None)
    from kcluster.commands import vertex_retrieve
    monkeypatch.setattr(vertex_retrieve.aiplatform, "BatchPredictionJob",
                        lambda name: SimpleNamespace(state=JobState.JOB_STATE_FAILED))

    out_dir = tmp_path / "pmi"
    main(argparse.Namespace(jobs_path=str(jobs_path), output_dir=str(out_dir),
                            symmetric=True, normalize=False, config=config_path))

    [saved] = sorted(out_dir.iterdir())
    assert saved.name == "job1_pmi-sym-unnorm.npy"
    mat = np.load(saved)
    assert mat.shape == (6, 6)
    assert np.allclose(mat, mat.T)  # symmetrized
    assert mat[0, 1] > mat[0, 3]  # within-cluster similarity beats across


def test_vertex_launch(gcs, config_path, tmp_path, monkeypatch):
    from kcluster.commands.vertex_launch import main

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dump_questions(_questions(2), str(data_dir / "course a.jsonl"))

    class FakeJob:
        _gca_resource = SimpleNamespace(name="jobs/1")
        resource_name = "jobs/1"
        name, display_name = "jobs/1", "job"
        state = JobState.JOB_STATE_FAILED  # terminal, so the wait loop exits
        error = "boom"

    captured = {}

    class FakeModel:
        def __init__(self, resource):
            captured["model"] = resource

        def batch_predict(self, **kwargs):
            captured["batch_predict"] = kwargs
            return FakeJob()

    monkeypatch.setattr(vertex.aiplatform, "init", lambda **kw: None)
    monkeypatch.setattr(vertex.aiplatform, "Model", FakeModel)
    monkeypatch.setattr(vertex.time, "sleep", lambda s: None)
    monkeypatch.setenv("KCLUSTER_RESULTS_DIR", str(tmp_path / "results"))

    main(argparse.Namespace(data_path=str(data_dir), completion_time=60.0,
                            secs_per_batch=0.1, batch_size=16, config=config_path))

    # The job log records the launched job with a data-derived, timestamped id
    [run] = list((tmp_path / "results").iterdir())  # run-major: <results>/<run>/<step>
    run_dir = run / "vertex-launch"
    [logged] = [json.loads(line) for line in (run_dir / "launched_jobs.jsonl").read_text().splitlines()]
    assert re.fullmatch(r"course-a_\d{8}-\d{6}", logged["job_id"])
    assert logged["resource_name"] == "jobs/1"

    # The instances landed in the fake bucket under the job id
    store = gcs.bucket("my-bucket").store
    assert f"batch-input/{logged['job_id']}/instances.jsonl" in store
    assert captured["batch_predict"]["batch_size"] == 16


def test_cli_lists_vertex_commands(capsys):
    from kcluster.cli import main

    main(["--help"])
    out = capsys.readouterr().out
    for command in ("vertex-launch", "vertex-retrieve", "vertex-build-kc"):
        assert command in out
