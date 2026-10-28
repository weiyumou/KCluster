"""Offline tests for the Vertex AI engine: config loading, job preparation,
and prediction collection against the in-memory GCS fake (conftest's ``gcs``)."""

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("google.cloud.aiplatform")

from kcluster.engine import vertex
from kcluster.engine.vertex import (
    VertexConfig,
    collect_predictions,
    launch_batch_job,
    prepare_concept_jobs,
    prepare_inputs,
    prepare_pmi_jobs,
)

CONFIG = VertexConfig(project="my-project", bucket="my-bucket", model_id="123")


# --- VertexConfig ---

def test_config_load_from_toml_with_env_override(tmp_path, monkeypatch):
    config_path = tmp_path / "vertex.toml"
    config_path.write_text(
        '[vertex]\nproject = "p1"\nbucket = "b1"\nmodel_id = "m1"\nlocation = "europe-west4"\n'
    )
    monkeypatch.setenv("KCLUSTER_VERTEX_BUCKET", "b2")

    config = VertexConfig.load(str(config_path))
    assert (config.project, config.bucket, config.location) == ("p1", "b2", "europe-west4")
    assert config.model_resource == "projects/p1/locations/europe-west4/models/m1@1"


def test_config_load_from_env_only(monkeypatch, tmp_path):
    config_path = tmp_path / "vertex.toml"
    config_path.write_text('[vertex]\nproject = "p1"\nbucket = "b1"\nmodel_id = "m1"\n')
    monkeypatch.setenv("KCLUSTER_VERTEX_CONFIG", str(config_path))
    assert VertexConfig.load().project == "p1"


def test_config_missing_required_fields(monkeypatch):
    monkeypatch.delenv("KCLUSTER_VERTEX_CONFIG", raising=False)
    with pytest.raises(ValueError, match="missing vertex config"):
        VertexConfig.load()


def test_config_rejects_unknown_keys(tmp_path):
    config_path = tmp_path / "vertex.toml"
    config_path.write_text('[vertex]\nproject = "p"\nbucket = "b"\nmodel_id = "m"\nbukcet = "typo"\n')
    with pytest.raises(ValueError, match="unknown vertex config keys"):
        VertexConfig.load(str(config_path))


# --- job preparation ---

def test_prepare_concept_jobs_prompt_and_parameters(make_questions):
    [q] = make_questions(1)
    jobs = prepare_concept_jobs([q])

    [instance] = jobs["instances"]
    assert instance["id"] == "concept-0"
    assert instance["purpose"] == "complete_prompts"
    assert instance["text"] == (
        f"{q.header(1)}\n{q}\n\nRemark:\nThe above exercise is a multiple-choice "
        "question that tests whether the student understands the concept of"
    )
    assert jobs["parameters"]["complete_prompts"]["stop_tokens"] == [".", ","]
    assert jobs["parameters"]["complete_prompts"]["num_beams"] == 5

    verbal = prepare_concept_jobs([q], verbal=True)
    assert verbal["instances"][0]["text"].endswith("tests whether the student can")

    overridden = prepare_concept_jobs([q], configs={"max_new_tokens": 10})
    assert overridden["parameters"]["complete_prompts"]["max_new_tokens"] == 10


def test_prepare_pmi_jobs_uses_the_canonical_grid(make_questions):
    questions = make_questions(2)
    jobs = prepare_pmi_jobs(questions)

    instances = jobs["instances"]
    assert len(instances) == 2**2 + 2
    assert [item["id"] for item in instances] == [f"pmi-{i}" for i in range(6)]
    # Marginals score under the "Exercise 2:" header (decision D2),
    # matching the local engine's PairQuestion exactly
    assert instances[0]["context"] == f"{questions[0].header(2)}\n"
    assert instances[0]["context"].startswith("Exercise 2:\n")
    assert instances[0]["text"] == str(questions[0])
    # Grid entry (row 1, col 0): question 0 conditioned on question 1
    grid = instances[2 + 1 * 2 + 0]
    assert grid["context"] == f"{questions[1].header(1)}\n{questions[1]}\n\n{questions[0].header(2)}\n"
    assert grid["text"] == str(questions[0])
    assert all(item["purpose"] == "log_prob" for item in instances)


def test_job_preparation_takes_a_prompt_style(make_questions):
    from kcluster.core.prompts import ChatPrompts, StockPrompts

    questions = make_questions(2)
    style = StockPrompts(remark="The above exercise tests {trailer}")
    concept = prepare_concept_jobs(questions, prompts=style)
    assert concept["instances"][0]["text"].endswith("Remark:\nThe above exercise tests whether the student "
                                                    "understands the concept of")
    pmi = prepare_pmi_jobs(questions, prompts=style)
    assert pmi["instances"][0]["context"] == f"{questions[0].header(2)}\n"

    # The style's stop tokens reach the job, as they reach the local engine; a chat style's prompts are
    # conversations, which the serving container's engine writes in the model's template
    chat = ChatPrompts(concept_request="?", first_request="Write one.", second_request="", end_of_turn="<turn|>")
    concept = prepare_concept_jobs(questions, prompts=chat)
    assert concept["parameters"]["complete_prompts"]["stop_tokens"] == [".", ",", "<turn|>"]
    assert concept["instances"][0]["text"] == chat.concept(questions[0])
    assert prepare_pmi_jobs(questions, prompts=chat)["instances"][0]["context"] == [{"role": "user", "content": "Write one."}]


def test_any_grid_becomes_scoring_jobs_in_its_flat_order():
    from kcluster.core.grid import ScoringGrid
    from kcluster.engine.vertex import prepare_scoring_jobs

    grid = ScoringGrid([" a.", " b."], 3, context=lambda row, col: f"q{row}", marginal="none")
    jobs = prepare_scoring_jobs(grid, configs={"pad_to_multiple_of": 16})
    assert [item["id"] for item in jobs["instances"]] == [f"pmi-{i}" for i in range(8)]
    assert [(item["context"], item["text"]) for item in jobs["instances"]] == [grid[i] for i in range(len(grid))]
    assert jobs["parameters"] == {"log_prob": {"pad_to_multiple_of": 16}}


def test_prepare_inputs_uploads_instances_and_questions(gcs, make_questions):
    questions = make_questions(2)
    instances, parameters = prepare_inputs(questions, "job1", CONFIG)

    store = gcs.bucket("my-bucket").store
    uploaded = [json.loads(line) for line in store["batch-input/job1/instances.jsonl"].splitlines()]
    assert uploaded == instances
    assert {item["purpose"] for item in uploaded} == {"complete_prompts", "log_prob"}
    assert set(parameters) == {"complete_prompts", "log_prob"}

    q_lines = [json.loads(line) for line in store["batch-input/job1/questions.jsonl"].splitlines()]
    assert [q["id"] for q in q_lines] == ["q-0", "q-1"]


# --- prediction collection ---

def _fill_predictions(store, job_id, n, concepts):
    lines = []
    for i, c in enumerate(concepts):
        lines.append(json.dumps({"instance": {"id": f"concept-{i}"}, "prediction": c}))
    for i in range(n * n + n):
        lines.append(json.dumps({"instance": {"id": f"pmi-{i}"}, "prediction": -float(i + 1)}))
    # Split across two shards to exercise the multi-file path
    store[f"batch-output/{job_id}/prediction.results-00000-of-00002"] = "\n".join(lines[:2])
    store[f"batch-output/{job_id}/prediction.results-00001-of-00002"] = "\n".join(lines[2:])
    store[f"batch-output/{job_id}/irrelevant.txt"] = "ignore me"


def test_collect_predictions(gcs):
    store = gcs.bucket("my-bucket").store
    _fill_predictions(store, "job1", 2, [" algebra. ", "geometry,"])

    results = collect_predictions("job1", 2, CONFIG)

    assert results["concept"] == ["algebra", "geometry"]  # stripped of whitespace and trailing punctuation

    from kcluster.core.prompts import ChatPrompts
    _fill_predictions(store, "job1-chat", 2, [" **algebra**. ", '"geometry",'])
    chat = ChatPrompts(concept_request="?", first_request="", second_request="")
    assert collect_predictions("job1-chat", 2, CONFIG, prompts=chat)["concept"] == ["algebra", "geometry"]
    assert results["pmi"].shape == (3, 2)  # [marginals; conditionals] stack
    assert results["pmi"][0].tolist() == [-1.0, -2.0]

    # The collected artifacts are uploaded back to the job's output prefix
    concepts = [json.loads(line)["concept"] for line in store["batch-output/job1/concepts.jsonl"].splitlines()]
    assert concepts == ["algebra", "geometry"]
    assert np.load(io.BytesIO(store["batch-output/job1/pmi.npy"])).shape == (3, 2)

    # And they round-trip through the download helpers
    assert vertex.download_concepts("job1", CONFIG) == ["algebra", "geometry"]
    assert vertex.download_pmi("job1", CONFIG).shape == (3, 2)
    assert vertex.download_pmi("no-such-job", CONFIG) is None


def test_collect_predictions_takes_the_grids_shape(gcs):
    store = gcs.bucket("my-bucket").store
    lines = [json.dumps({"instance": {"id": f"concept-{i}"}, "prediction": c}) for i, c in enumerate(["x", "y", "z"])]
    lines += [json.dumps({"instance": {"id": f"pmi-{i}"}, "prediction": -float(i)}) for i in range(3 * 2 + 2)]
    store["batch-output/job2/prediction.results-00000-of-00001"] = "\n".join(lines)

    results = collect_predictions("job2", 3, CONFIG, shape=(3, 2))
    assert results["concept"] == ["x", "y", "z"]
    assert results["pmi"].shape == (4, 2) and results["pmi"][1].tolist() == [-2.0, -3.0]


def test_collect_predictions_rejects_incomplete_results(gcs):
    store = gcs.bucket("my-bucket").store
    lines = [json.dumps({"instance": {"id": "concept-0"}, "prediction": "algebra"})]
    store["batch-output/job1/prediction.results-00000-of-00001"] = "\n".join(lines)

    with pytest.raises(AssertionError):
        collect_predictions("job1", 1, CONFIG)


# --- job launch ---

def test_launch_batch_job(gcs, monkeypatch, make_questions):
    captured = {}

    class FakeJob:
        _gca_resource = SimpleNamespace(name="jobs/1")
        resource_name = "jobs/1"

    class FakeModel:
        def __init__(self, resource):
            captured["model"] = resource

        def batch_predict(self, **kwargs):
            captured["batch_predict"] = kwargs
            return FakeJob()

    monkeypatch.setattr(vertex.aiplatform, "init", lambda **kw: captured.update(init=kw))
    monkeypatch.setattr(vertex.aiplatform, "Model", FakeModel)

    job, job_id = launch_batch_job(make_questions(1), CONFIG, job_id="job1", batch_size=1,
                                   completion_time_in_mins=1, secs_per_batch=30)

    assert job_id == "job1"
    assert captured["init"] == {"project": "my-project", "location": "us-central1"}
    assert captured["model"] == CONFIG.model_resource
    kwargs = captured["batch_predict"]
    assert kwargs["gcs_source"] == "gs://my-bucket/batch-input/job1/instances.jsonl"
    assert kwargs["gcs_destination_prefix"] == "gs://my-bucket/batch-output/job1"
    # 3 instances (1 concept + 2 pmi) at batch_size 1 within 1 min at 30 s/batch -> 2 replicas
    assert kwargs["starting_replica_count"] == 2
    assert kwargs["max_replica_count"] == 2
    assert kwargs["machine_type"] == "g2-standard-4"


def test_wait_for_job_completion_collects_succeeded_jobs(monkeypatch):
    from google.cloud.aiplatform_v1 import JobState

    collected = []
    monkeypatch.setattr(vertex, "collect_predictions",
                        lambda job_id, n, config: collected.append(job_id) or
                        {"concept": [], "pmi": np.zeros((1, 1))})
    monkeypatch.setattr(vertex.time, "sleep", lambda s: None)

    jobs = [
        {"job_id": "ok", "num_questions": 1,
         "job_obj": SimpleNamespace(name="j1", display_name="ok", state=JobState.JOB_STATE_SUCCEEDED)},
        {"job_id": "bad", "num_questions": 1,
         "job_obj": SimpleNamespace(name="j2", display_name="bad", state=JobState.JOB_STATE_FAILED,
                                    error="boom")},
    ]
    vertex.wait_for_job_completion(jobs, CONFIG, poll_interval_seconds=0)
    assert collected == ["ok"]  # succeeded jobs are collected; failed ones just terminate


def test_resolve_input_files(tmp_path):
    (tmp_path / "b.jsonl").write_text("{}")
    (tmp_path / "a.jsonl").write_text("{}")
    (tmp_path / "notes.txt").write_text("")

    assert [f.split("/")[-1] for f in vertex.resolve_input_files(str(tmp_path))] == ["a.jsonl", "b.jsonl"]
    assert vertex.resolve_input_files(str(tmp_path / "a.jsonl")) == [str(tmp_path / "a.jsonl")]
    with pytest.raises(FileNotFoundError):
        vertex.resolve_input_files(str(tmp_path / "missing"))
