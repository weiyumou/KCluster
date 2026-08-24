"""Vertex AI batch-prediction engine (bring-your-own-project).

Prepares KCluster jobs (concept extraction + the congruity grid) as JSONL
instances in Google Cloud Storage, launches them as Vertex AI batch
prediction jobs against a deployed KCluster model (see ``deploy/vertex/``),
and collects the predictions back into concepts and the stacked
``[marginals; conditionals]`` array that ``PointwiseMutualInfo.from_array``
consumes.

No GCP identifiers are baked in: every function takes a ``VertexConfig``,
loaded from a TOML file and/or ``KCLUSTER_VERTEX_*`` environment variables,
so the pipeline runs in the user's own project, bucket, and model registry.
"""

import dataclasses
import glob
import json
import logging
import math
import os
import time
import tomllib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from google.cloud import aiplatform, storage
from google.cloud.aiplatform_v1 import JobState

from kcluster.core.prompts import concept_prompt
from kcluster.core.question import Question
from kcluster.tasks.congruity import PairQuestion

logger = logging.getLogger(__name__)

_ENV_PREFIX = "KCLUSTER_VERTEX_"
_REQUIRED = ("project", "bucket", "model_id")


@dataclass
class VertexConfig:
    """Where to run: the user's project, bucket, and registered model."""

    project: str
    bucket: str
    model_id: str
    location: str = "us-central1"
    model_version: str = "1"
    input_prefix: str = "batch-input"
    input_name: str = "instances.jsonl"
    output_prefix: str = "batch-output"
    machine_type: str = "g2-standard-4"
    accelerator_type: str = "NVIDIA_L4"
    accelerator_count: int = 1

    @property
    def model_resource(self) -> str:
        return f"projects/{self.project}/locations/{self.location}/models/{self.model_id}@{self.model_version}"

    @classmethod
    def load(cls, path: str | None = None) -> "VertexConfig":
        """Load from a TOML file's ``[vertex]`` table, then apply env overrides.

        The file is ``path`` if given, else the ``KCLUSTER_VERTEX_CONFIG``
        environment variable if set. Every field can be overridden (or
        supplied entirely) by a ``KCLUSTER_VERTEX_<FIELD>`` environment
        variable, e.g. ``KCLUSTER_VERTEX_PROJECT``.
        """
        values = {}
        path = path or os.environ.get(_ENV_PREFIX + "CONFIG")
        if path:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            values.update(data.get("vertex", data))

        field_names = [f.name for f in dataclasses.fields(cls)]
        if unknown := set(values) - set(field_names):
            raise ValueError(f"{path}: unknown vertex config keys: {sorted(unknown)}")

        for name in field_names:
            if (env := os.environ.get(_ENV_PREFIX + name.upper())) is not None:
                values[name] = int(env) if name == "accelerator_count" else env

        if missing := [name for name in _REQUIRED if not values.get(name)]:
            raise ValueError(
                f"missing vertex config values: {missing} — set them in a TOML file "
                f"(passed explicitly or via {_ENV_PREFIX}CONFIG) or as "
                f"{_ENV_PREFIX}<NAME> environment variables"
            )
        return cls(**values)


def init(config: VertexConfig) -> None:
    """Point the aiplatform SDK at the configured project and location."""
    aiplatform.init(project=config.project, location=config.location)


def prepare_concept_jobs(questions: list[Question], verbal: bool = False, configs: dict | None = None) -> dict:
    PURPOSE = "complete_prompts"

    # Prepare concept-specific parameters, which can be overridden by configs
    configs = configs or dict()
    parameters = {
        PURPOSE: {"stop_tokens": [".", ","], "do_sample": False, "pad_to_multiple_of": 8,
                  "max_new_tokens": 20, "num_beams": 5, "length_penalty": -0.1, **configs}
    }

    # Prepare the content of each instance (the same prompt the local
    # concept task builds)
    instances = []
    for idx, q in enumerate(questions):
        instances.append(
            {"id": f"concept-{idx}", "text": concept_prompt(q, verbal=verbal),
             "purpose": PURPOSE, "config": parameters[PURPOSE]}
        )

    return {"instances": instances, "parameters": parameters}


def prepare_pmi_jobs(questions: list[Question], configs: dict | None = None) -> dict:
    # Prepare PMI-specific parameters, which can be overridden by configs
    PURPOSE = "log_prob"

    configs = configs or dict()
    parameters = {
        PURPOSE: {"pad_to_multiple_of": 8, **configs}
    }

    # Prepare the content of each instance, sharing the local engine's
    # scoring grid (marginals under the "Exercise 2:" header)
    ds = PairQuestion(questions)
    instances = []
    for idx in range(len(ds)):
        context, text = ds[idx]
        instances.append(
            {"id": f"pmi-{idx}", "text": text, "context": context, "purpose": PURPOSE, "config": parameters[PURPOSE]}
        )

    return {"instances": instances, "parameters": parameters}


def prepare_inputs(questions: list[Question], job_id: str, config: VertexConfig,
                   verbal_concepts: bool = False, concept_configs: dict | None = None,
                   pmi_configs: dict | None = None) -> tuple[list[dict], dict]:
    instances, parameters = [], {}
    # Prepare instances for concept extraction
    concept_jobs = prepare_concept_jobs(questions, verbal=verbal_concepts, configs=concept_configs)
    instances.extend(concept_jobs["instances"])
    parameters.update(concept_jobs["parameters"])

    # Prepare instances for PMI computation
    pmi_jobs = prepare_pmi_jobs(questions, configs=pmi_configs)
    instances.extend(pmi_jobs["instances"])
    parameters.update(pmi_jobs["parameters"])

    # Upload the instances to Google Cloud Storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(config.bucket)
    input_path = f"{config.input_prefix}/{job_id}/{config.input_name}"
    with bucket.blob(input_path).open("w") as f:
        f.write("\n".join(json.dumps(item) for item in instances))
    logger.info(f"{len(instances)} input instances are uploaded to gs://{config.bucket}/{input_path}")

    # Also upload the questions to GCS
    q_path = f"{config.input_prefix}/{job_id}/questions.jsonl"
    with bucket.blob(q_path).open("w") as f:
        f.write("\n".join(json.dumps(q.data) for q in questions))
    logger.info(f"{len(questions)} questions are uploaded to gs://{config.bucket}/{q_path}")

    return instances, parameters


def collect_predictions(job_id: str, num_questions: int, config: VertexConfig):
    nrows = ncols = num_questions
    results = {"concept": [None] * num_questions,
               "pmi": np.full((nrows * ncols + ncols,), np.inf, dtype=float)}

    storage_client = storage.Client()
    blobs = storage_client.list_blobs(config.bucket, prefix=f"{config.output_prefix}/{job_id}")
    for blob in blobs:
        if os.path.basename(blob.name).startswith("prediction.results"):
            with blob.open("r") as f:
                for line in f:
                    item = json.loads(line)
                    p, idx = item["instance"]["id"].split("-")
                    results[p][int(idx)] = item["prediction"]

    assert all(results["concept"]), "Not all concepts were extracted successfully."
    assert np.isfinite(results["pmi"]).all(), "Some PMI values are missing or infinite."

    bucket = storage_client.bucket(config.bucket)

    # Save the concepts to a jsonl file in GCS
    results["concept"] = [c.strip().rstrip(".,") for c in results["concept"]]
    concept_blob = bucket.blob(f"{config.output_prefix}/{job_id}/concepts.jsonl")
    with concept_blob.open("w") as f:
        f.write("\n".join(json.dumps({"concept": c}) for c in results["concept"]))

    # Save the PMI matrix to a numpy file in GCS
    results["pmi"] = results["pmi"].reshape(-1, ncols)
    pmi_blob = bucket.blob(f"{config.output_prefix}/{job_id}/pmi.npy")
    with pmi_blob.open("wb") as f:
        np.save(f, results["pmi"])

    return results


def download_pmi(job_id: str, config: VertexConfig) -> np.ndarray | None:
    """Download an already-collected pmi.npy from GCS, or None if it does not exist."""
    blob = storage.Client().bucket(config.bucket).blob(f"{config.output_prefix}/{job_id}/pmi.npy")
    if not blob.exists():
        return None
    with blob.open("rb") as f:
        return np.load(f)


def download_concepts(job_id: str, config: VertexConfig) -> list[str] | None:
    """Download the collected concepts from GCS, or None if they do not exist."""
    blob = storage.Client().bucket(config.bucket).blob(f"{config.output_prefix}/{job_id}/concepts.jsonl")
    if not blob.exists():
        return None
    with blob.open("r") as f:
        return [json.loads(line)["concept"] for line in f]


def collected_inputs(jobs_path: str, config: VertexConfig) -> dict[str, str]:
    """Map each input file in a job log to the job that already has results.

    Keyed on the *result*, not on the launch: a job that was started but failed,
    was cancelled, or is still running does not count, so a resumed launch
    reruns it and a build ignores it. Later entries win, so relaunching a course
    supersedes its earlier attempt.
    """
    if not os.path.exists(jobs_path):
        return {}
    collected = {}
    with open(jobs_path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if download_pmi(item["job_id"], config) is not None:
                collected[item["data_path"]] = item["job_id"]
    logging.info(f"{len(collected)} input file(s) already have collected results in {jobs_path}")
    return collected


def launch_batch_job(questions: list[Question], config: VertexConfig,
                     job_id: str | None = None, job_name: str | None = None, batch_size: int = 16,
                     starting_replica_count: int | None = None, max_replica_count: int | None = None,
                     completion_time_in_mins: int | None = None, secs_per_batch: int | None = None, **kwargs):
    init(config)

    # Prepare the inputs
    job_id = job_id or uuid.uuid4().hex
    job_name = job_name or job_id
    instances, parameters = prepare_inputs(questions, job_id, config, **kwargs)

    # Calculate the number of replicas based on the completion time and batch size
    num_batches = math.ceil(len(instances) / batch_size)
    if starting_replica_count is None:
        if completion_time_in_mins and secs_per_batch:
            starting_replica_count = max(1, math.ceil(num_batches / (completion_time_in_mins * 60 / secs_per_batch)))
        else:
            starting_replica_count = 1
    if max_replica_count is None:
        max_replica_count = starting_replica_count
    max_replica_count = max(max_replica_count, starting_replica_count)

    logger.info((
        f"Job ID: {job_id}, Batch size: {batch_size}, "
        f"Starting replicas: {starting_replica_count}, Max replicas: {max_replica_count}"))

    # Create a batch job
    model = aiplatform.Model(config.model_resource)
    input_path = f"gs://{config.bucket}/{config.input_prefix}/{job_id}/{config.input_name}"
    output_path = f"gs://{config.bucket}/{config.output_prefix}/{job_id}"

    job = model.batch_predict(
        job_display_name=job_name,
        gcs_source=input_path,
        gcs_destination_prefix=output_path,
        instances_format="jsonl",
        predictions_format="jsonl",
        model_parameters=parameters,
        machine_type=config.machine_type,
        accelerator_count=config.accelerator_count,
        accelerator_type=config.accelerator_type,
        starting_replica_count=starting_replica_count,
        max_replica_count=max_replica_count,
        sync=False,
        batch_size=batch_size,
    )

    # Wait for the job to be created
    while not getattr(job._gca_resource, "name", None):
        logger.info(f"Waiting for job '{job_name}' to be created.")
        time.sleep(5)

    return job, job_id


def job_rate(job, num_instances: int | None) -> str:
    """A throughput summary for a finished job, for sizing the next one.

    Prefers Vertex's own start/end timestamps over wall clock: a job can queue
    for a long time before a replica starts, and queueing is not throughput.
    Returns an empty string when the resource carries no usable times.
    """
    start, end = getattr(job, "start_time", None), getattr(job, "end_time", None)
    if not (start and end):
        return ""
    seconds = (end - start).total_seconds()
    if seconds <= 0:
        return ""
    summary = f"ran {seconds / 60:.1f} min"
    if num_instances:
        summary += f" for {num_instances:,} instances = {num_instances / seconds:,.0f} instances/s"
    return summary


def instances_done(job) -> int:
    """How many instances a running job has predicted so far, per Vertex.

    ``completion_stats.successful_count`` advances while the job runs, which is
    the only progress signal available without reading Cloud Logging: the GCS
    output is written in one go at the end, so an empty output prefix says
    nothing. A job that has not started dispatching leaves ``completion_stats``
    unset entirely, which reads as 0 here.
    """
    stats = getattr(job._gca_resource, "completion_stats", None)
    return int(getattr(stats, "successful_count", 0) or 0) if stats else 0


# A replica can spend ~15 min being provisioned and a couple more loading the
# model before the first instance is dispatched, all of it inside the RUNNING
# state. The default leaves generous room past that: a job that has been
# RUNNING for half an hour with a zero counter is not slow, it is stuck (the
# observed failure mode is a healthy replica that is never sent any work, e.g.
# when the region cannot place the rest of the fleet).
DEFAULT_STALL_AFTER_SECONDS = 30 * 60


def wait_for_job_completion(launched_jobs: list[dict], config: VertexConfig, poll_interval_seconds: int = 60,
                            stall_after_seconds: int = DEFAULT_STALL_AFTER_SECONDS):
    """
    Waits for a list of Vertex AI Batch Prediction Jobs to complete.

    Warns about jobs whose predicted-instance count has not moved for
    ``stall_after_seconds``. Warns rather than cancels: a stalled job is the
    caller's money to spend, and a genuinely slow job must not be killed by a
    heuristic. Pass 0 to disable the check.
    """
    completed_jobs = set()
    progress = {}  # job name -> (instances done, monotonic time it last moved)
    stalled_warned = set()
    while len(completed_jobs) < len(launched_jobs):
        logger.info(
            f"Waiting for {len(launched_jobs) - len(completed_jobs)} jobs to finish. "
            f"Polling again in {poll_interval_seconds} seconds..."
        )
        time.sleep(poll_interval_seconds)

        for item in launched_jobs:
            job = item["job_obj"]
            if job.name in completed_jobs:
                continue  # Already completed

            job_display_name = job.display_name
            match (state := job.state):
                case JobState.JOB_STATE_PARTIALLY_SUCCEEDED:
                    logger.warning(f"Job '{job_display_name}' partially succeeded. Some results may be missing.")
                case JobState.JOB_STATE_SUCCEEDED:
                    # The rate is what sizes the next launch, so report it here
                    # rather than leaving it to be dug out of the console.
                    rate = job_rate(job, item.get("num_instances"))
                    logger.info(f"Job '{job_display_name}' succeeded{': ' + rate if rate else ''}.")
                    try:
                        results = collect_predictions(item["job_id"], item["num_questions"], config)
                        logger.info(
                            f"Collected {len(results['concept'])} concepts "
                            f"and PMI matrix of shape {results['pmi'].shape} for job '{item['job_id']}'")
                    except Exception as e:
                        logger.error(f"Failed to collect predictions for job '{item['job_id']}': {e}")
                case JobState.JOB_STATE_FAILED:
                    logger.error(f"Job '{job_display_name}' failed. Error: {job.error}")
                case JobState.JOB_STATE_CANCELLED:
                    logger.warning(f"Job '{job_display_name}' was cancelled.")
                case _:
                    done = instances_done(job)
                    total = item.get("num_instances")
                    pct = f" ({done / total:.1%})" if total else ""
                    logger.info(f"Job '{job_display_name}' is still running ({state}); "
                                f"{done:,} instances done{pct}")

                    # Progress is per job and measured from the last time the
                    # counter actually moved, so a job that dispatches and then
                    # dies mid-run is caught as well as one that never starts.
                    seen, since = progress.get(job.name, (-1, time.monotonic()))
                    if done > seen:
                        progress[job.name] = (done, time.monotonic())
                        stalled_warned.discard(job.name)
                    elif stall_after_seconds:
                        stalled_for = time.monotonic() - since
                        progress.setdefault(job.name, (done, since))
                        if stalled_for > stall_after_seconds and job.name not in stalled_warned:
                            stalled_warned.add(job.name)
                            logger.warning(
                                f"*** STALLED: '{job_display_name}' has been running with "
                                f"{done:,} instances done for {stalled_for / 60:.0f} min. A healthy job "
                                f"starts dispatching within ~20 min of launch. Check the container log for "
                                f"'POST /predict'; if there is none, the region likely could not place the "
                                f"fleet — cancel this job and relaunch with fewer concurrent jobs or "
                                f"replicas. Job: {job.resource_name}")
                    continue
            # Add the job to completed jobs if it has reached a terminal state
            completed_jobs.add(job.name)

    logger.info("All batch prediction jobs have completed.")


def resolve_input_files(data_path: str | Sequence[str]) -> list[str]:
    """Resolve one or more input paths to a sorted, de-duplicated list of files.

    Each path is either a directory (all its ``*.jsonl`` files) or a single
    file. Several may be given to select a subset of a workspace by name --
    they must be launched from ONE call, because the caller launches every
    resolved file before it starts waiting; separate calls would run serially.
    """
    paths = [data_path] if isinstance(data_path, str) else list(data_path)
    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(glob.iglob(os.path.join(path, "*.jsonl")))
        elif os.path.isfile(path):
            files.append(path)
        else:
            raise FileNotFoundError(f"No such file or directory: {path}")
    return sorted(set(files))
