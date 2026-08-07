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
import uuid
from dataclasses import dataclass

import numpy as np
from google.cloud import aiplatform, storage
from google.cloud.aiplatform_v1 import JobState

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
            try:
                import tomllib
            except ModuleNotFoundError:  # Python 3.10
                import tomli as tomllib
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
    SPACE = Question.SPACE
    PURPOSE = "complete_prompts"

    # Prepare concept-specific parameters, which can be overridden by configs
    configs = configs or dict()
    parameters = {
        PURPOSE: {"stop_tokens": [".", ","], "do_sample": False, "pad_to_multiple_of": 8,
                  "max_new_tokens": 20, "num_beams": 5, "length_penalty": -0.1, **configs}
    }

    # Determine whether the generated concept should begin with a verb
    if verbal:
        trailer = "whether the student can"  # +verbal phrase
    else:
        trailer = "whether the student understands the concept of"  # +noun phrase

    # Prepare the content of each instance
    instances = []
    for idx, q in enumerate(questions):
        q_type = q.q_type.lower().replace(SPACE, "-")
        prompt = (
            f"{q.header(1)}\n{str(q)}\n\n"
            f"Remark:\nThe above exercise is a {q_type} question that tests {trailer}"
        )
        instances.append(
            {"id": f"concept-{idx}", "text": prompt, "purpose": PURPOSE, "config": parameters[PURPOSE]}
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


def wait_for_job_completion(launched_jobs: list[dict], config: VertexConfig, poll_interval_seconds: int = 60):
    """
    Waits for a list of Vertex AI Batch Prediction Jobs to complete.
    """
    completed_jobs = set()
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
                    logger.info(f"Job '{job_display_name}' succeeded.")
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
                    logger.info(f"Job '{job_display_name}' is still running. Current state: {state}")
                    continue
            # Add the job to completed jobs if it has reached a terminal state
            completed_jobs.add(job.name)

    logger.info("All batch prediction jobs have completed.")


def resolve_input_files(data_path: str) -> list[str]:
    """Resolve an input path to a sorted list of question files.

    Accepts either a directory (all its ``*.jsonl`` files) or a single file.
    """
    if os.path.isdir(data_path):
        return sorted(glob.iglob(os.path.join(data_path, "*.jsonl")))
    if os.path.isfile(data_path):
        return [data_path]
    raise FileNotFoundError(f"No such file or directory: {data_path}")
