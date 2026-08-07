"""Retrieve the PMI similarity matrices for completed Vertex batch jobs.

Reads a ``launched_jobs.jsonl`` (as written by ``vertex-launch``) and, for
each job that has finished, builds the PMI similarity matrix
(``PointwiseMutualInfo.pmi_mat`` under the given ``--symmetric`` /
``--normalize`` options) and saves it to a local directory. If a job
succeeded but its predictions were never collected (e.g. the launch process
died before ``wait_for_job_completion``), the raw values are collected on
the fly first.
"""

import argparse
import json
import logging
import os

import numpy as np
from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import JobState

from kcluster.core.pmi import PointwiseMutualInfo
from kcluster.engine.vertex import VertexConfig, collect_predictions, download_pmi, init


def count_questions(data_path: str) -> int:
    """Count the questions in a data file (needed to collect predictions)."""
    with open(data_path) as f:
        return sum(1 for line in f if line.strip())


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = VertexConfig.load(getattr(args, "config", None))
    init(config)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.jobs_path) as f:
        launched_jobs = [json.loads(line) for line in f]
    logging.info(f"Read {len(launched_jobs)} jobs from '{args.jobs_path}'")

    sym_tag = "sym" if args.symmetric else "asym"
    norm_tag = "norm" if args.normalize else "unnorm"

    saved = 0
    for item in launched_jobs:
        job_id, data_path, resource_name = item["job_id"], item["data_path"], item["resource_name"]

        # Prefer an already-collected matrix; it works regardless of the current job state.
        mtx = download_pmi(job_id, config)
        if mtx is None:
            state = aiplatform.BatchPredictionJob(resource_name).state
            if state != JobState.JOB_STATE_SUCCEEDED:
                logging.warning(f"Skipping '{job_id}': not collected and job state is {state.name}")
                continue
            logging.info(f"'{job_id}' succeeded but was not collected; collecting predictions...")
            try:
                mtx = collect_predictions(job_id, count_questions(data_path), config)["pmi"]
            except Exception as e:
                logging.error(f"Failed to collect predictions for '{job_id}': {e}")
                continue

        # Build the PMI similarity matrix under the requested options
        pmi = PointwiseMutualInfo.from_array(mtx, symmetric=args.symmetric, normalize=args.normalize).pmi_mat
        output_path = os.path.join(args.output_dir, f"{job_id}_pmi-{sym_tag}-{norm_tag}.npy")
        np.save(output_path, pmi)
        logging.info(f"Saved PMI matrix of shape {pmi.shape} to '{output_path}'")
        saved += 1

    logging.info(f"Retrieved {saved}/{len(launched_jobs)} PMI matrices into '{args.output_dir}'")


def add_arguments(parser):
    parser.add_argument("--jobs_path", required=True, type=str,
                        help="Path to a launched_jobs.jsonl file")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Directory to save the retrieved PMI matrices")
    parser.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=True,
                        help="Symmetrize the PMI matrix (default: enabled; use --no-symmetric to disable)")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False,
                        help="Normalize the PMI over the question set (default: disabled)")
    parser.add_argument("--config", default=argparse.SUPPRESS, type=str,
                        help="Path to a vertex TOML config (default: KCLUSTER_VERTEX_* environment)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
