"""Launch Vertex AI batch jobs for concept extraction + congruity scoring.

One job per question file; the job log (``launched_jobs.jsonl``) written to
the output directory is the input to ``vertex-retrieve`` and
``vertex-build-kc``.
"""

import argparse
import json
import logging
import os

from kcluster.engine.vertex import (
    VertexConfig,
    launch_batch_job,
    resolve_input_files,
    wait_for_job_completion,
)
from kcluster.io.jsonl import load_questions
from kcluster.paths import default_output_dir, prepare_output_dir, timestamp


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = VertexConfig.load(getattr(args, "config", None))

    output_dir = getattr(args, "output_dir", None) or default_output_dir("vertex-launch")
    args.output_dir = prepare_output_dir(output_dir)

    launched_jobs = []
    for data_path in resolve_input_files(args.data_path):
        questions = load_questions(data_path)
        logging.info(f"Loaded {len(questions)} questions from {data_path}")

        # Create Job ID based on the data file name and current time
        job_id = os.path.splitext(os.path.basename(data_path))[0].replace(" ", "-") + "_" + timestamp()

        logging.info(f"Launching job '{job_id}'...")
        job, _ = launch_batch_job(questions, config, job_id=job_id, batch_size=args.batch_size,
                                  completion_time_in_mins=args.completion_time, secs_per_batch=args.secs_per_batch)
        launched_jobs.append({"job_id": job_id, "job_obj": job, "num_questions": len(questions)})
        with open(os.path.join(args.output_dir, "launched_jobs.jsonl"), "a") as f:
            f.write(json.dumps({"job_id": job_id, "data_path": data_path, "resource_name": job.resource_name}) + "\n")

    wait_for_job_completion(launched_jobs, config)


def add_arguments(parser):
    parser.add_argument("--data_path", required=True, type=str,
                        help="A directory of *.jsonl question files, or a single .jsonl file")
    parser.add_argument("--completion_time", default=60.0, type=float, help="Expected completion time in minutes")
    parser.add_argument("--secs_per_batch", default=0.1, type=float, help="Estimated seconds per batch for the job")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for the job")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str, help="Directory to save the job log")
    parser.add_argument("--config", default=argparse.SUPPRESS, type=str,
                        help="Path to a vertex TOML config (default: KCLUSTER_VERTEX_* environment)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
