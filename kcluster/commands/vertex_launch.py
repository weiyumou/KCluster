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
    DEFAULT_STALL_AFTER_SECONDS,
    VertexConfig,
    download_pmi,
    launch_batch_job,
    resolve_input_files,
    wait_for_job_completion,
)
from kcluster.io.jsonl import load_questions
from kcluster.paths import default_output_dir, prepare_output_dir, timestamp


def collected_inputs(jobs_path: str, config: VertexConfig) -> dict[str, str]:
    """Map each input file in a job log to the job that already has results.

    Keyed on the *result*, not on the launch: a job that was started but failed,
    was cancelled, or is still running does not count, so a resumed launch
    reruns it. Later entries win, so relaunching a course supersedes its
    earlier attempt.
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


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = VertexConfig.load(getattr(args, "config", None))

    output_dir = getattr(args, "output_dir", None) or default_output_dir("vertex-launch", getattr(args, "run_dir", None))
    args.output_dir = prepare_output_dir(output_dir)

    jobs_path = os.path.join(args.output_dir, "launched_jobs.jsonl")
    done = collected_inputs(jobs_path, config) if getattr(args, "skip_completed", False) else {}

    launched_jobs = []
    for data_path in resolve_input_files(args.data_path):
        if data_path in done:
            logging.info(f"Skipping {data_path}: already collected as job '{done[data_path]}'")
            continue

        questions = load_questions(data_path)
        logging.info(f"Loaded {len(questions)} questions from {data_path}")

        # Create Job ID based on the data file name and current time
        job_id = os.path.splitext(os.path.basename(data_path))[0].replace(" ", "-") + "_" + timestamp()

        logging.info(f"Launching job '{job_id}'...")
        job, _ = launch_batch_job(questions, config, job_id=job_id, batch_size=args.batch_size,
                                  completion_time_in_mins=args.completion_time, secs_per_batch=args.secs_per_batch)
        # num_instances is the concept pass plus the n^2 + n congruity grid;
        # wait_for_job_completion turns it into a throughput figure on success.
        n = len(questions)
        launched_jobs.append({"job_id": job_id, "job_obj": job, "num_questions": n,
                              "num_instances": n * n + 2 * n})
        with open(jobs_path, "a") as f:
            f.write(json.dumps({"job_id": job_id, "data_path": data_path, "resource_name": job.resource_name}) + "\n")

    if not launched_jobs:
        logging.info("Nothing to launch — every input already has collected results.")
        return
    wait_for_job_completion(launched_jobs, config,
                            stall_after_seconds=getattr(args, "stall_after", DEFAULT_STALL_AFTER_SECONDS))


def add_arguments(parser):
    parser.add_argument("--data_path", required=True, type=str, nargs="+",
                        help="One or more paths, each a directory of *.jsonl question files or a single "
                             ".jsonl file. Give several to launch a chosen subset concurrently — they must "
                             "share one invocation, since waiting starts only after all are launched")
    parser.add_argument("--completion_time", default=60.0, type=float, help="Expected completion time in minutes")
    parser.add_argument("--secs_per_batch", default=0.1, type=float, help="Estimated seconds per batch for the job")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for the job")
    parser.add_argument("--stall_after", type=int, default=DEFAULT_STALL_AFTER_SECONDS,
                        help="Warn when a running job's predicted-instance count has not moved for this "
                             "many seconds (0 disables). A healthy job starts dispatching within ~20 min")
    parser.add_argument("--skip_completed", action="store_true",
                        help="Skip inputs whose job in this output directory's launched_jobs.jsonl "
                             "already has collected results, so a partial run can be resumed "
                             "without paying for the finished courses again")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str, help="Directory to save the job log")
    parser.add_argument("--run_dir", default=argparse.SUPPRESS, type=str,
                        help="Shared run folder; each step writes to <run_dir>/<step> (env: KCLUSTER_RUN_DIR)")
    parser.add_argument("--config", default=argparse.SUPPRESS, type=str,
                        help="Path to a vertex TOML config (default: KCLUSTER_VERTEX_* environment)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
