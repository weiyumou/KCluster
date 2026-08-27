"""Build KC models (Concept + KCluster) from completed Vertex batch jobs.

Reads a ``launched_jobs.jsonl`` (as written by ``vertex-launch``) and, for
each collected job, downloads the concepts and PMI values from GCS and runs
the same clustering as the local ``build-kc`` command.
"""

import argparse
import json
import logging
import os

import numpy as np

from kcluster.core.pmi import PointwiseMutualInfo
from kcluster.engine.vertex import VertexConfig, collected_inputs, download_concepts, download_pmi
from kcluster.io.jsonl import load_questions
from kcluster.paths import kc_dir, pmi_dir, prepare_output_dir
from kcluster.tasks.cluster import build_kcluster_models, build_res_df


def buildable_jobs(jobs_path: str, config: VertexConfig) -> list[dict]:
    """The job log entries worth building, one per input file.

    A course that was relaunched after a failure appears in the log more than
    once, and only the winning attempt has a pmi.npy. Building every entry would
    stop on the first dead one, so keep just the attempt that has results.
    """
    with open(jobs_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    collected = collected_inputs(jobs_path, config)
    for item in entries:
        if collected.get(item["data_path"]) != item["job_id"]:
            logging.info(f"Skipping '{item['job_id']}': superseded or never produced results")
    return [item for item in entries if collected.get(item["data_path"]) == item["job_id"]]


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = VertexConfig.load(getattr(args, "config", None))

    jobs_path = os.path.join(args.work_dir, "launched_jobs.jsonl")
    launched_jobs = buildable_jobs(jobs_path, config)
    if not launched_jobs:
        raise SystemExit(f"No collected results in {jobs_path} — run vertex-retrieve or relaunch first")

    for item in launched_jobs:
        job_id, data_path = item["job_id"], item["data_path"]
        data_name = os.path.splitext(os.path.basename(data_path))[0].replace(" ", "-")

        # Each course gets a full result dir under the work dir (D10)
        result_dir = os.path.join(args.work_dir, data_name)
        output_dir = prepare_output_dir(kc_dir(result_dir, "kcluster"))
        concept_dir = prepare_output_dir(kc_dir(result_dir, "concept"))
        mat_dir = prepare_output_dir(pmi_dir(result_dir))

        # Load questions
        questions = load_questions(data_path)
        num_qs = len(questions)
        print(f"Loaded {num_qs} questions from '{data_path}'")

        print(f"*** Building KCs for job_id='{job_id}' ***")
        mtx = download_pmi(job_id, config)
        assert mtx is not None, \
            f"No collected pmi.npy for job '{job_id}' — run vertex-retrieve (or wait for collection) first"

        # Create the Concept KC from the collected concepts
        concepts = download_concepts(job_id, config)
        assert concepts is not None, f"No collected concepts.jsonl for job '{job_id}'"
        concept_df = build_res_df(questions, concepts)

        # The KCluster KC models: every congruity estimator x format correction
        # the flags ask for — the same builder, flags and filenames as the
        # local build-kc command (tasks.cluster).
        def congruity(normalize: bool, mtx=mtx) -> np.ndarray:
            return PointwiseMutualInfo.from_array(mtx, symmetric=True, normalize=normalize).pmi_mat

        build_kcluster_models(concept_df, questions, congruity, ds=data_name, kc_out_dir=output_dir,
                              mat_out_dir=mat_dir, normalize=getattr(args, "normalize", False),
                              residualize=getattr(args, "residualize", False),
                              residualize_full=getattr(args, "residualize_full", False))

        print(f"*** Created {concept_df['KC'].nunique()} Concept KCs ***\n\n")
        concept_df.to_csv(os.path.join(concept_dir, f"{data_name}_concept-kc.csv"), index=False)

        # Provenance breadcrumb at the course's result root, mirroring the
        # local pipeline; the embed command recovers data_path from here
        breadcrumb = dict(vars(args), job_id=job_id, data_path=data_path)
        with open(os.path.join(result_dir, f"args-kc-{data_name}.json"), "w") as f:
            json.dump(breadcrumb, f, indent=2)


def add_arguments(parser):
    parser.add_argument("--work_dir", required=True, type=str,
                        help="Path to the working directory containing launched_jobs.jsonl")
    parser.add_argument("--normalize", action="store_true",
                        help="Also build the KC models from the joint-normalized congruity "
                             "(kcluster-norm, plus its format corrections) beside the raw-estimator "
                             "kcluster-unnorm ones")
    parser.add_argument("--residualize", action="store_true",
                        help="Also build a KC model from congruity with the per-format-pair means "
                             "subtracted, which stops a mixed-format bank from clustering by format")
    parser.add_argument("--residualize_full", action="store_true",
                        help="Also build a KC model from congruity with the joint item + format "
                             "correction removed — the recommended variant for mixed-format banks; "
                             "implies --residualize")
    parser.add_argument("--config", default=argparse.SUPPRESS, type=str,
                        help="Path to a vertex TOML config (default: KCLUSTER_VERTEX_* environment)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
