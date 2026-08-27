import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from kcluster.core.pmi import PointwiseMutualInfo
from kcluster.io.jsonl import load_questions
from kcluster.paths import concept_kc_path, kc_dir, pmi_dir, pmi_raw_dir, prepare_output_dir, run_dir
from kcluster.tasks.cluster import build_kcluster_models


def main(args):
    # The result dir is both input and output (D10): the concept step already
    # wrote kc/<ds>_concept-kc.csv and mat/pmi/raw/ holds the score shards.
    result_dir = getattr(args, "result_dir", None) or run_dir(getattr(args, "run_dir", None))
    if not result_dir:
        raise SystemExit("--result_dir is required unless --run_dir (or KCLUSTER_RUN_DIR) is set")
    args.result_dir = result_dir = os.path.abspath(result_dir)
    print(f"*** Writing results to {result_dir} ***")

    # Check the Concept KC written by the concept step is correctly filled
    concept_path = concept_kc_path(result_dir)
    ds = os.path.basename(concept_path).removesuffix("_concept-kc.csv")
    concept_df = pd.read_csv(concept_path)
    assert concept_df["KC"].str.strip().all(), "Some concepts are invalid"
    out_dir = prepare_output_dir(kc_dir(result_dir, "kcluster"))

    # Recover the questions behind the concepts. --data_path overrides the
    # recorded one, as in the embed command: a result dir is often rebuilt on a
    # different machine than it was scored on (a cluster run pulled back to a
    # laptop records a path that does not exist there).
    if not getattr(args, "data_path", None):
        [fname] = glob.glob("args-concept-*.json", root_dir=result_dir)
        with open(os.path.join(result_dir, fname), "r") as f:
            args.data_path = json.load(f)["data_path"]
    if not os.path.isfile(args.data_path):
        raise SystemExit(f"Question file not found: {args.data_path} — pass --data_path with a "
                         "reachable copy of this dataset's questions")
    questions = load_questions(args.data_path)

    # The KCluster KC models from the raw score shards: every congruity
    # estimator x format correction the flags ask for (tasks.cluster).
    raw_dir = getattr(args, "pmi_dir", None) or pmi_raw_dir(result_dir)
    if os.path.isdir(raw_dir) and glob.glob("predictions_*.pt", root_dir=raw_dir):
        num_questions = len(questions)
        mat_dir = prepare_output_dir(pmi_dir(result_dir))

        def congruity(normalize: bool) -> np.ndarray:
            # Reassembled per estimator: the shards are small, and the object
            # has nothing else to give once its matrix is out.
            return PointwiseMutualInfo.from_shards(raw_dir, num_questions, num_questions,
                                                   normalize=normalize, symmetric=True).pmi_mat

        build_kcluster_models(concept_df, questions, congruity, ds=ds, kc_out_dir=out_dir, mat_out_dir=mat_dir,
                              normalize=getattr(args, "normalize", False),
                              residualize=getattr(args, "residualize", False),
                              residualize_full=getattr(args, "residualize_full", False))

    # Save arguments
    with open(os.path.join(result_dir, f"args-kc-{ds}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def add_arguments(parser):
    parser.add_argument("--result_dir", default=argparse.SUPPRESS, type=str,
                        help="The result directory holding the concept step's output (default: --run_dir)")
    parser.add_argument("--pmi_dir", default=argparse.SUPPRESS, type=str,
                        help="Directory of raw score shards (default: <result_dir>/mat/pmi/raw)")
    parser.add_argument("--data_path", default=argparse.SUPPRESS, type=str,
                        help="Question file (default: the path recorded in args-concept-*.json)")
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
    parser.add_argument("--run_dir", default=argparse.SUPPRESS, type=str,
                        help="Result folder shared by every step of this run (env: KCLUSTER_RUN_DIR)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
