import glob
import os

import numpy as np
import pandas as pd

from kcluster.core.pmi import PointwiseMutualInfo
from kcluster.io.jsonl import load_questions
from kcluster.paths import (
    concept_kc_path,
    kc_dir,
    pmi_dir,
    pmi_raw_dir,
    prepare_output_dir,
    question_path,
    required_result_dir,
    result_dataset,
    save_args,
)
from kcluster.tasks.cluster import build_kcluster_models


def main(args):
    # The result dir is both input and output (D10): the concept step already
    # wrote kc/<ds>_concept-kc.csv and mat/pmi/raw/ holds the score shards.
    args.result_dir = result_dir = required_result_dir(args.result_dir, args.run_dir)
    print(f"*** Writing results to {result_dir} ***")

    # Check the Concept KC written by the concept step is correctly filled
    ds = result_dataset(result_dir)
    concept_df = pd.read_csv(concept_kc_path(result_dir))
    assert concept_df["KC"].str.strip().all(), "Some concepts are invalid"
    out_dir = prepare_output_dir(kc_dir(result_dir, "kcluster"))

    # Recover the questions behind the concepts (--data_path overrides the recorded path)
    args.data_path = question_path(result_dir, ds, args.data_path)
    questions = load_questions(args.data_path)

    # The KCluster KC models from the raw score shards: every congruity
    # estimator x format correction the flags ask for (tasks.cluster).
    raw_dir = args.pmi_dir or pmi_raw_dir(result_dir)
    if os.path.isdir(raw_dir) and glob.glob("predictions_*.pt", root_dir=raw_dir):
        num_questions = len(questions)
        mat_dir = prepare_output_dir(pmi_dir(result_dir))

        def congruity(normalize: bool) -> np.ndarray:
            # Reassembled per estimator: the shards are small, and the object
            # has nothing else to give once its matrix is out.
            return PointwiseMutualInfo.from_shards(raw_dir, num_questions, num_questions,
                                                   normalize=normalize, symmetric=True).pmi_mat

        build_kcluster_models(concept_df, questions, congruity, ds=ds, kc_out_dir=out_dir, mat_out_dir=mat_dir,
                              normalize=args.normalize, residualize=args.residualize,
                              residualize_full=args.residualize_full)

    save_args(result_dir, "kc", ds, vars(args))


def add_arguments(parser):
    parser.add_argument("--result_dir", default=None, type=str,
                        help="The result directory holding the concept step's output (default: --run_dir)")
    parser.add_argument("--pmi_dir", default=None, type=str,
                        help="Directory of raw score shards (default: <result_dir>/mat/pmi/raw)")
    parser.add_argument("--data_path", default=None, type=str,
                        help="Question file (default: the path recorded in args-*.json)")
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
    parser.add_argument("--run_dir", default=None, type=str,
                        help="Result folder shared by every step of this run (env: KCLUSTER_RUN_DIR)")
