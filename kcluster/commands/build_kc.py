import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from kcluster.core.pmi import PointwiseMutualInfo, residualize
from kcluster.io.jsonl import load_questions
from kcluster.paths import kc_dir, pmi_dir, pmi_raw_dir, prepare_output_dir, run_dir
from kcluster.tasks.cluster import create_kc


def main(args):
    # The result dir is both input and output (D10): the concept step already
    # wrote kc/<ds>_concept-kc.csv and mat/pmi/raw/ holds the score shards.
    result_dir = getattr(args, "result_dir", None) or run_dir(getattr(args, "run_dir", None))
    if not result_dir:
        raise SystemExit("--result_dir is required unless --run_dir (or KCLUSTER_RUN_DIR) is set")
    args.result_dir = result_dir = os.path.abspath(result_dir)
    print(f"*** Writing results to {result_dir} ***")

    # Check the Concept KC written by the concept step is correctly filled
    out_dir = prepare_output_dir(kc_dir(result_dir))
    match = glob.glob("*_concept-kc.csv", root_dir=out_dir)
    if len(match) != 1:
        raise SystemExit(f"Expected exactly one *_concept-kc.csv in {out_dir}, found {len(match)} — "
                         "run the concept step into this result dir first")
    [fname] = match
    ds = fname.removesuffix("_concept-kc.csv")
    concept_df = pd.read_csv(os.path.join(out_dir, fname))
    assert concept_df["KC"].str.strip().all(), "Some concepts are invalid"

    # Recover the questions behind the concepts
    [fname] = glob.glob("args-concept-*.json", root_dir=result_dir)
    with open(os.path.join(result_dir, fname), "r") as f:
        args.data_path = json.load(f)["data_path"]
    questions = load_questions(args.data_path)

    # Create KC for KCluster-PMI from the raw score shards
    raw_dir = getattr(args, "pmi_dir", None) or pmi_raw_dir(result_dir)
    if os.path.isdir(raw_dir) and glob.glob("predictions_*.pt", root_dir=raw_dir):
        num_questions = len(questions)
        pmi = PointwiseMutualInfo.from_shards(raw_dir, num_questions, num_questions,
                                              normalize=False, symmetric=True)
        # The assembled congruity matrix is what a pairwise analysis of these
        # questions should read, so it is saved beside the models rather than
        # left recoverable only by re-reading the shards (parity with vertex).
        mat_dir = prepare_output_dir(pmi_dir(result_dir))
        np.save(os.path.join(mat_dir, f"{ds}_pmi-unnorm.npy"), pmi.pmi_mat)

        print("*** Building KCs for KCluster-PMI ***")
        kc = create_kc(concept_df, questions, pmi.pmi_mat)
        if isinstance(kc, pd.DataFrame):
            kc.to_csv(os.path.join(out_dir, f"{ds}_kcluster-unnorm-kc.csv"), index=False)
            print(f"*** Finished with {kc['KC'].nunique()} KCs ***")

        # An additional KC model with the question-format nuisance divided out.
        # Written alongside the plain one rather than replacing it: whether the
        # correction helps is an empirical question per dataset, and on a
        # single-format bank it is a no-op the comparison should show.
        if getattr(args, "residualize", False):
            print("*** Building KCs for KCluster-PMI, residualized by question type ***")
            adjusted = residualize(pmi.pmi_mat, [q.q_type for q in questions])
            np.save(os.path.join(mat_dir, f"{ds}_pmi-unnorm-resid.npy"), adjusted)
            kc = create_kc(concept_df, questions, adjusted)
            if isinstance(kc, pd.DataFrame):
                kc.to_csv(os.path.join(out_dir, f"{ds}_kcluster-unnorm-resid-kc.csv"), index=False)
                print(f"*** Finished with {kc['KC'].nunique()} KCs ***")

    # Save arguments
    with open(os.path.join(result_dir, f"args-kc-{ds}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def add_arguments(parser):
    parser.add_argument("--result_dir", default=argparse.SUPPRESS, type=str,
                        help="The result directory holding the concept step's output (default: --run_dir)")
    parser.add_argument("--pmi_dir", default=argparse.SUPPRESS, type=str,
                        help="Directory of raw score shards (default: <result_dir>/mat/pmi/raw)")
    parser.add_argument("--residualize", action="store_true",
                        help="Also build a KC model from congruity residualized by question type, "
                             "which stops a mixed-format bank from clustering by format")
    parser.add_argument("--run_dir", default=argparse.SUPPRESS, type=str,
                        help="Result folder shared by every step of this run (env: KCLUSTER_RUN_DIR)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
