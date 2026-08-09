import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from kcluster.core.pmi import PointwiseMutualInfo, residualize
from kcluster.io.jsonl import load_questions
from kcluster.paths import default_output_dir, prepare_output_dir, step_dir
from kcluster.tasks.cluster import create_kc, sim_from_embeddings


def main(args):
    run = getattr(args, "run_dir", None)
    output_dir = getattr(args, "output_dir", None) or default_output_dir("kc", run)
    args.output_dir = prepare_output_dir(output_dir)
    print(f"*** Writing results to {args.output_dir} ***")

    # Inside a run folder the previous steps' outputs are found automatically
    args.concept_dir = getattr(args, "concept_dir", None) or step_dir("concept", run)
    if not args.concept_dir:
        raise SystemExit("--concept_dir is required unless --run_dir (or KCLUSTER_RUN_DIR) is set")
    if not getattr(args, "pmi_dir", None) and (pmi := step_dir("pmi", run)) and os.path.isdir(pmi):
        args.pmi_dir = pmi

    # Check all concepts are correctly filled
    [fname] = glob.glob("*-concept.csv", root_dir=args.concept_dir)
    concept_df = pd.read_csv(os.path.join(args.concept_dir, fname))
    assert concept_df["KC"].str.strip().all(), "Some concepts are invalid"

    # Concept is already a KC model; copy it to the output directory
    concept_df.to_csv(os.path.join(args.output_dir, "concept-kc.csv"), index=False)

    # Recover the questions behind the concepts
    [fname] = glob.glob("args*.json", root_dir=args.concept_dir)
    with open(os.path.join(args.concept_dir, fname), "r") as f:
        args.data_path = json.load(f)["data_path"]
    questions = load_questions(args.data_path)

    # Determine which types of embeddings are available
    embed_types = []
    if glob.glob("*-concept-embeds.npy", root_dir=args.concept_dir):
        embed_types.append("concept")
    if glob.glob("*-question-embeds.npy", root_dir=args.concept_dir):
        embed_types.append("question")

    # Create KC for baselines
    for embed_type in embed_types:
        for metric in ("cosine",):
            [fname] = glob.glob(f"*-{embed_type}-embeds.npy", root_dir=args.concept_dir)
            embeds = np.load(os.path.join(args.concept_dir, fname))
            assert embeds.shape[0] == len(questions), \
                f"Expected {len(questions)} questions, got {embeds.shape[0]} embeddings"
            print(f"*** Building KCs based on {embed_type}, metric='{metric}' ***")
            kc = create_kc(concept_df, questions, sim_from_embeddings(embeds, metric=metric))
            if isinstance(kc, pd.DataFrame):
                kc.to_csv(os.path.join(args.output_dir, f"{embed_type}-{metric}-kc.csv"), index=False)
                print(f"*** Finished with {kc['KC'].nunique()} KCs ***")

    # Create KC for KCluster-PMI
    if pmi_dir := getattr(args, "pmi_dir", None):
        num_questions = len(questions)
        pmi = PointwiseMutualInfo.from_shards(pmi_dir, num_questions, num_questions,
                                              normalize=False, symmetric=True)
        print("*** Building KCs for KCluster-PMI ***")
        kc = create_kc(concept_df, questions, pmi.pmi_mat)
        if isinstance(kc, pd.DataFrame):
            kc.to_csv(os.path.join(args.output_dir, "pmi-kc.csv"), index=False)
            print(f"*** Finished with {kc['KC'].nunique()} KCs ***")

        # An additional KC model with the question-format nuisance divided out.
        # Written alongside the plain one rather than replacing it: whether the
        # correction helps is an empirical question per dataset, and on a
        # single-format bank it is a no-op the comparison should show.
        if getattr(args, "residualize", False):
            print("*** Building KCs for KCluster-PMI, residualized by question type ***")
            adjusted = residualize(pmi.pmi_mat, [q.q_type for q in questions])
            kc = create_kc(concept_df, questions, adjusted)
            if isinstance(kc, pd.DataFrame):
                kc.to_csv(os.path.join(args.output_dir, "pmi-resid-kc.csv"), index=False)
                print(f"*** Finished with {kc['KC'].nunique()} KCs ***")

    # Save arguments
    fname = os.path.splitext(os.path.basename(args.data_path))[0]
    with open(os.path.join(args.output_dir, f"args-kc-{fname}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def add_arguments(parser):
    parser.add_argument("--concept_dir", default=argparse.SUPPRESS, type=str,
                        help="Path to a directory containing concepts (default: <run_dir>/concept)")
    parser.add_argument("--pmi_dir", default=argparse.SUPPRESS, type=str,
                        help="Path to a directory containing PMI values")
    parser.add_argument("--residualize", action="store_true",
                        help="Also build a KC model from congruity residualized by question type, "
                             "which stops a mixed-format bank from clustering by format")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str, help="The output directory")
    parser.add_argument("--run_dir", default=argparse.SUPPRESS, type=str,
                        help="Shared run folder; each step writes to <run_dir>/<step> (env: KCLUSTER_RUN_DIR)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
