import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from kcluster.core.pmi import PointwiseMutualInfo
from kcluster.io.jsonl import load_questions
from kcluster.paths import default_output_dir, prepare_output_dir
from kcluster.tasks.cluster import create_kc, sim_from_embeddings


def main(args):
    output_dir = getattr(args, "output_dir", None) or default_output_dir("kc")
    args.output_dir = prepare_output_dir(output_dir)
    print(f"*** Writing results to {args.output_dir} ***")

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

    # Save arguments
    fname = os.path.splitext(os.path.basename(args.data_path))[0]
    with open(os.path.join(args.output_dir, f"args-kc-{fname}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def add_arguments(parser):
    parser.add_argument("--concept_dir", required=True, type=str, help="Path to a directory containing concepts")
    parser.add_argument("--pmi_dir", default=argparse.SUPPRESS, type=str,
                        help="Path to a directory containing PMI values")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str, help="The output directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
