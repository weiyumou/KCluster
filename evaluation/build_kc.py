import argparse
import glob
import json
import os
import time
import warnings
from operator import itemgetter

import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning

from core.kcluster import KCluster


def create_kc(concept_df: pd.DataFrame, kcluster: KCluster, **kwargs) -> pd.DataFrame | None:
    # Flag convergence issues as errors
    warnings.filterwarnings("error", category=ConvergenceWarning)

    # Search for the minimal damping factor leading to convergence
    damping = 0.5
    while damping < 1.0:
        try:
            kc = kcluster.create_new_kc(damping=damping, **kwargs)
            assert kc.shape[0] == concept_df.shape[0], "Inconsistent number of questions"
        except ConvergenceWarning:
            print(f"Did not converge when damping = {damping}")
            damping += 0.05
        else:
            # populate the concepts of exemplars to its subordinates
            kc = kc.rename(columns={"KC": "KC-raw"})
            exemplars = kc["KC-raw"].str.split("-").apply(itemgetter(1)).apply(int)
            kc["KC"] = concept_df["KC"].iloc[exemplars].reset_index(drop=True)
            return kc
    print("*** Failed to create KCs ***")
    return None


def main(args):
    args.output_dir = getattr(args, "output_dir", os.path.join("results", "kc", time.asctime()))
    args.output_dir = args.output_dir.replace(' ', '_').replace(':', '-')
    os.makedirs(args.output_dir, exist_ok=True)

    # Check all concepts are correctly filled
    [fname] = glob.glob("*-concept.csv", root_dir=args.concept_dir)
    concept_df = pd.read_csv(os.path.join(args.concept_dir, fname))
    assert concept_df["KC"].str.strip().all(), "Some concepts are invalid"

    # Concept is already a KC model; copy it to the output directory
    concept_df.to_csv(os.path.join(args.output_dir, f"concept-kc.csv"), index=False)

    # Determine which types of embeddings are available
    embed_types = []
    if glob.glob("*-concept-embeds.npy", root_dir=args.concept_dir):
        embed_types.append("concept")
    if glob.glob("*-question-embeds.npy", root_dir=args.concept_dir):
        embed_types.append("question")

    # Create KC for baselines
    for embed_type in embed_types:
        for metric in ("cosine",):
            kcluster = KCluster(args.concept_dir, metric=metric, embed_type=embed_type)
            print(f"*** Building KCs based on {embed_type}, metric='{metric}' ***")
            kc = create_kc(concept_df, kcluster)
            if isinstance(kc, pd.DataFrame):
                kc.to_csv(os.path.join(args.output_dir, f"{embed_type}-{metric}-kc.csv"), index=False)
                print(f"*** Finished with {kc['KC'].nunique()} KCs ***")

    # Create KC for KCluster-PMI
    if pmi_dir := getattr(args, "pmi_dir", None):
        kcluster = KCluster(pmi_dir, metric="pmi",
                            normalize=True, symmetric=True, dtype=torch.float16)
        print("*** Building KCs for KCluster-PMI ***")
        kc = create_kc(concept_df, kcluster)
        if isinstance(kc, pd.DataFrame):
            kc.to_csv(os.path.join(args.output_dir, f"pmi-kc.csv"), index=False)
            print(f"*** Finished with {kc['KC'].nunique()} KCs ***")

    # Save arguments
    [fname] = glob.glob("args*.json", root_dir=args.concept_dir)
    with open(os.path.join(args.concept_dir, fname), "r") as f:
        args.data_path = json.load(f)["data_path"]

    fname = os.path.splitext(os.path.basename(args.data_path))[0]
    with open(os.path.join(args.output_dir, f"args-kc-{fname}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--concept_dir", required=True, type=str, help="Path to a directory containing concepts")
    parser.add_argument("--pmi_dir", default=argparse.SUPPRESS, type=str,
                        help="Path to a directory containing PMI values")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str, help="The output directory")
    cl_args = parser.parse_args()
    main(cl_args)
