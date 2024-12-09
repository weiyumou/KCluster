import argparse
import glob
import os
import warnings
from operator import itemgetter

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from core.kcluster import KCluster


def create_kc(concept_df: pd.DataFrame, kcluster: KCluster, **kwargs) -> pd.DataFrame | None:
    # Search for the minimal damping factor leading to convergence
    damping = 0.5
    while damping < 1.0:
        try:
            kc = kcluster.create_new_kc(damping=damping, **kwargs)
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


def main(args):
    # Flag convergence issues as errors
    warnings.filterwarnings("error", category=ConvergenceWarning)

    output_dir = os.path.join(args.concept_dir, "kc")
    os.makedirs(output_dir, exist_ok=True)

    # Check all concepts are correctly filled
    [fname] = glob.glob("*-concept.csv", root_dir=args.concept_dir)
    concept_df = pd.read_csv(os.path.join(args.concept_dir, fname))
    assert concept_df["KC"].apply(bool).sum() == concept_df.shape[0], "Some concepts are invalid"
    if q_types := getattr(args, "q_types", None):
        concept_df = concept_df.loc[concept_df["type"].isin(q_types)]
    concept_df.to_csv(os.path.join(output_dir, f"concept-kc.csv"), index=False)

    # Create KC for baselines
    for embed_type in ("concept", "question"):
        for metric in ("cosine", "euclidean"):
            kcluster = KCluster(args.concept_dir, metric=metric, embed_type=embed_type)
            print(f"*** Building KCs based on {embed_type}, metric='{metric}' ***")
            kc = create_kc(concept_df, kcluster, predicate=(lambda q: q.q_type in q_types) if q_types else None)
            if isinstance(kc, pd.DataFrame):
                kc.to_csv(os.path.join(output_dir, f"{embed_type}-{metric}-kc.csv"), index=False)
                print(f"*** Finished with {kc['KC'].nunique()} KCs ***")

    # Create KC for KCluster-PMI
    kcluster = KCluster(args.pmi_dir, metric="pmi", normalize_pmi=True)
    print("*** Building KCs for KCluster-PMI ***")
    kc = create_kc(concept_df, kcluster, predicate=(lambda q: q.q_type in q_types) if q_types else None)
    if isinstance(kc, pd.DataFrame):
        kc.to_csv(os.path.join(output_dir, f"pmi-kc.csv"), index=False)
        print(f"*** Finished with {kc['KC'].nunique()} KCs ***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--concept_dir", required=True, type=str, help="Path to a directory containing concepts")
    parser.add_argument("--pmi_dir", required=True, type=str, help="Path to a directory containing PMI values")
    parser.add_argument("--q_types", default=argparse.SUPPRESS, type=str, nargs="*", help="Question types to consider")
    cl_args = parser.parse_args()
    main(cl_args)
