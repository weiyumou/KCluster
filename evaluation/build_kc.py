import argparse
import glob
import os
import time
import warnings
from operator import itemgetter

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from core.kcluster import KCluster
from core.question import Question
from processing.elearning import get_step_name


def create_kc(concept_df: pd.DataFrame, kcluster: KCluster, **kwargs) -> pd.DataFrame | None:
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


def main(args):
    # Flag convergence issues as errors
    warnings.filterwarnings("error", category=ConvergenceWarning)

    output_dir = os.path.join(args.concept_dir, "kc", time.asctime().replace(' ', '_').replace(':', '-'))
    os.makedirs(output_dir, exist_ok=True)

    # Check all concepts are correctly filled
    [fname] = glob.glob("*-concept.csv", root_dir=args.concept_dir)
    concept_df = pd.read_csv(os.path.join(args.concept_dir, fname))
    assert concept_df["KC"].str.strip().all(), "Some concepts are invalid"

    # if kc_path is provided, adjust concept_df accordingly
    predicate = None
    if kc_path := getattr(args, "kc_path", None):
        assert hasattr(args, "year"), "'year' must also be provided when 'kc_path' is provided"
        step_names = set(get_step_name(kc_path, args.year, filter_by_kc=True))

        def any_valid(names: list[str]) -> bool:
            return any(t in step_names or t.split("_")[-1] in step_names for t in names)

        def predicate(q: Question) -> bool:
            return any_valid(q["step-name"])

        mask = concept_df["step-name"].apply(lambda x: any_valid(x.split("~")))
        assert mask.any(), "No valid questions found"
        concept_df = concept_df.loc[mask]

    concept_df.to_csv(os.path.join(output_dir, f"concept-kc.csv"), index=False)

    # Create KC for baselines
    for embed_type in ("concept", "question"):
        for metric in ("cosine",):
            kcluster = KCluster(args.concept_dir, metric=metric, embed_type=embed_type)
            print(f"*** Building KCs based on {embed_type}, metric='{metric}' ***")
            kc = create_kc(concept_df, kcluster, predicate=predicate)
            if isinstance(kc, pd.DataFrame):
                kc.to_csv(os.path.join(output_dir, f"{embed_type}-{metric}-kc.csv"), index=False)
                print(f"*** Finished with {kc['KC'].nunique()} KCs ***")

    # Create KC for KCluster-PMI
    if pmi_dir := getattr(args, "pmi_dir", None):
        kcluster = KCluster(pmi_dir, metric="pmi", normalize_pmi=True)
        print("*** Building KCs for KCluster-PMI ***")
        kc = create_kc(concept_df, kcluster, predicate=predicate)
        if isinstance(kc, pd.DataFrame):
            kc.to_csv(os.path.join(output_dir, f"pmi-kc.csv"), index=False)
            print(f"*** Finished with {kc['KC'].nunique()} KCs ***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--concept_dir", required=True, type=str, help="Path to a directory containing concepts")
    parser.add_argument("--pmi_dir", default=argparse.SUPPRESS, type=str,
                        help="Path to a directory containing PMI values")
    parser.add_argument("--kc_path", default=argparse.SUPPRESS, type=str,
                        help="Path to a DataShop KC model file, used to filter questions")
    parser.add_argument("--year", default=argparse.SUPPRESS, type=str,
                        help="Specifies the year of the E-learning dataset; required when kc_path is provided")
    cl_args = parser.parse_args()
    main(cl_args)
