import argparse
import glob
import os
import re
import time

import pandas as pd
import torch

from core.kcluster import KCluster
from evaluation.build_datashop_kc import merge_student_step_with_kc
from evaluation.build_kc import create_kc
from processing.elearning import get_step_to_kc
from processing.util import KC_PAT


def main(args):
    output_dir = os.path.join("results", "kc-refine", time.asctime()).replace(' ', '_').replace(':', '-')
    os.makedirs(output_dir, exist_ok=False)

    # Identify the KC model to refine and load the KC values
    kcm = os.path.split(args.kc_val_path)[1].split("_")[0]  # Extract the KC model from the file name
    kc_val = pd.read_csv(args.kc_val_path)
    val_mask = (kc_val["Slope"].le(0.001)) & (kc_val["Intercept (probability) at Opportunity 1"].between(0.2, 0.8))

    # Load the KC model to refine
    kc = pd.read_csv(args.kc_path, sep="\t", na_values=" ",
                     usecols=(lambda col: col == f"KC ({kcm})" or not re.match(KC_PAT, col)))
    assert f"KC ({kcm})" in kc, "The KC model to refine is not found in the template file"

    # Load concepts
    [fname] = glob.glob("*-concept.csv", root_dir=args.concept_dir)
    concept_df = pd.read_csv(os.path.join(args.concept_dir, fname))
    assert concept_df["KC"].str.strip().all(), "Some concepts are invalid"

    count = 0
    for kc_label, num_steps in kc_val.loc[val_mask, ["KC Name", "Number of Unique Steps"]].itertuples(index=False):
        print(f"*** Refining KC '{kc_label}' ***")

        label_mask = kc[f"KC ({kcm})"].eq(kc_label)
        assert label_mask.sum() == num_steps, "Inconsistent number of steps"

        step_name = set(kc.loc[label_mask, "Step Name"])
        concept_mask = concept_df["ds-step-name"].apply(lambda x: any(t in step_name for t in x.split("~")))
        if concept_mask.sum() <= 2:  # skip refinement if there are two or fewer questions
            print("Too few questions tagged with this KC, skipping refinement. ")
            continue

        # Refinement using 'concept'
        concept_kc = concept_df[concept_mask]

        # Refinement using 'question-cosine'
        kcluster = KCluster(args.concept_dir, metric="cosine", embed_type="question")
        print(f"*** Refining KCs based on question, metric='cosine' ***")
        q_cos_kc = create_kc(concept_kc, kcluster, predicate=(lambda q: any(t in step_name for t in q["ds-step-name"])))
        assert isinstance(q_cos_kc, pd.DataFrame), "Failed to create KCs with question-cosine"

        # Refinement using PMI
        kcluster = KCluster(args.pmi_dir, metric="pmi",
                            normalize=True, symmetric=True, dtype=torch.float16)
        print("*** Refining KCs with KCluster-PMI ***")
        pmi_kc = create_kc(concept_kc, kcluster, predicate=(lambda q: any(t in step_name for t in q["ds-step-name"])))
        assert isinstance(pmi_kc, pd.DataFrame), "Failed to create KCs with KCluster-PMI"

        for name, df in zip((f"{item}-{kc_label}" for item in ("cpt", "qcos", "pmi")), (concept_kc, q_cos_kc, pmi_kc)):
            kc[f"KC ({name})"] = kc[f"KC ({kcm})"]
            step_to_kc = get_step_to_kc(df)
            kc.loc[label_mask, f"KC ({name})"] = kc.loc[label_mask, "Step Name"].map(step_to_kc)
            assert kc[f"KC ({name})"].dropna().ne(kc[f"KC ({kcm})"].dropna()).sum() == num_steps, "Incorrect refinement"
        count += 1

    print(f"*** Refined {count} KCs in {kcm} ***")
    kc_path = os.path.join(output_dir, "refined-kc.txt")
    kc.to_csv(kc_path, sep="\t", index=False)

    # Merge KCs into student step (for cross-validation) if a path is present
    if ss_path := getattr(args, "ss_path", None):
        multiplier = getattr(args, "multiplier", 1)
        minimal = getattr(args, "minimal", False)
        print("*** Merging KCs with student steps ***")
        ss = merge_student_step_with_kc(ss_path, kc, minimal=minimal, multiplier=multiplier)
        fname = os.path.splitext(os.path.basename(kc_path))[0]
        fname = f"{fname}-merged-minimal={minimal}-multiplier={multiplier}.txt"
        ss.to_csv(os.path.join(output_dir, fname), sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kc_path", required=True, type=str, help="Path to a DataShop KC template file")
    parser.add_argument("--kc_val_path", required=True, type=str, help="Path to a CSV file containing KC values")
    parser.add_argument("--concept_dir", required=True, type=str, help="Path to a directory containing concepts")
    parser.add_argument("--pmi_dir", required=True, type=str,
                        help="Path to a directory containing PMI values")
    parser.add_argument("--ss_path", default=argparse.SUPPRESS, type=str, help="Path to a DataShop student-step file")
    parser.add_argument("--minimal", default=argparse.SUPPRESS, action="store_true",
                        help="Whether to minimize the merged student-step file")
    parser.add_argument("--multiplier", default=argparse.SUPPRESS, type=int,
                        help="Number of times to duplicate each KC model for cross-validation")

    cl_args = parser.parse_args()
    main(cl_args)
