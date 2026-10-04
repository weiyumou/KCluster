import glob
import os
import re

import numpy as np
import pandas as pd

from kcluster.core.pmi import PointwiseMutualInfo
from kcluster.io.datashop import KC_PAT, get_step_to_kc, merge_student_step_with_kc
from kcluster.io.jsonl import load_questions
from kcluster.paths import (
    concept_kc_path,
    embed_dir,
    pmi_raw_dir,
    prepare_output_dir,
    question_path,
    required_result_dir,
    result_dataset,
)
from kcluster.tasks.cluster import create_kc, sim_from_embeddings


def main(args):
    # The result dir (D10 layout) holds the concept KC, embeddings, and shards
    result_dir = required_result_dir(args.result_dir, args.run_dir)
    output_dir = prepare_output_dir(os.path.join(result_dir, "kc-refine"), exist_ok=False)
    print(f"*** Writing results to {output_dir} ***")

    # Identify the KC model to refine and load the KC values
    kcm = os.path.split(args.kc_val_path)[1].split("_")[0]  # Extract the KC model from the file name
    kc_val = pd.read_csv(args.kc_val_path)
    val_mask = (kc_val["Slope"].le(0.001)) & (kc_val["Intercept (probability) at Opportunity 1"].between(0.2, 0.8))

    # Load the KC model to refine
    kc = pd.read_csv(args.kc_path, sep="\t", na_values=" ",
                     usecols=(lambda col: col == f"KC ({kcm})" or not re.match(KC_PAT, col)))
    assert f"KC ({kcm})" in kc, "The KC model to refine is not found in the template file"

    # Load concepts (the concept step writes them under kc/)
    concept_df = pd.read_csv(concept_kc_path(result_dir))
    assert concept_df["KC"].str.strip().all(), "Some concepts are invalid"

    # Recover the questions behind the concepts (--data_path overrides the recorded path)
    questions = load_questions(question_path(result_dir, result_dataset(result_dir), args.data_path))

    # Build the two similarity matrices once
    [fname] = glob.glob("*_llm-embed.npy", root_dir=embed_dir(result_dir))
    embeds = np.load(os.path.join(embed_dir(result_dir), fname))
    assert embeds.shape[0] == len(questions), \
        f"Expected {len(questions)} questions, got {embeds.shape[0]} embeddings"
    q_cos_sim = sim_from_embeddings(embeds, metric="cosine")

    raw_dir = args.pmi_dir or pmi_raw_dir(result_dir)
    pmi = PointwiseMutualInfo.from_shards(raw_dir, len(questions), len(questions),
                                          normalize=True, symmetric=True)
    pmi_sim = pmi.pmi_mat

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

        def predicate(q, step_name=step_name):  # bind the loop variable
            return any(t in step_name for t in q["ds-step-name"])

        # Refinement using 'question-cosine'
        print("*** Refining KCs based on question, metric='cosine' ***")
        q_cos_kc = create_kc(concept_kc, questions, q_cos_sim, predicate=predicate)
        assert q_cos_kc is not None, "Failed to create KCs with question-cosine"

        # Refinement using PMI
        print("*** Refining KCs with KCluster-PMI ***")
        pmi_kc = create_kc(concept_kc, questions, pmi_sim, predicate=predicate)
        assert pmi_kc is not None, "Failed to create KCs with KCluster-PMI"

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
    if args.ss_path:
        print("*** Merging KCs with student steps ***")
        ss = merge_student_step_with_kc(args.ss_path, kc, minimal=args.minimal, multiplier=args.multiplier)
        fname = os.path.splitext(os.path.basename(kc_path))[0]
        fname = f"{fname}-merged-minimal={args.minimal}-multiplier={args.multiplier}.txt"
        ss.to_csv(os.path.join(output_dir, fname), sep="\t", index=False)


def add_arguments(parser):
    parser.add_argument("--kc_path", required=True, type=str, help="Path to a DataShop KC template file")
    parser.add_argument("--kc_val_path", required=True, type=str, help="Path to a CSV file containing KC values")
    parser.add_argument("--result_dir", default=None, type=str,
                        help="The result directory holding the concept KC, embeddings, and score shards "
                             "(default: --run_dir)")
    parser.add_argument("--pmi_dir", default=None, type=str,
                        help="Directory of raw score shards (default: <result_dir>/mat/pmi/raw)")
    parser.add_argument("--data_path", default=None, type=str,
                        help="Question file (default: the path recorded in args-*.json)")
    parser.add_argument("--run_dir", default=None, type=str,
                        help="Result folder shared by every step of this run (env: KCLUSTER_RUN_DIR)")
    parser.add_argument("--ss_path", default=None, type=str, help="Path to a DataShop student-step file")
    parser.add_argument("--minimal", action="store_true", help="Whether to minimize the merged student-step file")
    parser.add_argument("--multiplier", default=1, type=int,
                        help="Number of times to duplicate each KC model for cross-validation")
