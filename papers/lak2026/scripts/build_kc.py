"""Per-LO KC clustering, MCQ selection, and the study report.

Study orchestration over the library: for each generated-question directory,
cluster every LO's validated questions from the run_pmi.py shards
(normalized PMI), select MCQs by KC coverage, and render the standards-
keyed report. Replaces the legacy ext/-KCluster variant with the canonical
PointwiseMutualInfo + tasks.cluster.create_kc.
"""

import argparse
import glob
import json
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from kcluster.core.pmi import PointwiseMutualInfo
from kcluster.io.jsonl import dump_questions, load_questions
from kcluster.tasks.cluster import create_kc
from kcluster.tasks.qgen.select import build_report, select_mcq


def main(args):
    rng = np.random.default_rng(42)

    # Flag convergence issues as errors
    warnings.filterwarnings("error", category=ConvergenceWarning)

    # Read question root
    with open(os.path.join(args.pmi_root, "args-concept.json"), "r") as f:
        configs = json.load(f)
    question_root, use_gpt_valid = configs["root_dir"], configs["use_gpt_valid"]
    pat = "-gpt" if use_gpt_valid else ""

    for fname in tqdm(
            sorted(glob.iglob("**/concept.csv", recursive=True, root_dir=args.pmi_root)), desc="Building KCs"):
        curr_dir = os.path.split(fname)[0]

        # Check all concepts are correctly filled
        concept_df = pd.read_csv(os.path.join(args.pmi_root, fname))
        assert concept_df["KC"].str.strip().all(), "Some concepts are invalid"

        # Read all questions by LO
        all_questions = defaultdict(list)
        for q in load_questions(os.path.join(question_root, curr_dir, f"new-mcq{pat}-valid.jsonl")):
            all_questions[q["lo"]].append(q)

        # Read LO mappings
        with open(os.path.join(args.pmi_root, curr_dir, "lo_mappings.json"), "r") as f:
            lo_mappings = json.load(f)

        kc_dfs, sel_questions = [], []
        for lo_idx, lo in lo_mappings.items():
            if len(all_questions[lo]) < args.mcq_per_lo:
                print(f"Ignoring LO '{lo}'; no enough questions")
                continue

            n = len(all_questions[lo])
            pmi = PointwiseMutualInfo.from_shards(os.path.join(args.pmi_root, curr_dir, lo_idx), n, n,
                                                  normalize=True, symmetric=True)
            c_df = concept_df.loc[concept_df["lo"] == lo].reset_index(drop=True)
            kc_df = create_kc(c_df, all_questions[lo], pmi.pmi_mat)
            if isinstance(kc_df, pd.DataFrame):
                # Add KCs to each question
                for q, kc in zip(all_questions[lo], kc_df["KC"].tolist()):
                    q["kc"] = kc

                # Select questions based on KCs
                sel_questions.extend(select_mcq(kc_df, all_questions[lo], mcq_per_lo=args.mcq_per_lo, rng=rng))
                kc_dfs.append(kc_df)

        # Save KCs
        kc = pd.concat(kc_dfs, ignore_index=True)
        kc.to_csv(os.path.join(args.pmi_root, curr_dir, "pmi-kc.csv"), index=False)

        # Save selected MCQs
        dump_questions(sel_questions, os.path.join(args.pmi_root, curr_dir, "sel-mcq.jsonl"))

        # Build human-readable report
        if use_gpt_valid:
            std_path = os.path.join(args.std_root, os.path.dirname(curr_dir) + ".csv")
        else:
            std_path = os.path.join(args.std_root, curr_dir + ".csv")
        std_df = pd.read_csv(std_path)
        std_df["Standard Text"] = std_df["Standard Text"].str.rstrip(".*")
        res_df = build_report(sel_questions, dict(zip(std_df["Standard Text"], std_df["Standard"])))
        res_df.to_csv(os.path.join(args.pmi_root, curr_dir, "sel-mcq.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--std_root", required=True, type=str, help="Path to a directory containing standards")
    parser.add_argument("--pmi_root", required=True, type=str, help="Path to a directory containing PMI values")
    parser.add_argument("--mcq_per_lo", default=6, type=int, help="Number of MCQs to be selected for each LO")

    main(parser.parse_args())
