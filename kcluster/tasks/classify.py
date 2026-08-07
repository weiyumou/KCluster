"""LO-alignment: matching questions to learning objectives (LAK 2026).

``QuestionLO`` flattens one rectangular LO×question scoring job into a
linear index space of (context, text) pairs: the first ``n_questions``
items are the marginals — the scored question under its bare type header —
and item ``n + i*n + j`` scores question ``j`` conditioned on learning
objective ``i``. The reassembled ``pmi_mat`` is therefore rectangular with
rows = LOs (conditioning) and columns = questions, and uses the raw PMI
estimator only (``normalize=False``; the joint-normalization path is
square-only by design).

LOs come in two grammatical shapes with different scaffolds: ``actions``
("... test whether a student can <verb phrase>") and ``facts`` ("... test
whether a student knows:\\n<statement>").
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from kcluster.core.pmi import PointwiseMutualInfo
from kcluster.core.question import Question
from kcluster.io.jsonl import dump_questions, load_questions


class QuestionLO:
    def __init__(self, questions: list[Question], los: list[str], lo_type: str):
        match lo_type:
            case "actions":
                self.los = [lo[0].lower() + lo[1:].rstrip().rstrip(".") for lo in los]
            case "facts":
                self.los = [lo.rstrip().rstrip(".") for lo in los]
            case _:
                raise ValueError(f"Invalid lo_type: '{lo_type}'")

        self.questions, self.lo_type = questions, lo_type

    def __getitem__(self, index):
        n = len(self.questions)
        if index < n:
            q = self.questions[index]
            return f"{q.q_type}:\n", str(q)

        lo_idx, q_idx = (index - n) // n, (index - n) % n
        lo, q = self.los[lo_idx], self.questions[q_idx]
        header = None
        match self.lo_type:
            case "actions":
                header = f"The exercise below is designed to test whether a student can {lo}."
            case "facts":
                header = f"The exercise below is designed to test whether a student knows:\n{lo}."

        return f"{header}\n\n{q.q_type}:\n", str(q)

    def __len__(self):
        return len(self.questions) * len(self.los) + len(self.questions)


def classify_from_pmi(root_dir: str, topk: int = 3) -> pd.DataFrame:
    """Aggregate a scored classify run into top-k LO predictions per question.

    Reads the score shards plus the ``args-pmi-*.json`` / ``los-*.json``
    breadcrumbs the classify command writes, saves the questions whose true
    ``lo`` appears in the top-k (``matched-top{k}.jsonl``) and the full
    prediction table (``classified-top{k}.csv``).
    """
    # Read the args json file to extract data_path
    [args_f] = glob.glob(os.path.join(root_dir, "args-pmi-*.json"))
    with open(args_f, "r") as f:
        args = json.load(f)

    # Load questions from data_path
    all_questions = load_questions(args["data_path"])

    # Load LOs
    [los_f] = glob.glob(os.path.join(root_dir, "los-*.json"))
    with open(los_f, "r") as f:
        all_los = json.load(f)

    pmi = PointwiseMutualInfo.from_shards(root_dir, len(all_los), len(all_questions),
                                          symmetric=False, normalize=False)

    # Top K predictions (descending, per question)
    preds = np.argsort(-pmi.pmi_mat, axis=0)[:topk].T  # (n_questions, topk)
    records, matched = [], []
    for idx, q in enumerate(all_questions):
        d, is_matched = q.flat_dict, False
        for k, p in enumerate(preds[idx], 1):
            d |= {f"pred_lo_{k}": all_los[p]}
            is_matched |= q["lo"] == all_los[p]
        records.append(d)
        if is_matched:
            matched.append(q)

    # Save matched questions
    dump_questions(matched, os.path.join(root_dir, f"matched-top{topk}.jsonl"))
    print(f"Matched {len(matched)} questions out of {len(all_questions)}")

    # Save classification results
    res_df = pd.DataFrame.from_records(records)
    res_df.to_csv(os.path.join(root_dir, f"classified-top{topk}.csv"), index=False)

    return res_df
