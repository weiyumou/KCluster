"""MCQ selection by KC coverage (LAK 2026).

After clustering an LO's validated questions into KCs, ``select_mcq`` keeps
every cluster exemplar and, if more questions are needed, samples the rest
inversely proportional to their cluster's size — favoring coverage of small
clusters over redundancy in large ones. ``build_report`` renders selected
questions as a human-readable table keyed to the course standards.
"""

import copy

import numpy as np
import pandas as pd

from kcluster.core.question import Question


def select_mcq(kc: pd.DataFrame, questions: list[Question],
               mcq_per_lo: int = 6, rng: np.random.Generator = None) -> list[Question]:
    inv_val_counts = kc.value_counts("KC-raw", sort=True).rdiv(1.0)
    exemplars = [int(x.split("-")[1]) for x in inv_val_counts.index.tolist()]

    # all exemplars are selected; if not enough, randomly sample others
    if (n := mcq_per_lo - len(exemplars)) > 0:
        weights = kc["KC-raw"].apply(lambda x: inv_val_counts.loc[x])
        weights.loc[exemplars] = 0.0
        smp = kc["KC-raw"].sample(n=n, replace=False, weights=weights, random_state=rng)
        exemplars.extend(smp.index.tolist())

    return [questions[idx] for idx in exemplars[:mcq_per_lo]]


def build_report(questions: list[Question], std_to_code: dict[str, str]) -> pd.DataFrame:
    records = []
    for q in questions:
        choices = copy.deepcopy(q["question"]["choices"])
        ans_txt = choices.pop(ord(q["answerKey"]) - ord("a"))["text"]
        q_dict = {"Standard Code": None, "Standard Text": q["lo"][0].upper() + q["lo"][1:].rstrip(".*") + ".",
                  "Stem": q["question"]["stem"], "Answer": ans_txt, "Key": q["answerKey"], }
        q_dict["Standard Code"] = std_to_code[q_dict["Standard Text"].rstrip(".")]
        for idx, item in enumerate(choices, 1):
            q_dict[f"Distractor {idx}"] = item["text"]
        q_dict |= {"Explanation": q["explanation"], "KC": q["kc"]}
        records.append(q_dict)

    return pd.DataFrame.from_records(records)
