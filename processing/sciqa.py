import json
import os
from collections import Counter

import pandas as pd

from processing.util import compute_metrics


def write_questions(data_path: str, output_dir: str, min_choice_cnt: int = 2, min_skill_cnt: int = 100):
    """
    Write ScienceQA questions to a jsonl file
    :param data_path: A path to the original ScienceQA data file, e.g., "Downloads/ScienceQA/problems.json"
    :param output_dir: A path to the output directory, e.g., "data/sciqa/"
    :param min_choice_cnt: Include questions whose number of choices is above this threshold
    :param min_skill_cnt: Include questions whose skill has a count above this threshold
    :return: None
    """
    with open(data_path, "r") as f:
        all_questions = json.loads(f.read())

    questions = []
    for q_id in all_questions:
        q_dict = all_questions[q_id]
        if (q_dict["image"] is None) and (len(q_dict["choices"]) >= min_choice_cnt):
            new_q_dict = {"id": f"sciqa-{q_id}", "type": "Multiple Choice",
                          "question": {"stem": q_dict["question"], "choices": []}}
            for idx, chc in enumerate(q_dict["choices"]):
                new_q_dict["question"]["choices"].append({"label": chr(ord("a") + idx), "text": chc})
            new_q_dict["answerKey"] = chr(ord("a") + q_dict["answer"])

            new_q_dict["skill"] = q_dict["skill"]
            new_q_dict["subject"] = q_dict["subject"]
            new_q_dict["topic"] = q_dict["topic"]
            new_q_dict["category"] = q_dict["category"]
            new_q_dict["grade"] = q_dict["grade"]

            questions.append(new_q_dict)

    skill_cnt = Counter((item["skill"] for item in questions))
    sel_questions = [q_dict for q_dict in questions if skill_cnt[q_dict["skill"]] >= min_skill_cnt]

    output_path = os.path.join(output_dir, f"sciqa-skill-{min_skill_cnt}.jsonl")
    with open(output_path, "w") as f:
        f.write("\n".join([repr(q) for q in sel_questions]))
    print(f"Wrote {len(sel_questions)} questions to {output_path}")


def evaluate_kc(kc_path: str, true_kc: str = "skill"):
    kc = pd.read_csv(kc_path)
    return compute_metrics(kc[true_kc], kc["kc"])
