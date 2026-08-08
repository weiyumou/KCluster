"""Dataset drivers for the E-learning Design Principles and Methods courses.

Course-specific glue for the 2022 (DataShop ds5426) and 2023 (ds5843)
offerings: each driver parses the downloaded OLI course HTML with the generic
``oli_html`` loader, keeps only the questions whose steps appear in that
dataset's DataShop KC template, and attaches the DataShop step names
(``ds-step-name``) needed to join against student-step data. The two
offerings encode step names differently, hence the separate mappings.
"""

import copy
import itertools
import os
import re

from kcluster.io.datashop import KC_PAT, load_datashop_temp
from kcluster.io.jsonl import dump_questions
from kcluster.io.loaders.oli_html import parse_all_mcqs


def write_elearning22_mcqs(root_dir: str, out_dir: str, temp_path: str):
    """
    Extract MCQs from the E-learning 2022 dataset and write them to a JSON file
    :param root_dir: A path to downloaded HTML files,
    e.g., "Downloads/_E-Learning_Design_Principles_and_Methods__v_4_2/e_learning_dp-4.2_27gtpdr5/Course_Syllabus"
    :param out_dir: A path to an output dir, e.g., data/elearning/
    :param temp_path: A path to a KC model file, e.g., "data/datashop/ds5426-elearning/ds5426_kcm.txt"
    :return: None
    """
    # Load the KC template
    kc_temp = load_datashop_temp(temp_path)
    kc_mask = kc_temp.filter(regex=KC_PAT).notna().all(axis=1)

    raw_step_names = list(kc_temp.loc[kc_mask, "Step Name"].unique())
    step_names = [x.split(" ")[0] for x in raw_step_names]

    # Create a mapping between step names and raw step names
    step_dict = dict()
    for step, raw_step in zip(step_names, raw_step_names, strict=True):
        step_dict.setdefault(step, []).append(raw_step)

    # Parse all MCQs
    all_questions = parse_all_mcqs(root_dir)

    # Filter out questions that are not in the template
    elearning22 = []
    for q in all_questions:
        mask = [step in step_dict for step in q["step-name"]]
        if sum(mask) > 0:
            qc = copy.deepcopy(q)
            qc["skillref"] = list(itertools.compress(q["skillref"], mask))
            qc["step-name"] = list(itertools.compress(q["step-name"], mask))

            qc["ds-step-name"] = []
            for s in qc["step-name"]:
                qc["ds-step-name"].extend(step_dict[s])
            elearning22.append(qc)

    # Write MCQs to a JSON file for program readability
    out_path = os.path.join(out_dir, "elearning22-mcq.jsonl")
    dump_questions(elearning22, out_path)
    print(f"Wrote {len(elearning22)} questions to {out_path}")


def write_elearning23_mcqs(root_dir: str, out_dir: str, temp_path: str):
    """
    Extract MCQs from the E-learning 2023 dataset and write them to a JSON file
    :param root_dir: A path to downloaded HTML files,
    e.g., "Downloads/_E-Learning_Design_Principles_and_Methods__v_4_2/e_learning_dp-4.2_27gtpdr5/Course_Syllabus"
    :param out_dir: A path to an output dir, e.g., data/elearning/
    :param temp_path: A path to a KC model file, e.g., "data/datashop/ds5843-elearning/ds5843_kcm.txt"
    :return: None
    """
    # Load the KC template
    kc_temp = load_datashop_temp(temp_path)
    kc_mask = kc_temp.filter(regex=KC_PAT).notna().all(axis=1)

    raw_step_names = list(kc_temp.loc[kc_mask, "Step Name"].unique())
    step_names = [re.search(r"(?<=part ).+", x).group(0).split()[0] for x in raw_step_names]

    # Create a mapping between step names and raw step names
    step_dict = dict()
    for step, raw_step in zip(step_names, raw_step_names, strict=True):
        step_dict.setdefault(step, []).append(raw_step)

    # Parse all MCQs
    all_questions = parse_all_mcqs(root_dir)

    # Filter out questions that are not in the template
    elearning23 = []
    for q in all_questions:
        mask = [step.split("_")[-1] in step_dict for step in q["step-name"]]
        if sum(mask) > 0:
            qc = copy.deepcopy(q)
            qc["skillref"] = list(itertools.compress(q["skillref"], mask))
            qc["step-name"] = list(itertools.compress(q["step-name"], mask))

            qc["ds-step-name"] = []
            for s in qc["step-name"]:
                qc["ds-step-name"].extend(step_dict[s.split("_")[-1]])
            elearning23.append(qc)

    # Write MCQs to a JSON file for program readability
    out_path = os.path.join(out_dir, "elearning23-mcq.jsonl")
    dump_questions(elearning23, out_path)
    print(f"Wrote {len(elearning23)} questions to {out_path}")
