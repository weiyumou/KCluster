import copy
import glob
import itertools
import json
import os
import re

import pandas as pd
from bs4 import BeautifulSoup

from core.question import Question
from processing.util import KC_PAT


def parse_mcq(data_path: str) -> list[dict]:
    """Parse all MCQs from an HTML document at `data_path`."""
    with open(data_path) as fp:
        soup = BeautifulSoup(fp, features="html.parser")

    # Iterate through all question divs
    questions = []
    for q_div in soup.find_all("div", class_="oli-question"):
        q_dict = {
            "id": q_div["id"],
            "type": "Multiple Choice",
            "question": {"stem": "", "choices": []}
        }

        body_div = q_div.find("div", class_="oli-body")

        # Extract the stem
        stem = []
        for p in body_div.find_all("p", recursive=True):
            stem.append(" ".join(p.stripped_strings))  # concatenate content within <p> tag with space
        stem = "\n".join(stem).strip()  # concatenate <p> tags with \n
        if not stem:
            continue
        q_dict["question"]["stem"] = stem

        # Extract the image path, if any
        for img in body_div.find_all("img", recursive=False):
            q_dict.setdefault("images", []).append(os.path.basename(img["src"]))

        # Extract choices
        if not (mcq_tag := q_div.find("div", class_="oli-multiple-choice")):
            continue

        chc_values = []
        for idx, chc in enumerate(mcq_tag.find_all("div")):
            text = " ".join(chc.stripped_strings)
            text = re.sub(r"\[.*\]", "", re.sub(r"\(value:.*\)", "", text)).strip()
            q_dict["question"]["choices"].append(
                {"label": chr(ord("a") + idx), "text": text}
            )
            chc_values.append(chc["value"])

        if not all(opt["text"] for opt in q_dict["question"]["choices"]):
            continue

        # Extract the answer
        ans_tags = q_div.find_all("div", class_="oli-response")
        if not all(tag.get("score", None) for tag in ans_tags):
            continue

        answers = [tag["match"].split(",") for tag in ans_tags if tag["score"] != "0"]
        if not ((len(answers) == 1) and (len(answers[0]) == 1)):
            continue  # ignore questions with more than one answer
        if (ans := answers[0][0]) not in chc_values:
            continue  # ignore questions whose choices do not match with what's in the question text

        q_dict["answerKey"] = chr(ord("a") + chc_values.index(ans))
        q_dict["oli-part-id"] = q_div.find("div", class_="oli-part")["id"]

        if skillref := q_div.find("div", class_="oli-part").find("skillref"):
            q_dict["skillref"] = skillref["idref"].strip()

        q_dict["step-name"] = q_dict["id"] + "_" + q_dict["oli-part-id"]
        questions.append(q_dict)

    return questions


def parse_all_mcqs(root_dir: str) -> list[Question]:
    """
    Parse all unique MCQs from a root directory
    :param root_dir: A path to downloaded HTML files,
    e.g., "Downloads/_E-Learning_Design_Principles_and_Methods__v_4_2/e_learning_dp-4.2_27gtpdr5/Course_Syllabus"
    :return: A list of unique Questions
    """
    all_questions = []
    for fname in glob.iglob("**/*.html", root_dir=root_dir, recursive=True):
        all_questions.extend(parse_mcq(os.path.join(root_dir, fname)))
    all_questions = [Question(q) for q in all_questions]

    # Remove duplicates
    all_questions = list({repr(q): q for q in all_questions}.values())

    # Fold MCQs with identical content into one MCQ
    uniques = dict()
    for idx, q in enumerate(all_questions):
        skillref = q.pop("skillref", "")
        step_name = q.pop("step-name")

        q["id"] = f"elearning-mcq-{idx}"
        del q["oli-part-id"]
        q = uniques.setdefault(str(q), q)
        q.setdefault("skillref", []).append(skillref)
        q.setdefault("step-name", []).append(step_name)

    return list(uniques.values())


def load_datashop_temp(path: str) -> pd.DataFrame:
    """Load KC models from a DataShop template file"""
    return pd.read_csv(path, sep="\t", na_values=" ").dropna(axis="columns", how="all")


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
    with open(out_path, "w") as f:
        for q in elearning22:
            f.write(json.dumps(q.data) + "\n")
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
    with open(out_path, "w") as f:
        for q in elearning23:
            f.write(json.dumps(q.data) + "\n")
    print(f"Wrote {len(elearning23)} questions to {out_path}")


def get_step_name(kc_temp: str | pd.DataFrame, year: str, filter_by_kc: bool = False) -> pd.Series:
    if isinstance(kc_temp, str):
        kc_temp = pd.read_csv(kc_temp, sep="\t", na_values=" ").dropna(axis="columns", how="all")
    assert isinstance(kc_temp, pd.DataFrame), "Incorrect type for 'kc_temp'"

    match year:
        case "2022":
            step_name = kc_temp["Step Name"].apply(lambda x: x.split(" ")[0])
        case "2023":
            step_name = kc_temp["Step Name"].apply(lambda x: re.search(r"(?<=part ).+", x).group(0).split()[0])
        case _:
            raise ValueError(f"Unknown year {year}")

    kc_mask = kc_temp.filter(regex=KC_PAT).isna().any(axis=1)
    return step_name[~kc_mask] if filter_by_kc else step_name


def adjust_datashop_kc(data_path: str, kc_path: str, step_kc_path: str, save_to_file: bool = False,
                       old_to_new_kc: dict = None, new_kc_suffix: str = "new") -> pd.DataFrame:
    """
    Adjust existing DataShop KC models according to available questions
    :param data_path: A path to a jsonl file containing questions, e.g., "data/elearning/elearning22-mcq.jsonl"
    :param kc_path: A path to a (filled) DataShop KC template file, e.g., "data/datashop/ds5426-elearning/ds5426_kcm.txt"
    :param step_kc_path: A path to a (system-generated) unique-step KC model,
        e.g., "data/datashop/ds5426-elearning/unique-step.txt"
    :param save_to_file: Whether to save the new KC models to a file
    :param old_to_new_kc: A mapping between old and new KC names
    :param new_kc_suffix: If `old_to_new_kc` is not provided, extend old KC names by `new_kc_suffix`
    :return: A modified KC model as a DataFrame
    """
    # Load the existing KC model and identify the KC mask
    kc = load_datashop_temp(kc_path)
    kc_mask = kc.filter(regex=KC_PAT).isna().any(axis=1)

    # Extract existing KC models
    kc_names = [re.match(KC_PAT, col).group("name") for col in kc.filter(regex=KC_PAT).columns]

    # Load questions and identify the problem mask
    with open(data_path, "r") as f:
        questions = [Question(eval(line)) for line in f]

    ds_step_names = set(itertools.chain.from_iterable(q["ds-step-name"] for q in questions))
    prob_mask = ~kc["Step Name"].isin(ds_step_names)

    # Load the unique-step KC model
    step_kc = load_datashop_temp(step_kc_path)
    step_mask = step_kc["KC (Unique-step)"].isna()

    # Empty any cells where the problem name is not found in available questions
    mask = kc_mask | prob_mask | step_mask
    kc.loc[mask, [f"KC ({kcm})" for kcm in kc_names]] = None

    # Adjust old-to-new KC mappings
    old_to_new_kc = old_to_new_kc or {}
    old_to_new_kc = {f"KC ({key})": f"KC ({val})" for key, val in old_to_new_kc.items()}
    default_mapping = {f"KC ({kcm})": f"KC ({kcm.replace(' ', '-')}-{new_kc_suffix})" for kcm in kc_names}
    old_to_new_kc = default_mapping | old_to_new_kc

    # Rename and save KC models
    kc.rename(columns=old_to_new_kc, inplace=True)
    if save_to_file:
        kc.to_csv(f"{os.path.splitext(kc_path)[0]}-new.txt", sep="\t", index=False)
    return kc


def create_datashop_kc(kc: str | pd.DataFrame, kc_temp: str | pd.DataFrame,
                       step_kc_path: str, new_kc_name: str,
                       match_other_kc: bool = True, drop_other_kc: bool = True) -> pd.DataFrame:
    """
    Populate a custom Datashop KC model
    :param kc: Either a path to a human-readable KC model or a pd.DataFrame of such
    :param kc_temp: Either a path to DataShop KC template file,
            e.g., "data/datashop/ds5426-elearning/kc_temp.txt",
            or a pd.DataFrame of a loaded template
    :param step_kc_path: A path to a (system-generated) unique-step KC model,
            e.g., "data/datashop/ds5426-elearning/unique-step.txt"
    :param new_kc_name: The name given to the new KC model, e.g., "KCluster"
    :param match_other_kc: Whether to match other KC models in the template;
            if True, the new KC model will not map to steps that are not mapped by other KC models
    :param drop_other_kc: Whether to drop other KC models in the template
    :return: The new DataShop KC model as a DataFrame
    """
    # Load KC model
    if isinstance(kc, str):
        kc = pd.read_csv(kc)
    assert isinstance(kc, pd.DataFrame), "Incorrect type for 'kc'"

    # Load KC template
    if isinstance(kc_temp, str):
        kc_temp = load_datashop_temp(kc_temp)
    assert isinstance(kc_temp, pd.DataFrame), "Incorrect type for 'kc_temp'"

    kc_mask = False
    if match_other_kc:
        kc_mask = kc_temp.filter(regex=KC_PAT).isna().any(axis=1)  # match other KC models if any

    if drop_other_kc:
        kc_cols = kc_temp.filter(regex=KC_PAT).columns
        kc_temp.drop(columns=kc_cols, inplace=True)  # drop other KC models if any

    # Load the unique-step KC model
    step_kc = load_datashop_temp(step_kc_path)
    step_mask = step_kc["KC (Unique-step)"].isna()

    # Fill in KC
    steps, labels = [], []
    for step, label in kc[["ds-step-name", "KC"]].itertuples(index=False):
        step = step.split("~")
        steps.extend(step)
        labels.extend([label] * len(step))
    step_to_kc = dict(zip(steps, labels, strict=True))

    kc_temp[f"KC ({new_kc_name})"] = kc_temp["Step Name"].map(step_to_kc)
    kc_temp.loc[kc_mask | step_mask, f"KC ({new_kc_name})"] = None

    return kc_temp
