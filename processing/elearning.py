import glob
import itertools
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


def parse_all_mcqs(root_dir: str, temp_path: str = None) -> list[Question]:
    """
    Parse all unique MCQs from a root directory, with optional reference to a KC model file
    :param root_dir: A path to downloaded HTML files,
    e.g., "Downloads/_E-Learning_Design_Principles_and_Methods__v_4_2/e_learning_dp-4.2_27gtpdr5/Course_Syllabus"
    :param temp_path: An optional path to a KC model file, e.g., "data/datashop/ds5426-elearning/kc_temp.txt"
    :return: A list of unique Questions
    """
    all_questions = []
    for fname in glob.iglob("**/*.html", root_dir=root_dir, recursive=True):
        all_questions.extend(parse_mcq(os.path.join(root_dir, fname)))
    all_questions = [Question(q) for q in all_questions]

    # Remove duplicates
    all_questions = list({repr(q): q for q in all_questions}.values())

    # If path to a KC template is provided, retain only MCQs in the template
    if temp_path:
        kc_temp = pd.read_csv(temp_path, sep="\t")
        step_names = set(kc_temp["Step Name"].apply(lambda x: x.split(" ")[0]))
        all_questions = [q for q in all_questions if q["step-name"] in step_names]

    # Fold MCQs with identical content into one
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


def write_mcqs(root_dir: str, out_dir: str, temp_path: str = None):
    """
    Extract MCQs from 'root_dir' and write them to 'out_dir'
    :param root_dir: A path to downloaded HTML files,
    e.g., "Downloads/_E-Learning_Design_Principles_and_Methods__v_4_2/e_learning_dp-4.2_27gtpdr5/Course_Syllabus"
    :param out_dir: A path to an output dir, e.g., data/elearning/
    :param temp_path: An optional path to a KC model file, e.g., "data/datashop/ds5426-elearning/kc_temp.txt"
    :return: None
    """
    mcqs = parse_all_mcqs(root_dir, temp_path)

    # Write MCQs to a JSON file for program readability
    out_path = os.path.join(out_dir, "elearning-mcq.jsonl")
    with open(out_path, "w") as f:
        f.write("\n".join([repr(q) for q in mcqs]))
    print(f"Wrote {len(mcqs)} questions to {out_path}")

    # Write MCQs to a CSV file for human readability
    q_dicts = [q.flat_dict for q in mcqs]
    pd.DataFrame.from_records(q_dicts).to_csv(os.path.join(out_dir, "elearning-mcq.csv"), index=False)


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


def convert_existing_kc(data_path: str, kc_path: str, step_kc_path: str, year: str,
                        save_to_file: bool = False, old_to_new_kc: dict = None,
                        new_kc_suffix: str = "new") -> pd.DataFrame:
    """
    Modify and save existing KC models according to available questions
    :param data_path: A path to a jsonl file containing questions, e.g., "data/elearning/elearning-mcq.jsonl"
    :param kc_path: A path to a (filled) DataShop KC template file, e.g., "data/datashop/ds5426-elearning/LOs.txt"
    :param step_kc_path: A path to a (system-generated) unique-step KC model,
        e.g., "data/datashop/ds5426-elearning/unique-step.txt"
    :param year: Specify which year of the dataset to use
    :param save_to_file: Whether to save the new KC models to a file
    :param old_to_new_kc: A mapping between old and new KC names
    :param new_kc_suffix: If `old_to_new_kc` is not provided, extend old KC names by `new_kc_suffix`
    :return: A modified KC model as a DataFrame
    """
    # Load the existing KC model and identify the KC mask
    kc = pd.read_csv(kc_path, sep="\t", na_values=" ").dropna(axis="columns", how="all")
    kc_mask = kc.filter(regex=KC_PAT).isna().any(axis=1)
    # Extract existing KC models
    kc_names = [re.match(KC_PAT, col).group("name") for col in kc.filter(regex=KC_PAT).columns]

    # Load questions and identify the problem mask
    with open(data_path, "r") as f:
        questions = [Question(eval(line)) for line in f]
    match year:
        case "2022":
            prob_ids = set(itertools.chain.from_iterable(q["step-name"] for q in questions))
        case "2023":
            prob_ids = set(x.split("_")[-1] for x in itertools.chain.from_iterable(q["step-name"] for q in questions))
        case _:
            raise ValueError(f"Unknown year '{year}'")
    prob_mask = get_step_name(kc, year).apply(lambda x: x not in prob_ids)

    # Load the unique-step KC model
    step_kc = pd.read_csv(step_kc_path, sep="\t", na_values=" ").dropna(axis="columns", how="all")
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
    kc = kc.rename(columns=old_to_new_kc)
    if save_to_file:
        kc.to_csv(f"{os.path.splitext(kc_path)[0]}-new.txt", sep="\t", index=False)
    return kc


def get_step_to_kc(kc: pd.DataFrame, year: str) -> dict[str, str]:
    steps, labels = [], []
    for step, label in kc[["step-name", "KC"]].itertuples(index=False):
        step = step.split("~")
        steps.extend(step)
        labels.extend([label] * len(step))
    if year == "2023":
        steps = [s.split("_")[-1] for s in steps]
    return dict(zip(steps, labels))


def create_datashop_kc(kc: str | pd.DataFrame, kc_temp: str | pd.DataFrame,
                       step_kc_path: str, new_kc_name: str, year: str) -> pd.DataFrame:
    """
    Populate a custom Datashop KC model
    :param kc: Either a path to a human-readable KC model or a pd.DataFrame of such
    :param kc_temp: Either a path to DataShop KC template file,
            e.g., "data/datashop/ds5426-elearning/kc_temp.txt",
            or a pd.DataFrame of a loaded template
    :param step_kc_path: A path to a (system-generated) unique-step KC model,
            e.g., "data/datashop/ds5426-elearning/unique-step.txt"
    :param new_kc_name: The name given to the new KC model, e.g., "KCluster"
    :param year: Specify which year of the dataset to use
    :return: The new DataShop KC model as a DataFrame
    """
    # Load KC model
    if isinstance(kc, str):
        kc = pd.read_csv(kc)
    assert isinstance(kc, pd.DataFrame), "Incorrect type for 'kc'"

    # Load KC template
    if isinstance(kc_temp, str):
        kc_temp = pd.read_csv(kc_temp, sep="\t", na_values=" ").dropna(axis="columns", how="all")
    assert isinstance(kc_temp, pd.DataFrame), "Incorrect type for 'kc_temp'"
    filter_by_kc = (kc_temp.filter(regex=KC_PAT).shape[1] != 0)

    # Load the unique-step KC model
    step_kc = pd.read_csv(step_kc_path, sep="\t", na_values=" ").dropna(axis="columns", how="all")
    step_mask = step_kc["KC (Unique-step)"].isna()

    # Fill in KC
    step_to_kc = get_step_to_kc(kc, year)
    kc_temp[f"KC ({new_kc_name})"] = get_step_name(kc_temp, year, filter_by_kc).apply(step_to_kc.get)
    kc_temp.loc[step_mask, f"KC ({new_kc_name})"] = None

    return kc_temp
