import os
import re
from collections import Counter

import pandas as pd

from core.question import Question
from processing.util import adjust_existing_kc

good_to_bad = {  # correct -> wrong
    # MCQ
    "What does the yield sign (→) do in a chemical equation?":
        "What does the yield sign (?) do in a chemical equation?",
    # TOF
    "For density calculations of solids and liquids in the metric system, the units g/cm³ and g/mL are used, respectively.":
        "For density calculations of solids and liquids in the metric system, the units g/cm� and g/mL are used, respectively.",
    "Water has a density of 1 g/cm³, serving as a standard reference for density comparisons.":
        "Water has a density of 1 g/cm�, serving as a standard reference for density comparisons.",
    "Electrons are crucial for chemical bonding.":
        "Electrons are crucial for chemical bonding",
    # FIB
    "In the metric system, the density of solids is measured in g/cm³, and for liquids, it is measured in _____.":
        "In the metric system, the density of solids is measured in g/cm�, and for liquids, it is measured in _____.",
    "The density of water, used as a benchmark for comparison, is _____ g/cm³.":
        "The density of water, used as a benchmark for comparison, is _____ g/cm�.",
    "The yield sign (→) in a chemical equation ______ reactants from products.":
        "The yield sign (?) in a chemical equation ______ reactants from products.",
    "When forces are _____, an object’s motion does not change.":
        "When forces are _____, an object?s motion does not change.",
    # SHA
    'What classification applies to elements on the left side of the periodic table’s "staircase", excluding hydrogen?':
        'What classification applies to elements on the left side of the periodic table?s "staircase", excluding hydrogen?'
}


def write_mcq(tx_path: str, excel_path: str, out_path: str = "data/spacing"):
    # Read all questions from the transaction file
    tx = pd.read_csv(tx_path, sep="\t", dtype={"Anon Student Id": str})
    mask = (tx["CF (Question Type)"] == "select_one_multiple_choice") & (~tx["Selection"].isin(["FALSE", "TRUE"]))
    prob_names = set(tx[mask]["Problem Name"])

    # Search through the Excel file for questions
    questions = []
    all_stems = set()
    dfs = pd.read_excel(excel_path, sheet_name=None)
    for name, df in dfs.items():
        for row_idx, row in df.iterrows():
            q_dict = {"id": f"spacing-{name}-mcq-{row_idx}", "type": "Multiple Choice"}

            stem, *options = list(filter(bool, row["Multiple-choice"].split("\n")))
            if (stem in all_stems) or (good_to_bad.get(stem, stem) not in prob_names):
                continue
            else:
                all_stems.add(stem)

            options = list(map(lambda s: re.split(r"[).]\s", s), options))
            q_dict["question"] = {"stem": stem}
            q_dict["question"]["choices"] = [{"label": label.lower(), "text": text} for label, text in options]
            q_dict["answerKey"] = re.split(r"[).]\s", row["ANSWER"])[0].lower()

            q_dict["LO"] = row["Success Criteria"].rstrip(".")
            q_dict["Topic"] = row["Unit"]
            q_dict["problem-name"] = good_to_bad.get(stem, stem)

            questions.append(q_dict)

    with open(f"{out_path}/spacing-mcq.jsonl", "w") as f:
        f.write("\n".join([repr(q) for q in questions]))

    print(f"Wrote {len(questions)} questions to {out_path}/spacing-mcq.jsonl")


def write_tof(tx_path: str, excel_path: str, out_path: str = "data/spacing"):
    # Read all questions from the transaction file
    tx = pd.read_csv(tx_path, sep="\t", dtype={"Anon Student Id": str})
    mask = (tx["CF (Question Type)"] == "select_one_multiple_choice") & (tx["Selection"].isin(["FALSE", "TRUE"]))
    prob_names = set(tx[mask]["Problem Name"])

    # Search through the Excel file for questions
    questions = []
    all_stems = set()
    dfs = pd.read_excel(excel_path, sheet_name=None)
    for name, df in dfs.items():
        for row_idx, row in df.iterrows():
            q_dict = {"id": f"spacing-{name}-tof-{row_idx}", "type": "True or False"}
            stem = str(row["True/False"])

            if stem == "Energy levels' refer to the specific paths that individual electrons follow around the nucleus.":
                stem = "'" + stem

            if stem == "Electrons are crucial for chemical bonding":
                stem = stem + "."

            if (stem in all_stems) or (good_to_bad.get(stem, stem) not in prob_names):
                continue
            else:
                all_stems.add(stem)

            q_dict["question"] = {"stem": stem}
            q_dict["answerKey"] = str(row["ANSWER.3"]).strip().capitalize()

            q_dict["LO"] = row["Success Criteria"].rstrip(".")
            q_dict["Topic"] = row["Unit"]
            q_dict["problem-name"] = good_to_bad.get(stem, stem)

            questions.append(q_dict)

    with open(f"{out_path}/spacing-tof.jsonl", "w") as f:
        f.write("\n".join([repr(q) for q in questions]))

    print(f"Wrote {len(questions)} questions to {out_path}/spacing-tof.jsonl")


def write_fib(tx_path: str, excel_path: str, out_path: str = "data/spacing"):
    # Read all questions from the transaction file
    tx = pd.read_csv(tx_path, sep="\t", dtype={"Anon Student Id": str})
    mask = tx["CF (Question Type)"] == "short_answer"
    prob_names = set(tx[mask]["Problem Name"])

    # Search through the Excel file for questions
    questions = []
    all_stems = set()
    dfs = pd.read_excel(excel_path, sheet_name=None)
    for name, df in dfs.items():
        for row_idx, row in df.iterrows():
            q_dict = {"id": f"spacing-{name}-fib-{row_idx}", "type": "Fill in the Blank"}
            stem = str(row["Fill in blanks"])

            if (stem in all_stems) or (good_to_bad.get(stem, stem) not in prob_names):
                continue
            else:
                all_stems.add(stem)

            q_dict["question"] = {"stem": stem}
            q_dict["answerKey"] = str(row["ANSWER.1"]).strip()

            q_dict["LO"] = row["Success Criteria"].rstrip(".")
            q_dict["Topic"] = row["Unit"]
            q_dict["problem-name"] = good_to_bad.get(stem, stem)

            questions.append(q_dict)

    with open(f"{out_path}/spacing-fib.jsonl", "w") as f:
        f.write("\n".join([repr(q) for q in questions]))

    print(f"Wrote {len(questions)} questions to {out_path}/spacing-fib.jsonl")


def write_sha(tx_path: str, excel_path: str, out_path: str = "data/spacing"):
    # Read all questions from the transaction file
    tx = pd.read_csv(tx_path, sep="\t", dtype={"Anon Student Id": str})
    mask = tx["CF (Question Type)"] == "free_response"
    prob_names = set(tx[mask]["Problem Name"])

    # Search through the Excel file for questions
    questions = []
    all_stems = set()
    dfs = pd.read_excel(excel_path, sheet_name=None)
    for name, df in dfs.items():
        for row_idx, row in df.iterrows():
            q_dict = {"id": f"spacing-{name}-sha-{row_idx}", "type": "Short Answer"}
            stem = str(row["Short answer"])

            if (stem in all_stems) or (good_to_bad.get(stem, stem) not in prob_names):
                continue
            else:
                all_stems.add(stem)

            q_dict["question"] = {"stem": stem}
            q_dict["answerKey"] = str(row["ANSWER.2"]).strip()  # .capitalize()

            q_dict["LO"] = row["Success Criteria"].rstrip(".")
            q_dict["Topic"] = row["Unit"]
            q_dict["problem-name"] = good_to_bad.get(stem, stem)

            questions.append(q_dict)

    with open(f"{out_path}/spacing-sha.jsonl", "w") as f:
        f.write("\n".join([repr(q) for q in questions]))

    print(f"Wrote {len(questions)} questions to {out_path}/spacing-sha.jsonl")


def write_all(data_path: str = "data/spacing"):
    questions = []
    for q_type in ("mcq", "tof", "fib", "sha"):
        with open(f"{data_path}/spacing-{q_type}.jsonl") as f:
            questions.extend([eval(line) for line in f])

    with open(f"{data_path}/spacing-all.jsonl", "w") as f:
        f.write("\n".join([repr(q) for q in questions]))

    print(f"Wrote {len(questions)} questions to {data_path}/spacing-all.jsonl")


def load_questions(data_path: str, temp_path: str) -> tuple[list[Question], pd.DataFrame]:
    """
    Load all questions and check integrity against a template file
    :param data_path: A path to a jsonl file containing questions, e.g., "data/spacing/spacing-all.jsonl"
    :param temp_path: A path to a DataShop KC template file (new or old), e.g., "data/datashop/ds6160/kc_temp.txt"
    :return: A list of loaded questions and the KC template as a DataFrame
    """
    Q_CNT = {"Multiple Choice": 805, "True or False": 819, "Fill in the Blank": 815, "Short Answer": 808}

    # Load all questions from data file
    with open(data_path, "r") as f:
        questions = [Question(eval(line)) for line in f]

    # Load all questions from KC template
    kc_temp = pd.read_csv(temp_path, sep="\t").dropna(axis="columns", how="all")
    prob_names = set(kc_temp["Problem Name"])

    # Adjust stems and check existence in KC template
    for q in questions:
        q["question"]["stem"] = good_to_bad.get(q.stem, q.stem)
        assert q.stem in prob_names, f"'{q.stem}' is not found in KC template"

    # Check the count of each question type
    cnt = Counter((q.q_type for q in questions))
    assert all(cnt[t] == Q_CNT[t] for t in cnt), "Counts of question types do not match ground truth"

    # If the data file contains all question types, check if it contains all questions
    if len(cnt) == len(Q_CNT):
        stems = set(q.stem for q in questions)
        assert prob_names.issubset(stems), "Not all questions are contained in the data file"

    return questions, kc_temp


def convert_existing_kc(data_path: str, kc_path: str, step_kc_path: str,
                        save_to_file: bool = False, **kwargs) -> pd.DataFrame:
    """
    Modify and save multiple existing KC models according to available questions
    :param data_path: A path to a jsonl file containing questions, e.g., "data/spacing/spacing-all.jsonl"
    :param kc_path: A path to a (filled) DataShop KC template file, e.g., "data/datashop/ds6160-spacing/ds6160_expert_kcm.txt"
    :param step_kc_path: A path to a (system-generated) unique-step KC model,
            e.g., "data/datashop/ds6160-spacing/unique-step.txt"
    :param save_to_file: Whether to save the new KC models to a file
    :return: A modified KC model as a DataFrame
    """
    # Load questions and the existing KC
    questions, kc = load_questions(data_path, kc_path)
    all_stems = set(q.stem for q in questions)
    prob_mask = kc["Problem Name"].apply(lambda x: x not in all_stems)

    kc = adjust_existing_kc(kc, prob_mask, step_kc_path, **kwargs)
    if save_to_file:
        kc.to_csv(f"{os.path.splitext(kc_path)[0]}-new.txt", sep="\t", index=False)
    return kc


def create_datashop_kc(kc_temp: str | pd.DataFrame, kc: str | pd.DataFrame,
                       step_kc_path: str, new_kc_name: str) -> pd.DataFrame:
    """
    Populate a custom Datashop KC model
    :param kc_temp: Either a path to a DataShop KC template file,
        e.g., "data/datashop/ds6160-spacing/kc_temp.txt",
        or a pd.DataFrame of a loaded template
    :param kc: Either a path to a human-readable KC model or a pd.DataFrame of such
    :param step_kc_path: A path to a (system-generated) unique-step KC model,
            e.g., "data/datashop/ds6160-spacing/unique-step.txt"
    :param new_kc_name: The name given to the new KC model, e.g., "KCluster"
    :return: The new KC model as a DataFrame
    """
    # Load KC template
    if isinstance(kc_temp, str):
        kc_temp = pd.read_csv(kc_temp, sep="\t").dropna(axis="columns", how="all")
    assert isinstance(kc_temp, pd.DataFrame), "Incorrect type for 'kc_temp'"

    if isinstance(kc, str):
        kc = pd.read_csv(kc)
    assert isinstance(kc, pd.DataFrame), "Incorrect type for 'kc'"

    # Load the unique-step KC model
    step_kc = pd.read_csv(step_kc_path, sep="\t").dropna(axis="columns", how="all")
    step_mask = ~step_kc["KC (Unique-step)"].str.strip().apply(bool)

    # Populate KC template
    stems_to_kcs = dict(zip(kc["problem-name"], kc["KC"]))
    kc_temp[f"KC ({new_kc_name})"] = kc_temp["Problem Name"].apply(stems_to_kcs.get)
    kc_temp.loc[step_mask, f"KC ({new_kc_name})"] = None

    return kc_temp
