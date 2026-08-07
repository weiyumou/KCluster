import copy
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from io import BytesIO

import pandas as pd
import requests
from PIL import Image


def replace_unicode_chars(text: str) -> str:
    """Replace specific Unicode characters with standard ASCII equivalents."""
    trans_map = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "​": " "})
    return text.translate(trans_map)


def download_n_save_image(url: str, raw_data_dir: str = "elearning24/raw_data") -> str:
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
    except Exception as e:
        if url == "../../../webcontent/BIGPICTURE.png":
            img = Image.open(os.path.join(raw_data_dir, "resources", "BIGPICTURE.png"))
        else:
            logging.warning(f"Failed to download image from {url}: {e}")
            return url

    os.makedirs(os.path.join(raw_data_dir, "images"), exist_ok=True)
    fname, ext = os.path.splitext(os.path.basename(url))
    fname = hashlib.md5(fname.encode()).hexdigest()[:16] + ext
    img_path = os.path.join(raw_data_dir, "images", fname)
    img.save(img_path)
    return img_path


def extract_activity_part_ids(step_df: pd.DataFrame) -> pd.DataFrame:
    """Extract Activity ID and Part ID from the 'Step Name' column and add them as new columns."""
    pat = r"Activity (\d+), Part ([\w\d]+) (.+)"
    df = pd.DataFrame(step_df["Step Name"].apply(lambda x: re.match(pat, x).groups()).to_list(),
                      columns=["Activity ID", "Part ID", "Submission Type"])
    step_df = pd.concat([step_df, df], axis=1)
    return step_df


def extract_learning_objectives(data: dict, raw_data_dir: str) -> dict[str, list[str]]:
    objectives = defaultdict(list)
    for part_id in data["objectives"]:
        for obj_id in data["objectives"][part_id]:
            obj_file = os.path.join(raw_data_dir, f"{obj_id}.json")
            if not os.path.isfile(obj_file):
                continue
            with open(obj_file, "r") as f:
                obj_data = json.load(f)
            assert obj_data["type"] == "Objective"

            obj = replace_unicode_chars(obj_data["title"]).strip()
            obj = obj[0].upper() + obj[1:]
            objectives[part_id].append(obj)
    return objectives


def extract_answer(responses: list[dict], choices: dict[str, str], allow_unused_choices: bool = True, rng=None):
    ans_id = None
    sel_choice_ids = []
    feedback = dict()
    for rsp in responses:
        choice_id = re.search(r"\{([^}]+)\}", rsp["rule"]).group(1)
        if choice_id == ".*":
            choice_id = "default"

        if choice_id in choices:
            sel_choice_ids.append(choice_id)
        elif choice_id != "default":
            if allow_unused_choices:
                continue
            raise ValueError(f"Unknown choice ID: {choice_id}")

        if rsp["score"] > 0:
            assert ans_id is None, f"Multiple correct answers found"
            ans_id = choice_id

        # Collect feedback
        texts = []
        for chc in rsp["feedback"]["content"]:
            if chc.get("type", None) != "p":
                continue
            child_texts = []
            for child in chc["children"]:
                if "children" in child:
                    [child] = child["children"]
                child_texts.append(child["text"].strip())
            texts.append(" ".join(child_texts).strip())
        feedback[choice_id] = replace_unicode_chars("\n".join(texts)).strip()

    assert set(feedback.keys()).issubset(set(choices.keys()) | {"default"})
    assert ans_id is not None, f"No correct answer found"

    all_choices = []
    ans_option = None
    if rng is not None:
        sel_choice_ids = rng.sample(sel_choice_ids, k=len(sel_choice_ids))
    for idx, cid in enumerate(sel_choice_ids):
        option = chr(ord("a") + idx)
        if cid == ans_id:
            ans_option = option
        chc_text = replace_unicode_chars(choices[cid]).strip()
        assert chc_text, "Empty choice text found"
        all_choices.append({"label": option, "text": chc_text})
        feedback[option] = feedback.pop(cid, feedback.get("default", None))

    feedback.pop("default", None)
    assert all_choices, "No valid choices found"
    return all_choices, ans_option, feedback


def merge_duplicate_mcqs(questions: list[dict]) -> list[dict]:
    unique_questions = dict()
    for q in questions:
        stem = q["question"]["stem"]
        choices = [chc["text"] for chc in q["question"]["choices"]]
        sig = frozenset([stem] + choices)  # the signature of an MCQ
        if sig not in unique_questions:
            unique_questions[sig] = copy.deepcopy(q)
        else:
            old_q = unique_questions[sig]
            old_q["id"] += f"-{q['id'].split('-')[-1]}"

            old_q["images"] = list(set(old_q["images"] + q["images"]))
            old_q["objectives"] = sorted(set(old_q["objectives"] + q["objectives"]))
            old_q["ds-problem-hierarchy"] = list(set(old_q["ds-problem-hierarchy"] + q["ds-problem-hierarchy"]))
            old_q["ds-problem-name"] = list(set(old_q["ds-problem-name"] + q["ds-problem-name"]))
            old_q["ds-step-name"] = list(set(old_q["ds-step-name"] + q["ds-step-name"]))

    return list(unique_questions.values())


def reorganize_transactions(raw_tx_path: str) -> pd.DataFrame:
    key_cols = ["Anon Student Id", "Session Id", "Time", "Time Zone",
                "Student Response Type", "Tutor Response Type",
                "Problem Name", "Problem Start Time", "Step Name", "Outcome", "Input"]
    level_cols = ["Level (Container)", "Level (Container).1", "Level (Container).2", "Level (Page)"]
    kc_cols = ['KC (Default)', 'KC (Default).1']
    tx_df = pd.read_csv(raw_tx_path, sep="\t", dtype=str, usecols=key_cols + level_cols + kc_cols)

    # Combine the KC fields
    kc = tx_df.filter(like="KC (Default)").fillna("").apply(
        lambda row: "~~".join(sorted(t for t in row.to_list() if t)), axis=1)
    tx_df = tx_df.drop(columns=kc_cols)
    tx_df["KC (Default)"] = kc

    # Make the problem hierarchy a custom field
    def replace(row, category):
        row = row.to_list()
        return "".join(f"({category}) {x}, " for x in row if x).strip().rstrip(",")

    container = tx_df.filter(like="Level (Container)").fillna("").apply(replace, args=("Container",), axis=1)
    page = tx_df.filter(like="Level (Page)").fillna("").apply(replace, args=("Page",), axis=1)
    ph = pd.concat([container, page], axis=1).apply(lambda row: ", ".join(t for t in row.to_list() if t), axis=1)
    tx_df["CF (Problem Hierarchy)"] = ph
    tx_df = tx_df.drop(columns=level_cols)

    # MD5 hash for Session ID
    import hashlib
    tx_df["Session Id"] = tx_df["Session Id"].apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:32])

    tx_df.insert(6, "Level (Course)", "E-learning 24")
    return tx_df
