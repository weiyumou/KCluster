"""Loader for OLI (Open Learning Initiative) course HTML exports.

Parses multiple-choice questions out of downloaded OLI course pages into the
Question JSONL format, keeping each question's OLI part id, ``skillref`` and
derived step name so the questions can later be joined against DataShop
exports of the same course.

Requires BeautifulSoup (the ``datashop`` extra).
"""

import glob
import os
import re

from bs4 import BeautifulSoup

from kcluster.core.question import Question


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


def parse_all_mcqs(root_dir: str, id_prefix: str = "elearning-mcq") -> list[Question]:
    """
    Parse all unique MCQs from a root directory
    :param root_dir: A path to downloaded HTML files,
    e.g., "Downloads/_E-Learning_Design_Principles_and_Methods__v_4_2/e_learning_dp-4.2_27gtpdr5/Course_Syllabus"
    :param id_prefix: A prefix for the re-assigned question ids, e.g., "elearning-mcq"
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

        q["id"] = f"{id_prefix}-{idx}"
        del q["oli-part-id"]
        q = uniques.setdefault(str(q), q)
        q.setdefault("skillref", []).append(skillref)
        q.setdefault("step-name", []).append(step_name)

    return list(uniques.values())
