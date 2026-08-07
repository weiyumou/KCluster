import json
import logging
import os
import random

import pandas as pd

from kcluster.io.datashop import load_datashop_temp

from processing import extract_activity_part_ids, extract_learning_objectives
from processing import extract_answer, merge_duplicate_mcqs
from processing import replace_unicode_chars

SPACE = chr(32)
BLANK = "____"
Q_TYPE = "Multi input submission"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_elements(item: dict) -> list:
    elements = []
    match item["type"]:
        case "ul":
            for c in item["children"]:
                assert c["type"] == "li", f"Unordered list contains non-li item: {c}"
                if len(c["children"]) == 1:
                    [ch] = c["children"]
                    if "children" in ch:
                        elements.append(ch["children"])
                    else:
                        elements.append([ch])
                else:
                    elements.append(c["children"])
        case "p":
            elements.append(item["children"])
        case "table":
            raise ValueError("Table found in stem, skipping...")
        case _:
            raise ValueError(f"Unknown item type: {item['type']}")

    return elements


def parse_stem(activity_id: str, inputs: dict, parts: dict, choices: dict, children: list, rng=None):
    texts, blanks = [], []
    for child in children:
        if "type" not in child:
            texts.append(child["text"].strip() + " ")
        elif child["type"] == "input_ref":  # Finish parsing a problem
            input_id = child["id"]
            part_id = inputs[input_id]["part_id"]
            problem_name = f"Activity {activity_id}, Part {part_id}"
            step_name = f"{problem_name}{SPACE}{Q_TYPE}"
            texts.append(f"{BLANK}{SPACE}")

            all_choices, ans_option, feedback = extract_answer(parts[part_id]["responses"], choices,
                                                               allow_unused_choices=False, rng=rng)

            answer = next((chc["text"] for chc in all_choices if chc["label"] == ans_option), None)
            blanks.append({"part_id": part_id, "ds-problem-name": problem_name, "ds-step-name": step_name,
                           "choices": all_choices, "answerKey": ans_option, "answer": answer, "feedback": feedback})
    return texts, blanks


def extract_mfb(step_df: pd.DataFrame, raw_data_dir: str, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    step_df = extract_activity_part_ids(step_df)

    all_questions = []
    activities = []
    mask = step_df["Submission Type"].eq(Q_TYPE)
    for activity_id, grp in step_df[mask].groupby("Activity ID"):
        data_file = os.path.join(raw_data_dir, f"{activity_id}.json")
        if not os.path.isfile(data_file):
            logging.warning(f"Data file not found for activity {activity_id}: {data_file}")
            continue

        with open(data_file, "r") as f:
            data = json.load(f)

        all_part_ids = set(grp["Part ID"].tolist())

        # Extract inputs, choices, parts, and objectives
        inputs = {
            item["id"]: {"part_id": item["partId"], "choices": item["choiceIds"]} for item in data["content"]["inputs"]}
        choices = dict()
        for item in data["content"]["choices"]:
            [content] = item["content"]
            if "children" in content:
                [content] = content["children"]
            choices[item["id"]] = content["text"].strip()

        parts = {item["id"]: {"responses": item["responses"]} for item in data["content"]["authoring"]["parts"]}
        assert set(parts.keys()) == all_part_ids, f"Part IDs mismatch for activity {activity_id}"
        objectives = extract_learning_objectives(data, raw_data_dir)

        all_texts, all_blanks = [], []
        try:
            for item in data["content"]["stem"]["content"]:
                meta_children = parse_elements(item)
                for children in meta_children:
                    texts, blanks = parse_stem(activity_id, inputs, parts, choices, children, rng)
                    all_texts.extend(texts)
                    all_blanks.extend(blanks)
                    all_texts.append("\n")
        except Exception as e:
            logging.warning(f"Skipping activity {activity_id} due to error: {e}")
            continue

        if not all_blanks:
            logging.warning(f"Skipping activity {activity_id} due to no blank in stem")
            continue

        stem = replace_unicode_chars("".join(all_texts)).strip()
        parts = stem.split(BLANK)
        for i, blank in enumerate(all_blanks):  # Each blank will be a question
            assert blank["part_id"] in all_part_ids, f"Unknown part ID: {blank['part_id']}"
            repls = [BLANK] * len(all_blanks)
            for j in range(len(repls)):
                if j != i:
                    repls[j] = "*" + all_blanks[j]["answer"] + "*"
            new_stem = "".join(p + r for p, r in zip(parts, repls)) + parts[-1]

            ph = grp[grp["Step Name"].eq(blank["ds-step-name"])]["Problem Hierarchy"].unique().tolist()
            q_dict = {
                "id": f"mfb-{activity_id}_{blank['part_id']}",
                "type": "Multiple Choice",
                "question": {"stem": new_stem, "choices": blank["choices"]},
                "answerKey": blank["answerKey"],
                "images": [],
                "objectives": sorted(objectives[blank["part_id"]]),
                "feedback": blank["feedback"],
                "ds-problem-hierarchy": ph,
                "ds-problem-name": [blank["ds-problem-name"]],
                "ds-step-name": [blank["ds-step-name"]],
            }

            all_questions.append(q_dict)
        activities.append(activity_id)

    all_questions = merge_duplicate_mcqs(all_questions)
    logging.info(f"Extracted {len(all_questions)} questions from {len(activities)} activities")
    return all_questions


def main(args):
    args.output_path = getattr(args, "output_path",
                               os.path.join("elearning24", "data", "elearning24-mfb.jsonl"))
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    step_df = load_datashop_temp(args.step_path)
    all_questions = extract_mfb(step_df, args.raw_data_dir, seed=args.seed)

    with open(args.output_path, "w") as f_out:
        for q_dict in all_questions:
            f_out.write(json.dumps(q_dict) + "\n")

    logging.info(f"Extracted {len(all_questions)} questions to {args.output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract MFBs from DataShop export")
    parser.add_argument("--step_path", required=True, type=str, help="Path to a DataShop unique-step file")
    parser.add_argument("--raw_data_dir", required=True, type=str,
                        help="Directory containing raw JSON data files")
    parser.add_argument("--output_path", default=argparse.SUPPRESS, type=str, help="Output JSONL file path")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for shuffling choices")
    cli_args = parser.parse_args()

    main(cli_args)
