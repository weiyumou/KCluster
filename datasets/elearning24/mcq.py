import json
import logging
import os
import random

import pandas as pd

from kcluster.io.datashop import load_datashop_temp

from processing import extract_activity_part_ids, extract_learning_objectives
from processing import extract_answer, merge_duplicate_mcqs
from processing import replace_unicode_chars, download_n_save_image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_stem(data: dict) -> tuple[str, list[str]]:
    """Extract text and images from the stem content."""
    parts, images = [], []
    for item in data["content"]["stem"]["content"]:
        match item["type"]:
            case "p":
                texts = []
                for child in item["children"]:
                    if "type" not in child:
                        texts.append(child["text"].strip())
                    elif child["type"] == "img_inline":
                        img_path = download_n_save_image(child["src"])
                        alt_text = child.get("alt", "image")
                        texts.append(f"![{alt_text}]({img_path})")
                        images.append(img_path)
                parts.append(" ".join(texts).strip())
            case "img":
                # Download the image
                img_path = download_n_save_image(item["src"])
                alt_text = item.get("alt", "image")
                parts.append(f"![{alt_text}]({img_path})")
                images.append(img_path)
            case "ol":
                ol_texts = []
                for idx, li in enumerate(item["children"], start=1):
                    assert li["type"] == "li"
                    [child] = li["children"]
                    if "children" in child:
                        child = child["children"][0]
                    ol_texts.append(f"{idx}. {child['text'].strip()}")
                parts.append("\n".join(ol_texts).strip())
            case "ul":
                # ul is used for block quotes
                ul_texts = []
                for li in item["children"]:
                    assert li["type"] == "li"
                    [child] = li["children"]
                    if "children" in child:
                        child = child["children"][0]
                    ul_texts.append(child["text"].strip())
                parts.append("> " + " ".join(ul_texts).strip())
            case "table":
                raise ValueError("Table found in stem, skipping...")
            case _:
                raise ValueError(f"Unknown item type: {item['type']}")
    stem = replace_unicode_chars("\n".join(parts)).strip()
    assert stem, "No stem text found"
    return stem, images


def extract_choices(data: dict, part_id: str, rng=None) -> tuple[list[dict], str, dict]:
    """Extract choices, correct answer, and feedback from the choices content."""
    # Extract choices
    choices = dict()
    for chc in data["content"]["choices"]:
        chc_texts = []
        for item in chc["content"]:
            if item.get("type", None) != "p":
                continue
            chc_texts.append(" ".join(child["text"].strip() for child in item["children"]).strip())
        chc_text = "; ".join(chc_texts).strip("; ")
        choices[chc["id"]] = replace_unicode_chars(chc_text).strip()

    assert len(choices) >= 2, f"Not enough choices found"

    # Extract correct answer and feedback
    [part] = data["content"]["authoring"]["parts"]
    assert part["id"] == part_id

    return extract_answer(part["responses"], choices, allow_unused_choices=True, rng=rng)


def extract_mcqs(step_df: pd.DataFrame, raw_data_dir: str, seed: int = 42) -> list[dict]:
    """Extract multiple-choice questions from a unique-step KC template."""
    rng = random.Random(seed)
    step_df = extract_activity_part_ids(step_df)

    all_questions = []
    activity_ids = []
    mask = step_df["Submission Type"].eq("Multiple choice submission")
    for activity_id, grp in step_df[mask].groupby("Activity ID"):
        if activity_id in {"42356"}:  # no correct answer
            continue
        if activity_id in {"42408", "98094", "99135", "41430", "41474", "42615"}:  # multiple answers
            continue

        data_file = os.path.join(raw_data_dir, f"{activity_id}.json")
        if not os.path.isfile(data_file):
            logging.warning(f"Data file not found for activity {activity_id}: {data_file}")
            continue

        with open(data_file, "r") as f:
            data = json.load(f)

        # All MCQs should have only one part        
        [part_id] = grp["Part ID"].unique()

        try:
            # Extract stem
            stem, images = extract_stem(data)
            # Extract choices
            all_choices, ans_option, feedback = extract_choices(data, part_id, rng)
        except Exception as e:
            logging.warning(f"Skipping activity {activity_id} due to error: {e}")
            continue

        # Extract objectives
        objectives = extract_learning_objectives(data, raw_data_dir)[part_id]

        q_dict = {
            "id": f"mcq-{activity_id}",
            "type": "Multiple Choice",
            "question": {"stem": stem, "choices": all_choices},
            "answerKey": ans_option,
            "images": images,
            "objectives": sorted(objectives),
            "feedback": feedback,
            "ds-problem-hierarchy": grp["Problem Hierarchy"].unique().tolist(),
            "ds-problem-name": grp["Problem Name"].unique().tolist(),
            "ds-step-name": grp["Step Name"].unique().tolist(),
        }
        all_questions.append(q_dict)
        activity_ids.append(activity_id)

    all_questions = merge_duplicate_mcqs(all_questions)
    logging.info(f"Extracted {len(all_questions)} unique MCQs from {len(activity_ids)} activities")
    return all_questions


def main(args):
    args.output_path = getattr(args, "output_path", os.path.join("elearning24", "data", f"elearning24-mcq.jsonl"))
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    step_df = load_datashop_temp(args.step_path)
    all_questions = extract_mcqs(step_df, args.raw_data_dir, seed=args.seed)

    with open(args.output_path, "w") as f_out:
        for q in all_questions:
            f_out.write(json.dumps(q) + "\n")

    logging.info(f"Extracted {len(all_questions)} questions to {args.output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract MCQs from DataShop export")
    parser.add_argument("--step_path", required=True, type=str, help="Path to a DataShop unique-step file")
    parser.add_argument("--raw_data_dir", required=True, type=str,
                        help="Directory containing raw JSON data files")
    parser.add_argument("--output_path", default=argparse.SUPPRESS, type=str, help="Output JSONL file path")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for shuffling choices")

    cli_args = parser.parse_args()
    main(cli_args)
