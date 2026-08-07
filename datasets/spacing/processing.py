import random

import pandas as pd

Q_TYPES = {"Multiple Choice": "mcq", "Short Answer": "sha", "True or False": "tof", "Fill in Blank": "fib"}


def clean_transactions(raw_tx_path: str) -> pd.DataFrame:
    """This function cleans a raw transaction file"""
    key_cols = ["Anon Student Id", "Session Id", "Time", "Level (Course)", "Problem Name",
                "Problem Start Time", "Step Name", "Outcome", "Input",
                "Condition Name", "Condition Type", "Condition Name.1", "Condition Type.1",
                "School", "Class"]
    cf_cols = [
        "CF (Question Id)", "CF (Full Problem Name)", "CF (Question Type)",
        "CF (Answer Options)", "CF (Correct Answer Options)",
        "CF (Topic Text)", "KC (question_group)",
        "CF (Stage)", "CF (Assignment Day)",
        "CF (Response Time)", "CF (Completion Time)",
        "CF (Anon Teacher Id)", "CF (Course Name)",
        "CF (Timed Out)", "CF (District Id)"
    ]
    df = pd.read_csv(raw_tx_path, sep="\t", dtype=str)
    df = df[key_cols + cf_cols].rename(
        columns={"Condition Name.1": "Condition Name",
                 "Condition Type.1": "Condition Type",
                 "KC (question_group)": "CF (Question Group)"})
    mask = (
        df["CF (Stage)"].isin(["pre-test", "learning", "post-test", "post-post-test"])
    )
    df = df[mask].reset_index(drop=True)
    df = df.apply(lambda x: x.str.replace("\n", " "), axis=0)
    df = df.apply(lambda x: x.str.replace('"', "'"), axis=0)
    df = df.apply(lambda x: x.str.strip(), axis=0)

    # Create unique question IDs
    df["question"] = df["CF (Full Problem Name)"] + df["CF (Answer Options)"].fillna("")  # temporary key

    tmp = df.groupby("CF (Question Type)", sort=False)["question"].apply(
        lambda grp: pd.DataFrame({"question": grp.unique()})).reset_index()
    q_type = tmp["CF (Question Type)"].map(Q_TYPES)
    tmp["CF (Question ID)"] = q_type + "-" + tmp["level_1"].apply(str)
    tmp = tmp.drop(columns=["level_1"])
    tmp = tmp.set_index(["CF (Question Type)", "question"])

    df = df.join(tmp, on=["CF (Question Type)", "question"], how="left")
    df = df.drop(columns=["CF (Question Id)", "question"])

    # Make Step Name unique to each distinct question
    df["Step Name"] = df["CF (Question ID)"]
    df["CF (Question Group)"] = "QG-" + df["CF (Question Group)"]

    # Format completion time
    df["CF (Completion Time)"] = pd.to_datetime(df["CF (Completion Time)"].astype(int), unit="ms")
    df["CF (Completion Time)"] = df["CF (Completion Time)"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df


def extract_questions_by_type(tx_df: pd.DataFrame, question_type: str, rng=None) -> list[dict]:
    """This function extracts a question DataFrame from a cleaned transaction DataFrame."""
    assert question_type in Q_TYPES, f"Unknown question type: {question_type}"
    rng = rng or random.Random()

    key_cols = ["CF (Question ID)", "CF (Question Type)", "CF (Full Problem Name)",
                "CF (Correct Answer Options)", "Level (Course)", "Problem Name", "Step Name",
                "CF (Topic Text)", "CF (Question Group)"]
    mcq_cols = ["CF (Answer Options)"]
    new_cols = ["id", "type", "question", "answerKey",
                "ds-course", "ds-problem-name", "ds-step-name",
                "standard", "q-group"]

    mask = tx_df["CF (Question Type)"].eq(question_type)
    if question_type == "Multiple Choice":
        df = tx_df.loc[mask, key_cols + mcq_cols]
    else:
        df = tx_df.loc[mask, key_cols]

    df = df.rename(columns=dict(zip(key_cols, new_cols, strict=True)))
    df = df.drop_duplicates(subset=["id"], ignore_index=True)

    if question_type == "Multiple Choice":
        df["choices"] = df["CF (Answer Options)"].str.split("|").apply(lambda x: rng.sample(x, k=len(x)))
        df["choices"] = df["choices"].apply(
            lambda choices: [{"label": chr(ord("a") + idx), "text": chc} for idx, chc in enumerate(choices)])
        df = df.drop(columns=mcq_cols)

        def get_ans_choice(row):
            answer, choices = row["answerKey"], row["choices"]
            for choice in choices:
                if choice["text"] == answer:
                    return choice["label"]
            return None

        df["answerKey"] = df.apply(get_ans_choice, axis=1)
    else:
        def get_ans(s):
            options = s.split("|")
            return options[0] if options[0] else options[-1]

        df["answerKey"] = df["answerKey"].apply(get_ans)

    df["question"] = df["question"].str.strip().str.capitalize()
    df["answerKey"] = df["answerKey"].str.strip()
    assert df.notna().all().all(), f"NaN values found in {question_type} DataFrame"

    if question_type == "Multiple Choice":
        df["question"] = df.apply(lambda row: {"stem": row["question"], "choices": row["choices"]}, axis=1)
        df = df.drop(columns=["choices"])
    else:
        df["question"] = df["question"].apply(lambda s: {"stem": s})

    return df.to_dict(orient="records")


if __name__ == "__main__":
    import argparse
    import warnings
    from collections import defaultdict, Counter
    import json
    import logging
    import os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = argparse.ArgumentParser(description="Extract questions from a DataShop transaction file.")
    parser.add_argument("--raw_tx_path", type=str, required=True, help="Path to a raw DataShop transaction file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling choices in MCQs")
    args = parser.parse_args()

    rng_ = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tx = clean_transactions(args.raw_tx_path)

    all_questions = defaultdict(list)
    for course, df_ in tx.groupby("Level (Course)", sort=False):
        for q_type_ in Q_TYPES:
            all_questions[course].extend(extract_questions_by_type(df_, q_type_, rng=rng_))

        # Display statistics and save questions to JSON files
        cnt = Counter(q["type"] for q in all_questions[course])
        logging.info(f"*** Extracted the following number of questions for course '{course}': ***")
        logging.info(cnt)
        output_path = os.path.join(args.output_dir, f"{course}.jsonl")
        with open(output_path, "w") as f:
            for q in all_questions[course]:
                f.write(json.dumps(q) + "\n")
