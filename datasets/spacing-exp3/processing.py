"""Extract the Spacing-Exp3 multiple-choice bank from a DataShop transaction file.

Sibling driver to ``datasets/spacing-exp2/processing.py``. The two exports share
neither template nor question representation, so the extraction logic is not
shared: exp3 (ds6824) ships no ``CF (Answer Options)``, no
``CF (Correct Answer Options)``, no course/standard/question-group columns, and
its ``CF (Question Id)`` is reused across unrelated stems. Answer options are
therefore *reconstructed* from the ``Input`` values students actually selected,
and the answer key comes from ``CF (Exemplar Answer)``.

Scope is the multiple-choice pool only. The export's 1,511 short-answer items
were each answered exactly once (one generated item per student per posttest),
so they carry no response distribution for KCluster to evaluate against.
"""

import logging
import random

import pandas as pd

Q_TYPES = {"Multiple Choice": "mcq"}
PHASES = ["pretest1", "pretest2", "learning", "posttest1", "posttest2"]

# Recorded when a student answered outside the option list (drawing canvas).
NON_ANSWERS = {"Responded on canvas"}

# DataShop truncates the exported problem text at this width.
TRUNCATION_WIDTH = 255


def clean_transactions(raw_tx_path: str) -> pd.DataFrame:
    """This function cleans a raw transaction file."""
    key_cols = ["Anon Student Id", "Session Id", "Time", "Level (Phase)", "Problem Name",
                "Problem Start Time", "Step Name", "Outcome", "Input",
                "Condition Name", "Condition Type", "Condition Name.1", "Condition Type.1",
                "Class"]
    cf_cols = ["CF (Full Problem Name)", "CF (Question Type)", "CF (Exemplar Answer)",
               "CF (Enrollment Id)", "CF (Student Profile Id)", "CF (Question Attempt Id)",
               "CF (Response Time)", "CF (Completion Time)", "CF (Timed Out)"]

    df = pd.read_csv(raw_tx_path, sep="\t", dtype=str)
    df = df[key_cols + cf_cols].rename(
        columns={"Condition Name": "CF (Spacing Condition)",
                 "Condition Type": "CF (Spacing Condition Type)",
                 "Condition Name.1": "CF (Algorithm Condition)",
                 "Condition Type.1": "CF (Algorithm Condition Type)"})

    df = df[df["Level (Phase)"].isin(PHASES)].reset_index(drop=True)
    df = df.apply(lambda x: x.str.replace("\n", " "), axis=0)
    df = df.apply(lambda x: x.str.replace('"', "'"), axis=0)
    df = df.apply(lambda x: x.str.strip(), axis=0)

    return df


def select_mcq_items(tx_df: pd.DataFrame, min_students: int = 20,
                     variant_policy: str = "dominant") -> pd.DataFrame:
    """This function selects the study's multiple-choice items and resolves stem variants.

    Two kinds of noise are removed. Filler items -- one-off questions from
    unrelated assignments that a handful of students happened to answer -- are
    dropped by requiring at least ``min_students`` respondents. Stems that carry
    more than one ``CF (Exemplar Answer)`` are resolved to a single variant,
    either the one with the most responses (``dominant``, the default, which
    preserves the most student data) or the one answered most recently
    (``latest``). Options are then read only from rows sharing the kept
    exemplar, so a rewritten item never mixes option sets across versions.

    Adds an ``answer`` column holding the resolved exemplar for every row.
    """
    assert variant_policy in ("dominant", "latest"), f"Unknown variant policy: {variant_policy}"

    df = tx_df[tx_df["CF (Question Type)"].eq("Multiple Choice")].copy()

    respondents = df.groupby("CF (Full Problem Name)")["Anon Student Id"].nunique()
    filler = respondents[respondents < min_students].index
    if len(filler):
        dropped = df["CF (Full Problem Name)"].isin(filler)
        logging.info(f"Dropping {len(filler)} filler stems ({dropped.sum()} rows) "
                     f"answered by fewer than {min_students} students")
        df = df[~dropped]

    ranking = df.groupby(["CF (Full Problem Name)", "CF (Exemplar Answer)"]).agg(
        responses=("Anon Student Id", "size"), last_seen=("Time", "max"))
    sort_by = ["responses", "last_seen"] if variant_policy == "dominant" else ["last_seen", "responses"]
    ranking = ranking.sort_values(sort_by, ascending=False)
    kept = ranking.reset_index().drop_duplicates(subset=["CF (Full Problem Name)"])

    variants = ranking.reset_index().groupby("CF (Full Problem Name)").size()
    if (variants > 1).any():
        logging.info(f"Resolving {(variants > 1).sum()} stems with multiple exemplar answers "
                     f"by '{variant_policy}' policy")

    answers = kept.set_index("CF (Full Problem Name)")["CF (Exemplar Answer)"].rename("answer")
    df = df.join(answers, on="CF (Full Problem Name)")
    return df[df["CF (Exemplar Answer)"].eq(df["answer"])].reset_index(drop=True)


def assign_question_ids(mcq_df: pd.DataFrame) -> pd.DataFrame:
    """This function gives every item a stable ID and makes Step Name unique to it.

    The export's own ``CF (Question Id)`` is reused across unrelated stems, and its
    ``Step Name`` is the constant "Submit Answer", so neither identifies an item. IDs
    are assigned in order of first appearance, and Step Name is overwritten with the
    ID (as in exp2) so that the transaction file and the question file join on
    problem/step the way a DataShop KC model expects.
    """
    df = mcq_df.copy()
    order, _ = pd.factorize(df["CF (Full Problem Name)"])
    df["CF (Question ID)"] = Q_TYPES["Multiple Choice"] + "-" + pd.Series(order, index=df.index).astype(str)
    df["Step Name"] = df["CF (Question ID)"]
    return df


def extract_questions(mcq_df: pd.DataFrame, rng=None) -> list[dict]:
    """This function extracts a question list from the selected multiple-choice transactions."""
    rng = rng or random.Random()

    questions = []
    for stem, group in mcq_df.groupby("CF (Full Problem Name)", sort=False):
        answer = group["answer"].iloc[0]
        options = set(group["Input"].dropna()) - NON_ANSWERS
        options.add(answer)  # an option nobody picked is still an option

        choices = rng.sample(sorted(options), k=len(options))
        choices = [{"label": chr(ord("a") + idx), "text": text} for idx, text in enumerate(choices)]
        answer_key = next(choice["label"] for choice in choices if choice["text"] == answer)

        questions.append({
            "id": group["CF (Question ID)"].iloc[0],
            "type": "Multiple Choice",
            "question": {"stem": stem, "choices": choices},
            "answerKey": answer_key,
            "ds-problem-name": group["Problem Name"].iloc[0],
            "ds-step-name": group["Step Name"].iloc[0],
        })

    return questions


def save_transactions(mcq_df: pd.DataFrame, path: str) -> None:
    """This function writes the retained transactions back out in DataShop's tab-delimited format.

    Only rows belonging to the extracted items survive, and the two condition pairs get
    their original duplicated headers back, so the file stays a transaction file rather
    than an analysis frame. Timed-out attempts (blank ``Input``) are kept: a non-response
    is part of the response record.
    """
    df = mcq_df.drop(columns=["answer"])
    df = df.rename(columns={"CF (Spacing Condition)": "Condition Name",
                            "CF (Spacing Condition Type)": "Condition Type",
                            "CF (Algorithm Condition)": "Condition Name",
                            "CF (Algorithm Condition Type)": "Condition Type"})
    df.to_csv(path, sep="\t", index=False)


def report_caveats(questions: list[dict]) -> None:
    """This function logs the items that need a human eye before they are used as text."""
    images = [q["id"] for q in questions if "![" in q["question"]["stem"]]
    if images:
        logging.warning(f"{len(images)} stems embed image references, which carry no text content: "
                        f"{', '.join(images)}")

    truncated = [q["id"] for q in questions if len(q["question"]["stem"]) >= TRUNCATION_WIDTH]
    if truncated:
        logging.warning(f"{len(truncated)} stems hit the {TRUNCATION_WIDTH}-char DataShop export limit "
                        f"and are cut off: {', '.join(truncated)}")

    sizes = pd.Series([len(q["question"]["choices"]) for q in questions]).value_counts()
    logging.info(f"Choices per question: {sizes.sort_index().to_dict()}")


if __name__ == "__main__":
    import argparse
    import json
    import os
    import warnings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = argparse.ArgumentParser(description="Extract questions from a DataShop transaction file.")
    parser.add_argument("--raw_tx_path", type=str, required=True, help="Path to a raw DataShop transaction file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--min_students", type=int, default=20,
                        help="Minimum distinct respondents for a stem to count as a study item")
    parser.add_argument("--variant_policy", type=str, default="dominant", choices=["dominant", "latest"],
                        help="How to resolve stems carrying more than one exemplar answer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling choices in MCQs")
    args = parser.parse_args()

    rng_ = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tx = clean_transactions(args.raw_tx_path)
    mcq = select_mcq_items(tx, min_students=args.min_students, variant_policy=args.variant_policy)
    mcq = assign_question_ids(mcq)
    all_questions = extract_questions(mcq, rng=rng_)

    logging.info(f"*** Extracted {len(all_questions)} multiple-choice questions "
                 f"from {mcq['Anon Student Id'].nunique()} students and {len(mcq)} responses ***")
    report_caveats(all_questions)

    output_path = os.path.join(args.output_dir, "mcq.jsonl")
    with open(output_path, "w") as f:
        for q in all_questions:
            f.write(json.dumps(q) + "\n")
    logging.info(f"Wrote {output_path}")

    tx_path = os.path.join(args.output_dir, "mcq-tx.txt")
    save_transactions(mcq, tx_path)
    logging.info(f"Wrote {tx_path} ({len(mcq)} of {len(tx)} transactions, "
                 f"{mcq['Input'].isna().sum()} of them timed out)")
