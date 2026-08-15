"""Dataset driver for the spacing study (Podsie; DataShop ds6527-era templates).

    python processing.py --raw_tx_path data/raw/<export>.txt --output_dir data/processed

Writes the questions of each of the 13 courses as its own ``<Course>.jsonl``,
because that is the unit KCluster runs on, and — in the same pass, so the two
cannot come to describe different step sets — the whole study's interaction log
as one untagged ``spacing-exp2_student-step-minimal.txt`` (the contract in
``kcluster.io.student_step``), beside the cleaned transaction file it was
reduced from. Those two are working files and land in ``--interim_dir``; the
question banks are the dataset's output and land in ``--output_dir``.

Step names are this driver's own per-type question ids rather than a native
DataShop value, and they are unique across all 13 courses (the id is derived
from the course name as well as the problem text), so ``ds-step-name`` alone
keys a student-step row to its question and one undivided file is unambiguous.
Tagging it needs the courses' KC models concatenated, one frame per model.
"""

import logging
import random

import pandas as pd

from kcluster.io.student_step import (
    MINIMAL_SUFFIX,
    MULTI_KC_SEP,
    check_coverage,
    save_student_step,
    validate_student_step,
)

#: The student-step file's stem: the dataset id, not any one course's.
DS = "spacing-exp2"

#: The cleaned transaction file, named for the export it was cleaned from —
#: change it with the export. Both interim artifacts are regenerable working
#: files: nothing downstream reads them, they are for inspection.
TX_FILE = "tx_2025-07-14_filtered.txt"

Q_TYPES = {"Multiple Choice": "mcq", "Short Answer": "sha", "True or False": "tof", "Fill in Blank": "fib"}

#: The expert KC models the export ships, as transaction columns. Each becomes a
#: question field (``standard``, ``q-group``) and a ``KC (...)`` column of the
#: student-step file, so they are collapsed once here, at the source of both.
EXPERT_KC_COLUMNS = ("CF (Topic Text)", "CF (Question Group)")

#: The one contested step whose disagreement is real rather than a defect:
#: ``tof-220`` is the true-false member of two question-group quartets, and
#: QG-954's other three steps are uncontested — so the majority rule would
#: quietly take the quartet's T/F question away from it. It carries both labels
#: instead, ``~~``-joined as DataShop writes a step with two KCs. Keyed by the
#: labels observed, so a re-numbered or re-exported dataset raises here rather
#: than silently tagging the wrong step.
MULTI_KC_STEPS = {"CF (Question Group)": {"tof-220": ("QG-954", "QG-955")}}

#: Transaction column -> minimal student-step column. ``Session Id`` is not part
#: of the file; it survives only long enough to split a student's repeats of one
#: question into one row per encounter (see :func:`build_student_step`).
SS_COLUMNS = {
    "Anon Student Id": "Anon Student Id",
    "Level (Course)": "Problem Hierarchy",
    "Problem Name": "Problem Name",
    "Step Name": "Step Name",
    "Outcome": "First Attempt",
    "Problem Start Time": "First Transaction Time",
    "CF (Topic Text)": "KC (Topic)",
    "CF (Question Group)": "KC (Question Group)",
    "Session Id": "Session Id",
}


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
    df["question"] = df["CF (Course Name)"] + df["CF (Full Problem Name)"] + df["CF (Answer Options)"].fillna("")  # temporary key

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

    # One expert KC label per step, before anything downstream reads them
    df = collapse_expert_kcs(df)

    # Format completion time
    df["CF (Completion Time)"] = pd.to_datetime(df["CF (Completion Time)"].astype(int), unit="ms")
    df["CF (Completion Time)"] = df["CF (Completion Time)"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df


def collapse_expert_kcs(df: pd.DataFrame, key: str = "CF (Question ID)") -> pd.DataFrame:
    """Make the export's expert KC models functions of the step.

    A few steps carry more than one topic or question group across their
    transactions: a single stray row in some, and in EPLA Chemistry a regrouping
    that moved several questions from one group to another partway through. AFM
    and DataShop alike take a KC model to map *steps* to KCs, and every model
    generated from the questions is one label per question, so a step with two
    labels is both an invalid Q-matrix and an unfair comparison — its rows are
    split between two KCs whose opportunity counts then each understate the
    practice the student actually had.

    The step's majority label wins, except for the steps named in
    :data:`MULTI_KC_STEPS`, which keep every label they carry. Both are logged
    rather than applied silently: an overwhelming majority is a defect being
    repaired, but a near-tie is a question about the data, and only the log
    distinguishes them.
    """
    for col in EXPERT_KC_COLUMNS:
        labels = df.groupby(key, sort=False)[col]
        majority = labels.agg(lambda values: values.value_counts().idxmax())
        contested = {step: df.loc[df[key].eq(step), col].value_counts()
                     for step in labels.nunique().pipe(lambda n: n[n > 1].index)}
        overrides = MULTI_KC_STEPS.get(col, {})

        for step, counts in contested.items():
            if step in overrides:
                continue
            winner, *losers = counts.index
            dropped = ", ".join(f"{value!r} ({counts[value]})" for value in losers)
            logging.info(f"{col}: step '{step}' collapsed to {winner!r} ({counts[winner]} rows), "
                         f"dropping {dropped}")

        df[col] = df[key].map(majority)
        for step, kept in overrides.items():
            rows = df[key].eq(step)
            if not rows.any():
                continue  # a subset of the study, or an export without this step
            observed = set(contested[step].index) if step in contested else set(df.loc[rows, col])
            assert observed == set(kept), (
                f"{col}: step '{step}' carries {sorted(observed)}, not the {sorted(kept)} it is pinned "
                "to. The pinning names a step of one export by its id; re-check it against this one.")
            df.loc[rows, col] = MULTI_KC_SEP.join(kept)
            logging.info(f"{col}: step '{step}' kept as multi-KC {MULTI_KC_SEP.join(kept)} "
                         f"({', '.join(f'{v}: {n}' for v, n in contested[step].items())} rows)")
    return df


def build_student_step(tx_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce cleaned transactions to a minimal student-step frame.

    One row per (course, step, student, session): that student's first
    transaction on the question in that sitting. ``First Attempt`` is the outcome
    of the first transaction of an encounter, per DataShop's step rollup, so the
    retries a student makes after seeing feedback collapse into the encounter
    that opened — their evidence of learning is the *next* encounter's first
    attempt, which is what an opportunity counts.

    The session is what separates encounters, because the platform logs it and a
    clock would have to guess: within one sitting a student may answer, read the
    feedback and retry, while the spaced re-presentations days later are the
    study's own treatment. Both look like "repeats on the same step". Measured on
    the 2025-07-14 export: of 77,629 repeat transactions, the 52% under ten
    minutes apart never cross a session, the 45% more than a day apart cross one
    99.8% of the time, and the 856 in between — exactly what a threshold would
    have to guess at — the session id simply settles.

    The two expert KC models the export ships (topic text and question group)
    ride along for the tagger to count opportunities for, and define the
    comparable universe — here every row, since both are populated throughout.
    """
    # Selected by label, not reindexed: the cleaned frame carries the export's
    # two "Condition Name"/"Condition Type" pairs, and duplicate labels defeat
    # reindex. Every column named here is unique in it.
    ss = tx_df[list(SS_COLUMNS)].rename(columns=SS_COLUMNS)

    # DataShop renders each Level column as "<level name> <value>", so a course
    # that is Level (Course) here is "Course <name>" in an export of this data.
    ss["Problem Hierarchy"] = "Course " + ss["Problem Hierarchy"]
    # The export's outcome vocabulary is CORRECT/INCORRECT; DataShop's is lowercase.
    ss["First Attempt"] = ss["First Attempt"].str.lower()
    # Constant precision and no offset, so sorting this column as text is
    # sorting it by time — the ordering opportunity counts are defined against.
    ss["First Transaction Time"] = pd.to_datetime(ss["First Transaction Time"].astype("int64"), unit="ms")
    ss["First Transaction Time"] = ss["First Transaction Time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # A stable sort so that transactions sharing a timestamp keep export order,
    # and the row head(1) keeps for an encounter is the same on every run.
    ss = ss.sort_values(["Anon Student Id", "First Transaction Time"], kind="stable").reset_index(drop=True)
    ss = ss.groupby(["Problem Hierarchy", "Step Name", "Anon Student Id", "Session Id"], sort=False).head(1)
    return ss.drop(columns="Session Id").reset_index(drop=True)


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
    import json
    import os
    import warnings
    from collections import Counter, defaultdict

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = argparse.ArgumentParser(description="Extract questions from a DataShop transaction file.")
    parser.add_argument("--raw_tx_path", type=str, required=True, help="Path to a raw DataShop transaction file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--interim_dir", type=str, default="data/interim",
                        help="Where to write the cleaned transaction and student-step files "
                             "(default: data/interim)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling choices in MCQs")
    args = parser.parse_args()

    rng_ = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.interim_dir, exist_ok=True)

    tx = clean_transactions(args.raw_tx_path)

    all_questions = defaultdict(list)
    for course, df_ in tx.groupby("Level (Course)", sort=False):
        for q_type_ in Q_TYPES:
            all_questions[course].extend(extract_questions_by_type(df_, q_type_, rng=rng_))

        # Display statistics and save questions to JSON files
        cnt = Counter(q["type"] for q in all_questions[course])
        logging.info(f"*** Extracted the following number of questions for course '{course}': ***")
        logging.info(cnt)
        # Hyphens, not underscores: the rest of the pipeline derives a stem the
        # same way (vertex-build-kc's data_name), and a KC file is read back as
        # "<ds>_<model>-kc.csv" — an underscore in <ds> would be parsed as the
        # start of the model name.
        stem = course.replace(" ", "-")
        output_path = os.path.join(args.output_dir, f"{stem}.jsonl")
        with open(output_path, "w") as f:
            for q in all_questions[course]:
                f.write(json.dumps(q) + "\n")

    # One student-step file for the whole study, checked against every course's
    # questions at once: each row must resolve to exactly one of them. It is
    # untagged — expert KC models only, no generated ones — which is what the
    # tagger takes as input, so it lives with the other regenerable working files.
    ss = build_student_step(tx)
    validate_student_step(ss)
    questions = [q for qs in all_questions.values() for q in qs]
    uncovered = check_coverage(questions, ss)
    assert not uncovered, f"{len(uncovered)} question(s) have no student-step rows: {uncovered[:5]}"
    ss_path = os.path.join(args.interim_dir, f"{DS}{MINIMAL_SUFFIX}")
    save_student_step(ss, ss_path)
    logging.info(f"*** Saved {len(ss)} student-step rows for {len(questions)} questions "
                 f"({ss['Anon Student Id'].nunique()} students, "
                 f"{ss['First Attempt'].eq('correct').mean():.1%} correct) to '{ss_path}' ***")

    tx_path = os.path.join(args.interim_dir, TX_FILE)
    tx.to_csv(tx_path, sep="\t", index=False)
    logging.info(f"*** Saved {len(tx)} cleaned transactions to '{tx_path}' ***")
