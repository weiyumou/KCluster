"""Assemble the expert-study Qualtrics survey.

``mix_n_match`` implements the study manipulation: per LO, half the
sampled questions keep their true LO and half get a randomly chosen false
one (the ``false_lo`` field the survey writer and judges read). The main
mixes questions from several validated sets, shuffles, and renders the
survey via the library's Qualtrics writer. After importing the survey and
exporting its .qsf, run ``kcluster.output.qualtrics.force_response`` on it.
"""

import argparse
import os
import random
import time
from collections import defaultdict

from kcluster.core.question import Question
from kcluster.io.jsonl import dump_questions, load_questions
from kcluster.output.qualtrics import write_txt


def mix_n_match(questions: list[Question], num_los: int, group_sz: int, seed: int = 0) -> list[Question]:
    # Fix the RNG
    rng = random.Random(seed)

    # Group questions by LOs
    questions_by_lo = defaultdict(list)
    for q in questions:
        questions_by_lo[q["lo"]].append(q)

    # Randomly select enough LOs so that each LO has enough questions
    while all_los := set(rng.sample(list(questions_by_lo), k=num_los)):
        if all(len(questions_by_lo[lo]) >= group_sz * 2 for lo in all_los):
            break

    new_questions = []
    for lo in all_los:
        sel_qs = rng.sample(questions_by_lo[lo][:group_sz * 2], k=group_sz * 2)
        for q in sel_qs[group_sz:]:  # some questions get a false LO
            q["false_lo"] = rng.choice(list(all_los - {lo}))
            assert q["false_lo"] != q["lo"]
        new_questions.extend(sel_qs)

    assert len(new_questions) == num_los * group_sz * 2
    rng.shuffle(new_questions)
    return new_questions


def main(args):
    # Create a folder to store results
    if not hasattr(args, "output_dir"):
        args.output_dir = os.path.join("results", "qualtrics", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.output_dir, exist_ok=False)

    # Select survey questions for each data path
    survey_questions = []
    for seed, data_path in enumerate(args.data_paths, 10):
        questions = load_questions(data_path)
        survey_questions.extend(mix_n_match(questions, args.num_los, args.group_sz, seed))

    # Randomize all survey questions
    random.Random(42).shuffle(survey_questions)

    # Create a Qualtrics survey in TXT format
    write_txt(survey_questions, args.output_dir)

    # Save all survey questions
    dump_questions(survey_questions, os.path.join(args.output_dir, "survey_questions.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_paths", nargs="+", type=str, help="Paths to a JSONL file of questions")
    parser.add_argument("--num_los", required=True, type=int, help="Number of LOs to include in the survey")
    parser.add_argument("--group_sz", required=True, type=int, help="Size of each experimental group")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str, help="Path to the output directory")

    main(parser.parse_args())
