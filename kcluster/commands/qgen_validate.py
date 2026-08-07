import argparse
import glob
import os
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import set_seed
from transformers.utils import logging

from kcluster.commands.qgen_generate import PROMPT_SEP
from kcluster.engine.local import LargeLangModel
from kcluster.io.jsonl import dump_questions, load_questions
from kcluster.tasks.qgen.validate import sort_questions, validate_mcq


def main(args):
    set_seed(42)
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    # Load an LLM
    llm = LargeLangModel(args.llm_path, trust_remote_code=True, torch_dtype=torch.float16)

    # Iterate through all generated questions
    for fname in tqdm(
            sorted(glob.iglob("**/new-mcq.jsonl", recursive=True, root_dir=args.root_dir)), desc="Validating MCQs"):
        output_dir = os.path.join(args.root_dir, os.path.split(fname)[0])

        # Read all questions
        all_questions = load_questions(os.path.join(args.root_dir, fname))
        ids_to_inds = {q["id"]: idx for idx, q in enumerate(all_questions)}

        # Group questions by LO and remove duplicates
        all_ids = set()
        questions = defaultdict(list)
        for q in all_questions:
            if q["id"] not in all_ids:
                questions[q["lo"]].append(q)
                all_ids.add(q["id"])

        questions = validate_mcq(llm, questions, args.batch_size, args.prob_thd, args.num_choices)
        if not questions:
            print(f"No valid questions found for {fname}. Skipping...")
            continue

        # Read all prompts
        with open(os.path.join(output_dir, "prompts.txt"), "r") as f:
            all_prompts = [p.strip() for p in f.read().split("/br/")]
        prompts = [all_prompts[ids_to_inds[q["id"]]] for lo in questions for q in questions[lo]]

        # Sort questions by their perplexity
        questions, prompts = sort_questions(llm, questions, prompts, args.batch_size)

        # Flatten questions
        questions = [q for lo in questions for q in questions[lo]]

        # Save questions
        dump_questions(questions, os.path.join(output_dir, "new-mcq-valid.jsonl"))
        res_df = pd.DataFrame.from_records(q.flat_dict for q in questions)
        res_df.to_csv(os.path.join(output_dir, "new-mcq-valid.csv"), index=False)

        # Save prompts
        with open(os.path.join(output_dir, "prompts-valid.txt"), "w") as f:
            f.write(PROMPT_SEP.join(p for p in prompts))


def add_arguments(parser):
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--root_dir", required=True, type=str, help="Path to a root directory of generated questions")
    parser.add_argument("--num_choices", default=4, type=int, help="Number of choices in an MCQ")
    parser.add_argument("--batch_size", default=16, type=int, help="Number of questions to validate in a batch")
    parser.add_argument("--prob_thd", default=0.9, type=float,
                        help="The minimal probability a choice must have for the question to be valid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
