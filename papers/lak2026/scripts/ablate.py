"""Threshold ablation for the answer-confidence validator (LAK 2026, RQ on
prob_thd). Uses a cheaper single-ordering confidence check (no permutation
averaging) after shuffling choices once, sweeping prob_thd over [0.2, 1.0).
"""

import argparse
import copy
import itertools
import json
import math
import os
import string
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import set_seed
from transformers.utils import logging

from kcluster.core.question import Question
from kcluster.engine.local import LargeLangModel
from kcluster.io.jsonl import dump_questions, load_questions
from kcluster.tasks.qgen.validate import shuffle_choices


@torch.inference_mode()
def validate_mcq(llm: LargeLangModel, questions: dict[str, list[Question]],
                 batch_size: int = 16, prob_thd: float = 0.9,
                 num_choices: int = 4, shuffle: bool = True) -> tuple[dict[str, list[Question]], list[Question]]:
    choices = [Question.SPACE + chc for chc in string.ascii_lowercase[:num_choices + 1]]
    chc_ids = list(itertools.chain.from_iterable(llm.tokenizer(choices)["input_ids"]))
    undesired = {"all of the above", "none of the above", "yes", "no", "true", "false"}

    syntactic_valid_questions = []
    # Filter questions
    for lo in list(questions):
        valid = []
        # First check if each question is complete
        for q in questions[lo]:
            q_choices = {chc["text"].lower().strip() for chc in q["question"]["choices"]}
            if any([
                # if the question has a trivial stem
                len(q.stem) < 10,
                # if there are invalid or duplicate choices
                len(q_choices) < num_choices,
                # if any choice is trivial
                any(len(chc) < 5 for chc in q_choices),
                # if the question has a trivial explanation
                len(q["explanation"]) < 10,
                # if any undesired choice is present
                len(q_choices & undesired) > 0,
                # if any choice starts with "both" or "neither"
                any(chc.startswith("both") or chc.startswith("neither") for chc in q_choices),
            ]):
                continue
            # only complete questions are kept
            valid.append(q)

        # Retain only valid questions
        questions[lo], valid = valid, []
        if shuffle:  # Shuffle the choices to avoid positional bias
            shuffle_choices(questions[lo])
        syntactic_valid_questions.extend(copy.deepcopy(questions[lo]))

        # Then check if any choice has high enough probability
        for batch in itertools.batched(questions[lo], batch_size):
            prompts = []
            for q in batch:
                q["question"]["choices"].append({"label": choices[-1].strip(), "text": "None of the above"})
                prompts.append(q.prompt())
                q["question"]["choices"].pop()

            log_probs = torch.log_softmax(llm.next_logits(prompts)[:, chc_ids], dim=-1)
            values, indices = torch.max(log_probs, dim=-1)
            is_valid = torch.ge(values, math.log(prob_thd)) & torch.ne(indices, num_choices)
            valid_inds = torch.nonzero(is_valid, as_tuple=True)[0].tolist()
            valid.extend([batch[idx] for idx in valid_inds])

        # Retain only valid questions
        if valid:
            questions[lo] = valid
        else:
            del questions[lo]

    return questions, syntactic_valid_questions


def main(args):
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    output_dir = os.path.join(args.root_dir, "threshold-ablation", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=False)

    # Load an LLM
    llm = LargeLangModel(args.llm_path, trust_remote_code=True, torch_dtype=torch.float16)

    # Read all questions
    all_questions = load_questions(os.path.join(args.root_dir, "new-mcq-raw.jsonl"))
    los = set(q["lo"] for q in all_questions)
    print(f"*** Loaded {len(all_questions)} questions with {len(los)} LOs ***")

    # Group questions by LO and remove duplicates
    all_ids = set()
    questions_by_lo = defaultdict(list)
    for q in all_questions:
        if q["id"] not in all_ids:
            questions_by_lo[q["lo"]].append(q)
            all_ids.add(q["id"])
    num_los, num_questions = len(questions_by_lo), sum(len(questions_by_lo[lo]) for lo in questions_by_lo)
    print(f"*** After removing duplicates, {num_questions} questions with {num_los} LOs remain ***")

    for seed in range(42, 42 + args.n_rounds):
        save_dir = os.path.join(output_dir, f"seed-{seed}")
        os.makedirs(save_dir, exist_ok=True)
        stats = []
        syn_valid_qs = []
        for prob_thd in np.arange(0.2, 1.0, 0.05):
            set_seed(seed)
            prob_thd = float(prob_thd)
            print(f"\n\n=== Validating questions with prob_thd = {prob_thd:.2f} ===", flush=True)
            questions, syn_valid_qs = validate_mcq(llm, copy.deepcopy(questions_by_lo), args.batch_size, prob_thd)
            if not questions:
                print(f"No valid questions found for prob_thd = {prob_thd}. Skipping...")
                continue
            num_los, num_questions = len(questions), sum(len(questions[lo]) for lo in questions)
            print(f"*** Syntactically valid questions so far: {len(syn_valid_qs)} ***")
            print(f"*** After validation, {num_questions} questions with {num_los} LOs remain ***")
            stats.append({"prob_thd": prob_thd, "num_los": num_los, "num_questions": num_questions})

            # Flatten questions
            questions = [q for lo in questions for q in questions[lo]]

            # Save questions
            fname = f"new-mcq-valid-{prob_thd:.2f}".replace(".", "_")
            dump_questions(questions, os.path.join(save_dir, f"{fname}.jsonl"))

        # Save syntactically valid questions
        dump_questions(syn_valid_qs, os.path.join(save_dir, "new-mcq-syn-valid.jsonl"))

        # Save stats to jsonl
        with open(os.path.join(output_dir, f"stats-seed={seed}.jsonl"), "w") as f:
            for r in stats:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--root_dir", required=True, type=str, help="Path to a root directory of generated questions")
    parser.add_argument("--batch_size", default=16, type=int, help="Number of questions to validate in a batch")
    parser.add_argument("--n_rounds", default=1, type=int, help="Number of validation rounds")

    main(parser.parse_args())
