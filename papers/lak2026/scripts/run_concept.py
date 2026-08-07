"""Extract concepts for every validated question set under a run root.

Study orchestration over the library's concept task: one concept.csv per
generated-question directory, plus the args-concept.json breadcrumb that
build_kc.py reads.
"""

import argparse
import glob
import json
import os
import time

import torch
from tqdm import tqdm
from transformers import set_seed
from transformers.utils import logging

from kcluster.engine.local import LargeLangModel
from kcluster.io.jsonl import load_questions
from kcluster.tasks.cluster import build_res_df
from kcluster.tasks.concept import extract_concepts


def main(args):
    set_seed(42)
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    # Create a folder to store results
    if not hasattr(args, "output_dir"):
        args.output_dir = os.path.join("results", "kcluster", time.strftime("%Y%m%d-%H%M%S"))

    # Load an LLM
    llm = LargeLangModel(args.llm_path, trust_remote_code=True, torch_dtype=torch.float16)

    pat = "-gpt" if args.use_gpt_valid else ""
    for fname in tqdm(
            sorted(glob.iglob(f"**/new-mcq{pat}-valid.jsonl", recursive=True, root_dir=args.root_dir)),
            desc="Extracting concepts"):
        curr_dir = os.path.split(fname)[0]
        output_dir = os.path.join(args.output_dir, curr_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Read all questions
        questions = load_questions(os.path.join(args.root_dir, fname))

        # Extract concepts (the library task builds the identical prompt)
        concepts = extract_concepts(llm, questions, args.batch_size, pad_to_multiple_of=8, do_sample=False,
                                    num_beams=args.num_beams, length_penalty=args.length_penalty)

        # Save results
        res_df = build_res_df(questions, concepts)
        res_df.to_csv(os.path.join(output_dir, "concept.csv"), index=False)

    # Save arguments
    with open(os.path.join(args.output_dir, "args-concept.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--root_dir", required=True, type=str, help="Path to a root directory of generated questions")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams employed in beam search")
    parser.add_argument("--length_penalty", type=float, default=-0.1, help="Length penalty for beam search")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str, help="Path to the output directory")
    parser.add_argument("--use_gpt_valid", action="store_true", help="Whether to use GPT-valid questions")

    main(parser.parse_args())
