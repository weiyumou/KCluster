import argparse
import glob
import json
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import set_seed
from transformers.utils import logging

from kcluster.engine.local import LargeLangModel
from kcluster.io.jsonl import dump_questions
from kcluster.paths import default_output_dir, prepare_output_dir
from kcluster.tasks.qgen.generate import generate_mcq_from_std, read_standards

PROMPT_SEP = "\n\n\n\n\n/br/"


def load_generation_configs(path: str) -> dict:
    """Load per-step generation configs from a TOML or JSON file.

    Replaces the legacy ``eval()``-based config reader: tables/objects named
    after the steps ("stem", "choice", "explanation") hold keyword arguments
    for ``complete_prompts``.
    """
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    import tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def main(args):
    set_seed(42)
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    # Create a folder to store results
    output_dir = getattr(args, "output_dir", None) or default_output_dir("qgen", getattr(args, "run_dir", None))
    args.output_dir = prepare_output_dir(output_dir)
    print(f"*** Writing results to {args.output_dir} ***")

    # Load an LLM
    llm = LargeLangModel(args.llm_path, trust_remote_code=True, torch_dtype=torch.float16)

    # Load configurations
    configs = dict()
    if c_path := getattr(args, "config_path", None):
        configs.update(load_generation_configs(c_path))
    configs.setdefault("stem", {}).update(guidance_scale=args.guidance_scale)

    # Load standards and generate MCQs
    for std_type in ("facts", "actions"):
        for fname in tqdm(
                sorted(glob.iglob(f"**/{std_type}/*.txt", recursive=True, root_dir=args.std_dir)),
                desc=f"Generating MCQs ({std_type})"):
            # Create the output directory
            output_dir = os.path.join(args.output_dir, os.path.splitext(fname)[0])
            os.makedirs(output_dir, exist_ok=False)

            # Read standards
            standards = read_standards(os.path.join(args.std_dir, fname), std_type)

            # Generate MCQs
            stds_per_batch = getattr(args, "stds_per_batch", len(standards))
            all_questions, all_prompts = generate_mcq_from_std(llm, standards, std_type, stds_per_batch,
                                                               args.qs_per_std, configs, args.num_choices)

            # Save MCQs
            dump_questions(all_questions, os.path.join(output_dir, "new-mcq.jsonl"))
            res_df = pd.DataFrame.from_records(q.flat_dict for q in all_questions)
            res_df.to_csv(os.path.join(output_dir, "new-mcq.csv"), index=False)

            # Save prompts
            with open(os.path.join(output_dir, "prompts.txt"), "w") as f:
                f.write(PROMPT_SEP.join(p for p in all_prompts))

    # Save arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)


def add_arguments(parser):
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--std_dir", required=True, type=str,
                        help="Path to a directory containing standards (facts/*.txt, actions/*.txt)")
    parser.add_argument("--config_path", default=argparse.SUPPRESS, type=str,
                        help="Path to a TOML/JSON file of generation configs")
    parser.add_argument("--num_choices", default=4, type=int, help="Number of choices in an MCQ")
    parser.add_argument("--qs_per_std", default=1, type=int, help="Number of questions to generate per standard")
    parser.add_argument("--guidance_scale", default=1.0, type=float, help="Guidance scale for CFG")
    parser.add_argument("--stds_per_batch", default=argparse.SUPPRESS, type=int, help="Number of standards in a batch")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str, help="Path to the output directory")
    parser.add_argument("--run_dir", default=argparse.SUPPRESS, type=str,
                        help="Shared run folder; each step writes to <run_dir>/<step> (env: KCLUSTER_RUN_DIR)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
