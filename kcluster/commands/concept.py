import argparse
import json
import os

import torch
from transformers.utils import logging

from kcluster.engine.local import LargeLangModel
from kcluster.io.jsonl import load_questions
from kcluster.paths import default_result_dir, kc_dir, prepare_output_dir
from kcluster.tasks.cluster import build_res_df
from kcluster.tasks.concept import extract_concepts


def main(args):
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    # Resolve the per-dataset result folder (D10 layout)
    result_dir = getattr(args, "output_dir", None) or default_result_dir(getattr(args, "run_dir", None))
    args.output_dir = result_dir = prepare_output_dir(result_dir)
    print(f"*** Writing results to {result_dir} ***")

    # Load an LLM
    llm = LargeLangModel(args.llm_path, trust_remote_code=True, torch_dtype=torch.float16)

    # Read all questions
    questions = load_questions(args.data_path)

    # Extract concepts
    concepts = extract_concepts(llm, questions, args.batch_size, verbal=args.verbal,
                                do_sample=False, pad_to_multiple_of=args.pad_to_multiple_of,
                                num_beams=args.num_beams, length_penalty=args.length_penalty)

    # The concept table *is* the Concept KC model, so it goes straight into
    # kc/ under the standard name — no separate raw copy exists (D10).
    ds = os.path.splitext(os.path.basename(args.data_path))[0].replace(" ", "-")
    res_df = build_res_df(questions, concepts)
    res_df.to_csv(os.path.join(prepare_output_dir(kc_dir(result_dir, "concept")), f"{ds}_concept-kc.csv"),
                  index=False)

    # Save arguments at the result root; build-kc and embed recover data_path from here
    with open(os.path.join(result_dir, f"args-concept-{ds}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def add_arguments(parser):
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--data_path", required=True, type=str, help="Path to a jsonl file of questions")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str,
                        help="The result directory (default: --run_dir or a fresh timestamped folder)")
    parser.add_argument("--run_dir", default=argparse.SUPPRESS, type=str,
                        help="Result folder shared by every step of this run (env: KCLUSTER_RUN_DIR)")
    parser.add_argument("--verbal", action="store_true", help="Whether the concept should start with a verb")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams employed in beam search")
    parser.add_argument("--length_penalty", type=float, default=-0.1, help="Length penalty for beam search")
    parser.add_argument("--pad_to_multiple_of", type=int, default=None, help="Pad to multiple of")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
