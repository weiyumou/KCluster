import argparse
import os

import torch
from transformers.utils import logging

from kcluster.core.prompts import CONCEPT_LENGTH_PENALTY, CONCEPT_NUM_BEAMS
from kcluster.engine.local import LargeLangModel
from kcluster.io.jsonl import load_questions
from kcluster.paths import dataset_name, default_result_dir, kc_dir, prepare_output_dir, save_args
from kcluster.tasks.cluster import build_res_df
from kcluster.tasks.concept import extract_concepts


def main(args):
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    # Resolve the per-dataset result folder (D10 layout)
    result_dir = args.output_dir or default_result_dir(args.run_dir)
    args.output_dir = result_dir = prepare_output_dir(result_dir)
    print(f"*** Writing results to {result_dir} ***")

    # Load an LLM
    llm = LargeLangModel(args.llm_path, add_bos=args.add_bos, trust_remote_code=True, dtype=torch.float16)

    # Read all questions
    questions = load_questions(args.data_path)

    # Extract concepts
    concepts = extract_concepts(llm, questions, args.batch_size, verbal=args.verbal,
                                do_sample=False, pad_to_multiple_of=args.pad_to_multiple_of,
                                num_beams=args.num_beams, length_penalty=args.length_penalty)

    # The concept table *is* the Concept KC model, so it goes straight into
    # kc/ under the standard name — no separate raw copy exists (D10).
    ds = dataset_name(args.data_path)
    res_df = build_res_df(questions, concepts)
    res_df.to_csv(os.path.join(prepare_output_dir(kc_dir(result_dir, "concept")), f"{ds}_concept-kc.csv"),
                  index=False)

    # build-kc and embed recover data_path from this breadcrumb
    save_args(result_dir, "concept", ds, vars(args))


def add_arguments(parser):
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--data_path", required=True, type=str, help="Path to a jsonl file of questions")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The result directory (default: --run_dir or a fresh timestamped folder)")
    parser.add_argument("--run_dir", default=None, type=str,
                        help="Result folder shared by every step of this run (env: KCLUSTER_RUN_DIR)")
    parser.add_argument("--verbal", action="store_true", help="Whether the concept should start with a verb")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
    parser.add_argument("--add_bos", action=argparse.BooleanOptionalAction, default=None,
                        help="Put the model's BOS token before every prompt that lacks one (default: the "
                             "tokenizer's own rule; a tokenizer that adds none, such as Gemma 4's instruction-tuned one, wants it)")
    parser.add_argument("--num_beams", type=int, default=CONCEPT_NUM_BEAMS,
                        help="Number of beams employed in beam search")
    parser.add_argument("--length_penalty", type=float, default=CONCEPT_LENGTH_PENALTY,
                        help="Length penalty for beam search")
    parser.add_argument("--pad_to_multiple_of", type=int, default=None, help="Pad to multiple of")
