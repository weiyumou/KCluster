import argparse
import json
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers.utils import logging

from kcluster.engine.local import LargeLangModel
from kcluster.io.jsonl import load_questions
from kcluster.paths import default_output_dir, prepare_output_dir
from kcluster.tasks.concept import build_res_df, extract_concepts, extract_question_embeds


def main(args):
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    # Create a folder to store results
    output_dir = getattr(args, "output_dir", None) or default_output_dir("concept")
    args.output_dir = prepare_output_dir(output_dir)
    print(f"*** Writing results to {args.output_dir} ***")

    # Load an LLM
    llm = LargeLangModel(args.llm_path, trust_remote_code=True, torch_dtype=torch.float16)

    # Read all questions
    questions = load_questions(args.data_path)

    # Extract concepts
    concepts = extract_concepts(llm, questions, args.batch_size, verbal=args.verbal,
                                do_sample=False, pad_to_multiple_of=args.pad_to_multiple_of,
                                num_beams=args.num_beams, length_penalty=args.length_penalty)

    # Save results
    fname = os.path.splitext(os.path.basename(args.data_path))[0]
    res_df = build_res_df(questions, concepts)
    res_df.to_csv(os.path.join(args.output_dir, f"{fname}-concept.csv"), index=False)

    # Compute concept embeddings if path to SentenceTransformer is provided
    if sent_path := getattr(args, "sent_path", None):
        model = SentenceTransformer(sent_path, local_files_only=True)
        with torch.inference_mode():
            embeddings = model.encode(concepts)
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
        # Save results
        np.save(os.path.join(args.output_dir, f"{fname}-concept-embeds.npy"), embeddings)

    # Compute question embeddings
    if args.q_embeds:
        embeddings = extract_question_embeds(llm, questions, args.batch_size).cpu().numpy()
        np.save(os.path.join(args.output_dir, f"{fname}-question-embeds.npy"), embeddings)

    # Save arguments
    with open(os.path.join(args.output_dir, f"args-concept-{fname}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def add_arguments(parser):
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--data_path", required=True, type=str, help="Path to a jsonl file of questions")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str, help="Path to the output directory")
    parser.add_argument("--verbal", action="store_true", help="Whether the concept should start with a verb")
    parser.add_argument("--sent_path", type=str, default=argparse.SUPPRESS, help="Path to a SentenceTransformer")
    parser.add_argument("--q_embeds", action="store_true", help="Whether to compute question embeddings")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams employed in beam search")
    parser.add_argument("--length_penalty", type=float, default=-0.1, help="Length penalty for beam search")
    parser.add_argument("--pad_to_multiple_of", type=int, default=None, help="Pad to multiple of")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
