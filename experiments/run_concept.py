import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers.utils import logging

from core.model import LargeLangModel, batched
from core.question import Question


@torch.inference_mode()
def extract_concepts(llm: LargeLangModel, questions: list[Question], batch_size: int, **kwargs) -> list[str]:
    """Extracts the key concept for a list of Questions"""
    SPACE = Question.SPACE
    all_concepts = []
    for batch in batched(questions, batch_size):
        prompts = []
        for q in batch:
            q_type = q.q_type.lower().replace(SPACE, "-")
            prompt = (
                f"{q.prompt()}{SPACE}{q.answer}\n\n"
                f"Remark:\nThe above exercise is a {q_type} question{SPACE}"
                f"that tests whether the student understands the concept of"
            )
            prompts.append(prompt)

        concepts = llm.complete_prompts(prompts, stop_tokens=[".", ","], **kwargs)
        all_concepts.extend([c.lstrip().rstrip(".,") for c in concepts])

    return all_concepts


def build_res_df(questions: list[Question], concepts: list[str]) -> pd.DataFrame:
    q_dicts = []
    for q, c in zip(questions, concepts):
        q_dict = q.flat_dict
        q_dict["concept"] = c
        q_dicts.append(q_dict)

    return pd.DataFrame.from_records(q_dicts)


def main(args):
    # Create a folder to store results
    if not hasattr(args, "output_dir"):
        args.output_dir = os.path.join("results", "concept", time.asctime())
    args.output_dir = args.output_dir.replace(' ', '_').replace(':', '-')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load an LLM
    llm = LargeLangModel(args.llm_path, trust_remote_code=True,
                         torch_dtype=torch.float16, attn_implementation="flash_attention_2")

    # Read all questions
    with open(args.data_path, "r") as f:
        questions = [Question(eval(line)) for line in f]

    # Extract concepts
    concepts = extract_concepts(llm, questions, args.batch_size, pad_to_multiple_of=8, do_sample=False,
                                num_beams=args.num_beams, length_penalty=args.length_penalty)

    # Compute (concept) similarity matrix
    if hasattr(args, "sent_path"):
        model = SentenceTransformer(args.sent_path, local_files_only=True)
    else:
        model = llm

    with torch.inference_mode():
        embeddings = model.encode(concepts)
        sim_mtx = cos_sim(embeddings, embeddings).cpu().numpy()

    # Save results
    fname = os.path.splitext(os.path.basename(args.data_path))[0]
    res_df = build_res_df(questions, concepts)
    res_df.to_csv(os.path.join(args.output_dir, f"{fname}-concept.csv"), index=False)
    np.save(os.path.join(args.output_dir, f"{fname}-sim_mtx.npy"), sim_mtx)

    # Save arguments
    with open(os.path.join(args.output_dir, f"args-{fname}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--data_path", required=True, type=str, help="Path to a jsonl file of questions")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str, help="Path to the output directory")
    parser.add_argument("--sent_path", type=str, default=argparse.SUPPRESS, help="Path to a SentenceTransformer")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams employed in beam search")
    parser.add_argument("--length_penalty", type=float, default=-0.1, help="Length penalty for beam search")

    cl_args = parser.parse_args()
    main(cl_args)
