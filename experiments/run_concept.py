import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers.utils import logging

from core.model import LargeLangModel, batched
from core.question import Question


@torch.inference_mode()
def extract_concepts(llm: LargeLangModel, questions: list[Question],
                     batch_size: int, verbal: bool = False, **kwargs) -> list[str]:
    """Extracts the key concept for a list of Questions"""
    SPACE = Question.SPACE

    # determine whether the generated concept should begin with a verb
    if verbal:
        trailer = "whether the student can"  # +verbal phrase
    else:
        trailer = "whether the student understands the concept of"  # +noun phrase

    all_concepts = []
    for batch in tqdm(batched(questions, batch_size), desc="Extracting concepts"):
        prompts = []
        for q in batch:
            q_type = q.q_type.lower().replace(SPACE, "-")
            prompt = (
                f"{q.prompt()}{SPACE}{q.answer}\n\n"
                f"Remark:\nThe above exercise is a {q_type} question that tests {trailer}"
            )
            prompts.append(prompt)

        concepts = llm.complete_prompts(prompts, stop_tokens=[".", ","], **kwargs)
        all_concepts.extend([c.strip().rstrip(".,") for c in concepts])

    return all_concepts


@torch.inference_mode()
def extract_question_embeds(llm: LargeLangModel, questions: list[Question], batch_size: int, **kwargs) -> torch.Tensor:
    all_embeddings = []
    for batch in tqdm(batched(questions, batch_size), desc="Extracting question embeddings"):
        contexts = [f"{q.header(2)}\n" for q in batch]
        texts = [str(q) for q in batch]
        all_embeddings.append(llm.encode(texts, contexts, **kwargs))

    return torch.cat(all_embeddings, dim=0)


def build_res_df(questions: list[Question], concepts: list[str]) -> pd.DataFrame:
    q_dicts = []
    for q, c in zip(questions, concepts):
        q_dict = q.flat_dict
        q_dict.pop("images", None)
        q_dict["KC"] = c
        q_dicts.append(q_dict)

    return pd.DataFrame.from_records(q_dicts)


def main(args):
    # Create a folder to store results
    args.output_dir = getattr(args, "output_dir", os.path.join("results", "concept", time.asctime()))
    args.output_dir = args.output_dir.replace(' ', '_').replace(':', '-')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load an LLM
    llm = LargeLangModel(args.llm_path, trust_remote_code=True,
                         torch_dtype=torch.float16, attn_implementation="flash_attention_2")

    # Read all questions
    with open(args.data_path, "r") as f:
        questions = [Question(eval(line)) for line in f]

    # Extract concepts
    concepts = extract_concepts(llm, questions, args.batch_size, verbal=args.verbal,
                                do_sample=False, pad_to_multiple_of=8,
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
    with open(os.path.join(args.output_dir, f"args-{fname}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--data_path", required=True, type=str, help="Path to a jsonl file of questions")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str, help="Path to the output directory")
    parser.add_argument("--verbal", action="store_true", help="Whether the concept should start with a verb")
    parser.add_argument("--sent_path", type=str, default=argparse.SUPPRESS, help="Path to a SentenceTransformer")
    parser.add_argument("--q_embeds", action="store_true", help="Whether to compute question embeddings")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams employed in beam search")
    parser.add_argument("--length_penalty", type=float, default=-0.1, help="Length penalty for beam search")

    cl_args = parser.parse_args()
    main(cl_args)
