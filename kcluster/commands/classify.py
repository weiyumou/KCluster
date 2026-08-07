import argparse
import json
import os
from functools import partial

import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers.utils import logging

from kcluster.engine.local import CustomWriter, LargeLangModel, LogProbScorer, collate_pair
from kcluster.io.jsonl import load_questions
from kcluster.paths import default_output_dir, prepare_output_dir
from kcluster.tasks.classify import QuestionLO, classify_from_pmi


def main(args):
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    # Create a folder to store results
    output_dir = getattr(args, "output_dir", None) or default_output_dir("classify")
    args.output_dir = prepare_output_dir(output_dir)
    print(f"*** Writing results to {args.output_dir} ***")

    # Load tokenizer
    tokenizer = LargeLangModel.load_tokenizer(args.llm_path)

    # Load model
    model = LogProbScorer(args.llm_path, trust_remote_code=True, torch_dtype=torch.float16)

    # Read all questions and collect their LOs
    all_questions = load_questions(args.data_path)
    all_los = list(set(q["lo"] for q in all_questions))

    # Compute PMI for questions and LOs
    ds = QuestionLO(all_questions, all_los, args.lo_type)
    dl = DataLoader(ds, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=args.num_workers,
                    collate_fn=partial(collate_pair, tokenizer=tokenizer, pad_to_multiple_of=args.pad_to_multiple_of))

    pred_writer = CustomWriter(output_dir=args.output_dir, write_interval="epoch")
    trainer = L.Trainer(accelerator="gpu", devices=-1, callbacks=[pred_writer], logger=False)
    trainer.predict(model, dataloaders=dl, return_predictions=False)

    fname = os.path.splitext(os.path.basename(args.data_path))[0]
    # Save LOs
    with open(os.path.join(args.output_dir, f"los-{fname}.json"), "w") as f:
        json.dump(all_los, f)

    # Save arguments
    with open(os.path.join(args.output_dir, f"args-pmi-{fname}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Aggregate into top-k predictions per question
    classify_from_pmi(args.output_dir, topk=args.topk)


def add_arguments(parser):
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--data_path", required=True, type=str,
                        help="Path to a jsonl file of questions with an 'lo' field")
    parser.add_argument("--lo_type", required=True, type=str, choices=("actions", "facts"),
                        help="Type of LOs in the data")
    parser.add_argument("--topk", type=int, default=3, help="Number of LO predictions to keep per question")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
    parser.add_argument("--output_dir", type=str, default=argparse.SUPPRESS, help="Path to the output directory")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for DataLoader")
    parser.add_argument("--pad_to_multiple_of", type=int, default=8, help="Pad to multiple of")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
