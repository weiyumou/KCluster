import json
import os
from functools import partial

import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers.utils import logging

from kcluster.engine.local import CustomWriter, LargeLangModel, LogProbScorer, collate_pair
from kcluster.io.jsonl import load_questions
from kcluster.paths import dataset_name, default_output_dir, prepare_output_dir, save_args
from kcluster.tasks.classify import QuestionLO, classify_from_pmi


def main(args):
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    # Create a folder to store results
    output_dir = args.output_dir or default_output_dir("classify", args.run_dir)
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

    ds = dataset_name(args.data_path)
    # Save LOs
    with open(os.path.join(args.output_dir, f"los-{ds}.json"), "w") as f:
        json.dump(all_los, f)

    save_args(args.output_dir, "pmi", ds, vars(args))

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
    parser.add_argument("--output_dir", type=str, default=None, help="Path to the output directory")
    parser.add_argument("--run_dir", default=None, type=str,
                        help="Shared run folder; each step writes to <run_dir>/<step> (env: KCLUSTER_RUN_DIR)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for DataLoader")
    parser.add_argument("--pad_to_multiple_of", type=int, default=8, help="Pad to multiple of")
