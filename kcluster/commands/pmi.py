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
from kcluster.tasks.congruity import PairQuestion


def main(args):
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    # Create a folder to store results
    output_dir = getattr(args, "output_dir", None) or default_output_dir("pmi", getattr(args, "run_dir", None))
    args.output_dir = prepare_output_dir(output_dir)
    print(f"*** Writing results to {args.output_dir} ***")

    # Load tokenizer
    tokenizer = LargeLangModel.load_tokenizer(args.llm_path)

    # Load model
    model = LogProbScorer(args.llm_path, trust_remote_code=True, torch_dtype=torch.float16)

    # Compute PMI
    questions = load_questions(args.data_path)
    ds = PairQuestion(questions)
    dl = DataLoader(ds, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=args.num_workers,
                    collate_fn=partial(collate_pair, tokenizer=tokenizer, pad_to_multiple_of=args.pad_to_multiple_of))

    pred_writer = CustomWriter(output_dir=args.output_dir, write_interval="epoch")
    trainer = L.Trainer(accelerator="gpu", strategy="ddp", devices=-1, callbacks=[pred_writer], logger=False)
    trainer.predict(model, dataloaders=dl, return_predictions=False)

    # Save arguments
    fname = os.path.splitext(os.path.basename(args.data_path))[0]
    with open(os.path.join(args.output_dir, f"args-pmi-{fname}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def add_arguments(parser):
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--data_path", required=True, type=str, help="Path to a jsonl file of questions")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
    parser.add_argument("--output_dir", type=str, default=argparse.SUPPRESS, help="Path to the output directory")
    parser.add_argument("--run_dir", default=argparse.SUPPRESS, type=str,
                        help="Shared run folder; each step writes to <run_dir>/<step> (env: KCLUSTER_RUN_DIR)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--pad_to_multiple_of", type=int, default=None, help="Pad to multiple of")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
