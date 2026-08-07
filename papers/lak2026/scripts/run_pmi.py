"""Compute per-LO congruity shards for every validated question set.

Study orchestration over the library's congruity task: for each generated-
question directory, one shard directory per LO (LO-<idx>/) plus the
lo_mappings.json that build_kc.py reads.
"""

import argparse
import glob
import json
import os
import time
from collections import defaultdict
from functools import partial

import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers.utils import logging

from kcluster.engine.local import CustomWriter, LargeLangModel, LogProbScorer, collate_pair
from kcluster.io.jsonl import load_questions
from kcluster.tasks.congruity import PairQuestion


def main(args):
    L.seed_everything(42)
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    # Create a folder to store results
    if not hasattr(args, "output_dir"):
        args.output_dir = os.path.join("results", "kcluster", time.strftime("%Y%m%d-%H%M%S"))

    # Load tokenizer and model
    tokenizer = LargeLangModel.load_tokenizer(args.llm_path)
    model = LogProbScorer(args.llm_path, trust_remote_code=True, torch_dtype=torch.float16)

    pat = "-gpt" if args.use_gpt_valid else ""
    for fname in sorted(glob.iglob(f"**/new-mcq{pat}-valid.jsonl", recursive=True, root_dir=args.root_dir)):
        curr_dir = os.path.split(fname)[0]
        output_dir = os.path.join(args.output_dir, curr_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Read all questions by LO
        all_questions = defaultdict(list)
        for q in load_questions(os.path.join(args.root_dir, fname)):
            all_questions[q["lo"]].append(q)

        # Compute PMI for questions under each LO
        lo_mappings = dict()
        for idx, lo in enumerate(all_questions):
            lo_mappings[f"LO-{idx}"] = lo
            pred_dir = os.path.join(output_dir, f"LO-{idx}")
            os.makedirs(pred_dir, exist_ok=True)

            dl = DataLoader(PairQuestion(all_questions[lo]), batch_size=args.batch_size,
                            pin_memory=True, shuffle=False, num_workers=2,
                            collate_fn=partial(collate_pair, tokenizer=tokenizer, pad_to_multiple_of=8))
            pred_writer = CustomWriter(output_dir=pred_dir, write_interval="epoch")
            trainer = L.Trainer(accelerator="gpu", devices=1, callbacks=[pred_writer], logger=False)
            trainer.predict(model, dataloaders=dl, return_predictions=False)

        with open(os.path.join(output_dir, "lo_mappings.json"), "w") as f:
            json.dump(lo_mappings, f, indent=2)

    # Save arguments
    with open(f"{args.output_dir}/args-pmi.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--root_dir", required=True, type=str, help="Path to a root directory of generated questions")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
    parser.add_argument("--output_dir", type=str, default=argparse.SUPPRESS, help="Path to the output directory")
    parser.add_argument("--use_gpt_valid", action="store_true", help="Whether to use GPT-valid questions")

    main(parser.parse_args())
