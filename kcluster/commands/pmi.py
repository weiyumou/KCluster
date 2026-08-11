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
from kcluster.paths import default_result_dir, pmi_raw_dir, prepare_output_dir
from kcluster.tasks.congruity import PairQuestion


def main(args):
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    # Resolve the per-dataset result folder (D10 layout); the raw score
    # shards are an intermediate, so they live under mat/pmi/raw/
    result_dir = getattr(args, "output_dir", None) or default_result_dir(getattr(args, "run_dir", None))
    args.output_dir = result_dir = prepare_output_dir(result_dir)
    shard_dir = prepare_output_dir(pmi_raw_dir(result_dir))
    print(f"*** Writing results to {result_dir} ***")

    # Load tokenizer
    tokenizer = LargeLangModel.load_tokenizer(args.llm_path)

    # Load model
    model = LogProbScorer(args.llm_path, trust_remote_code=True, torch_dtype=torch.float16)

    # Compute PMI
    questions = load_questions(args.data_path)
    ds = PairQuestion(questions)
    dl = DataLoader(ds, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=args.num_workers,
                    collate_fn=partial(collate_pair, tokenizer=tokenizer, pad_to_multiple_of=args.pad_to_multiple_of))

    pred_writer = CustomWriter(output_dir=shard_dir, write_interval="epoch")
    trainer = L.Trainer(accelerator="gpu", strategy="ddp", devices=-1, callbacks=[pred_writer], logger=False)
    trainer.predict(model, dataloaders=dl, return_predictions=False)

    # Save arguments at the result root
    ds = os.path.splitext(os.path.basename(args.data_path))[0].replace(" ", "-")
    with open(os.path.join(result_dir, f"args-pmi-{ds}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def add_arguments(parser):
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--data_path", required=True, type=str, help="Path to a jsonl file of questions")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
    parser.add_argument("--output_dir", type=str, default=argparse.SUPPRESS,
                        help="The result directory (default: --run_dir or a fresh timestamped folder)")
    parser.add_argument("--run_dir", default=argparse.SUPPRESS, type=str,
                        help="Result folder shared by every step of this run (env: KCLUSTER_RUN_DIR)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--pad_to_multiple_of", type=int, default=None, help="Pad to multiple of")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
