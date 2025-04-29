import argparse
import json
import os
import time
from functools import partial

import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.utils import logging

from core.model import LargeLangModel
from core.pmi import PMI, CustomWriter, collate_pair
from core.question import Question


class PairQuestion(Dataset):
    def __init__(self, data_path: str):
        with open(data_path, "r") as f:
            self.questions = [Question(eval(line)) for line in f]

    def __getitem__(self, index):
        n = len(self.questions)
        if index < n:
            q = self.questions[index]
            return f"{q.header(2)}\n", str(q)
        row, col = (index - n) // n, (index - n) % n
        q1, q2 = self.questions[row], self.questions[col]
        return f"{q1.header(1)}\n{str(q1)}\n\n{q2.header(2)}\n", str(q2)

    def __len__(self):
        # Conditional-prob matrix + marginal-prob vector
        return len(self.questions) ** 2 + len(self.questions)


def main(args):
    # Create a folder to store results
    args.output_dir = getattr(args, "output_dir", os.path.join("results", "pmi", time.asctime()))
    args.output_dir = args.output_dir.replace(' ', '_').replace(':', '-')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = LargeLangModel.load_tokenizer(args.llm_path)

    # Load model
    model = PMI(args.llm_path, trust_remote_code=True,
                torch_dtype=torch.float16, attn_implementation="flash_attention_2")

    # Compute PMI
    ds = PairQuestion(args.data_path)
    dl = DataLoader(ds, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=args.num_workers,
                    collate_fn=partial(collate_pair, tokenizer=tokenizer, pad_to_multiple_of=args.pad_to_multiple_of))

    pred_writer = CustomWriter(output_dir=args.output_dir, write_interval="epoch")
    trainer = L.Trainer(accelerator="gpu", strategy="ddp", devices=-1, callbacks=[pred_writer], logger=False)
    trainer.predict(model, dataloaders=dl, return_predictions=False)

    # Save arguments
    fname = os.path.splitext(os.path.basename(args.data_path))[0]
    with open(os.path.join(args.output_dir, f"args-{fname}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--data_path", required=True, type=str, help="Path to a jsonl file of questions")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
    parser.add_argument("--output_dir", type=str, default=argparse.SUPPRESS, help="Path to the output directory")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--pad_to_multiple_of", type=int, default=None, help="Pad to multiple of")

    cl_args = parser.parse_args()
    main(cl_args)
