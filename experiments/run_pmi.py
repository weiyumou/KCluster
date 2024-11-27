import argparse
import json
import os
import time
from functools import partial

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import BasePredictionWriter
from torch.utils.data import Dataset, DataLoader
from transformers.utils import logging

from core.model import LargeLangModel
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
        return f"{q2.header(1)}\n{str(q2)}\n\n{q1.header(2)}\n", str(q1)

    def __len__(self):
        # Conditional-prob matrix + marginal-prob vector
        return len(self.questions) ** 2 + len(self.questions)


def question_collate(batch: list[tuple[str, str]], tokenizer,
                     pad_to_multiple_of: int = None, ignore_idx: int = -100):
    contexts, texts = list(zip(*batch))

    inputs = tokenizer(text=contexts, text_pair=texts,
                       return_tensors="pt", return_token_type_ids=True,
                       padding=True, pad_to_multiple_of=pad_to_multiple_of)
    mask = torch.logical_not(inputs.pop("token_type_ids"))
    labels = torch.masked_fill(inputs["input_ids"], mask, ignore_idx)

    return inputs, labels


class PMI(L.LightningModule):
    def __init__(self, llm_path: str, **model_args):
        super().__init__()
        self.model = LargeLangModel.load_model(llm_path, device_map=self.device, **model_args)

    def forward(self, batch):
        inputs, labels = batch
        logits = self.model(**inputs).logits.transpose(-1, -2)  # (N, V, S)
        loss = F.cross_entropy(logits[..., :-1], labels[:, 1:], reduction="none")  # (N, S)
        return -torch.sum(loss, dim=-1)  # (N,)


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in output_dir, each containing the predictions of its respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

        # save `batch_indices` to get the information about the data index from prediction data
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))


def main(args):
    # Create a folder to store results
    if not hasattr(args, "output_dir"):
        args.output_dir = os.path.join("results", "pmi", time.asctime())
    args.output_dir = args.output_dir.replace(' ', '_').replace(':', '-')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = LargeLangModel.load_tokenizer(args.llm_path)

    # Load model
    model = PMI(args.llm_path, trust_remote_code=True,
                torch_dtype=torch.float16, attn_implementation="flash_attention_2")

    # Compute PMI
    dl = DataLoader(PairQuestion(args.data_path), batch_size=args.batch_size, pin_memory=True, shuffle=False,
                    collate_fn=partial(question_collate, tokenizer=tokenizer, pad_to_multiple_of=8))

    pred_writer = CustomWriter(output_dir=args.output_dir, write_interval="epoch")
    trainer = L.Trainer(accelerator="gpu", strategy="ddp", devices=-1, callbacks=[pred_writer], logger=False)
    trainer.predict(model, dataloaders=dl, return_predictions=False)

    # Save arguments
    save_file = os.path.splitext(os.path.basename(args.data_path))[0]
    with open(f"{args.output_dir}/args-{save_file}.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--data_path", required=True, type=str, help="Path to a jsonl file of questions")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
    parser.add_argument("--output_dir", type=str, default=argparse.SUPPRESS, help="Path to the output directory")

    cl_args = parser.parse_args()
    main(cl_args)
