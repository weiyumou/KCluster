import argparse
import glob
import json
import os
import time
from functools import partial, cached_property

import lightning as L
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.utils import logging

from core.model import LargeLangModel
from core.question import Question
from experiments.run_pmi import question_collate, PMI, CustomWriter

SPACE = Question.SPACE


class PointwiseMutualInfo:
    def __init__(self, pmi_dir: str, nrows: int, ncols: int, normalize: bool = True):
        self._vec, self._mat = self.load_probs(pmi_dir, nrows, ncols)
        self._normalize = normalize

    @cached_property
    def marginals(self) -> torch.Tensor:
        # normalize the marginals
        return torch.log_softmax(self._vec, dim=-1) if self._normalize else self._vec

    @cached_property
    def conditionals(self) -> torch.Tensor:
        # use PMI to re-calculate conditionals
        return self.pmi_mat + self.marginals

    @cached_property
    def pmi_mat(self) -> torch.Tensor:
        mat = torch.log_softmax(self._mat, dim=-1) if self._normalize else self._mat
        mat = mat - self.marginals
        if mat.shape[0] == mat.shape[1]:
            mat = (mat + mat.T) / 2  # make symmetric
        return mat

    @staticmethod
    def load_probs(pmi_dir: str, nrows: int, ncols: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load conditional and marginal log-probabilities from a result folder
        :param pmi_dir: A path to saved log-probability data produced by 'run_pmi.py'
        :param nrows: The number of conditioning items
        :param ncols: The number of conditioned items
        :return: A Tuple consisting of
         - a `ncols` vector of marginal log-probabilities
         - a `nrows x ncols` conditional-log-probability matrix
        """
        # Prepare an empty matrix of the appropriate size to fill in
        mtx = torch.full((nrows * ncols + ncols,), torch.inf)

        # Load data from all available files
        rank = 0
        while os.path.exists(fname := os.path.join(pmi_dir, f"batch_indices_{rank}.pt")):
            batch_inds = torch.load(fname)[0]
            predictions = torch.load(os.path.join(pmi_dir, f"predictions_{rank}.pt"))
            for inds, preds in zip(batch_inds, predictions, strict=True):
                mtx[inds] = preds
            rank += 1
        assert torch.isinf(mtx).sum() == 0, "Number of questions doesn't match the save value"

        mtx = mtx.reshape(-1, ncols).float()
        marginals = mtx[0]  # first row is the marginals
        conds = mtx[1:]  # second row and below is the conditionals
        return marginals, conds


class QuestionLO(Dataset):
    def __init__(self, questions: list[Question], los: list[str], lo_type: str):
        match lo_type:
            case "actions":
                self.los = [lo[0].lower() + lo[1:].rstrip().rstrip(".") for lo in los]
                self.header = f"The exercise below is designed to test whether a student can{SPACE}"
            case "facts":
                self.los = [lo.rstrip().rstrip(".") for lo in los]
                self.header = f"The exercise below is designed to test whether a student knows:\n"
            case _:
                raise ValueError(f"Invalid lo_type: '{lo_type}'")

        self.questions = questions

    def __getitem__(self, index):
        n = len(self.questions)
        if index < n:  # calculate marginal log-probabilities
            q = self.questions[index]
            return f"{q.q_type}:\n", str(q)

        # calculate conditional log-probabilities
        lo_idx, q_idx = (index - n) // n, (index - n) % n
        lo, q = self.los[lo_idx], self.questions[q_idx]
        header = f"{self.header}{lo}."
        return f"{header}\n\n{q.q_type}:\n", str(q)

    def __len__(self):
        return len(self.questions) * len(self.los) + len(self.questions)


def classify_from_pmi(root_dir: str, topk: int = 3) -> pd.DataFrame:
    torch.set_default_dtype(torch.float16)

    # Read the args json file to extract data_path
    [args_f] = glob.glob(os.path.join(root_dir, "args-pmi-*.json"))
    with open(args_f, "r") as f:
        args = json.load(f)

    # Load questions from data_path
    with open(args["data_path"], "r") as f:
        all_questions = [Question(eval(line)) for line in f]

    # Load LOs
    [los_f] = glob.glob(os.path.join(root_dir, "los-*.json"))
    with open(los_f, "r") as f:
        all_los = json.load(f)

    pmi = PointwiseMutualInfo(root_dir, len(all_los), len(all_questions), normalize=False)

    # Top K predictions
    preds = torch.topk(pmi.pmi_mat, k=topk, dim=0).indices.T  # (n_questions, topk)
    records, matched = [], []
    for idx, q in enumerate(all_questions):
        d, is_matched = q.flat_dict, False
        for k, p in enumerate(preds[idx], 1):
            d |= {f"pred_lo_{k}": all_los[p]}
            is_matched |= q["lo"] == all_los[p]
        records.append(d)
        if is_matched:
            matched.append(q)

    # Save matched questions
    with open(os.path.join(root_dir, f"matched-top{topk}.jsonl"), "w") as f:
        f.write("\n".join(repr(q) for q in matched))
    print(f"Matched {len(matched)} questions out of {len(all_questions)}")

    # Save classification results
    res_df = pd.DataFrame.from_records(records)
    res_df.to_csv(os.path.join(root_dir, f"classified-top{topk}.csv"), index=False)

    return res_df


def main(args):
    # Create a folder to store results
    if not hasattr(args, "output_dir"):
        args.output_dir = os.path.join("results", "classify", time.asctime())
    args.output_dir = args.output_dir.replace(' ', '_').replace(':', '-')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = LargeLangModel.load_tokenizer(args.llm_path)

    # Load model
    model = PMI(args.llm_path, trust_remote_code=True,
                torch_dtype=torch.float16, attn_implementation="flash_attention_2")

    # Read all questions
    with open(args.data_path, "r") as f:
        all_questions = [Question(eval(line)) for line in f]

    # Collect all LOs
    if hasattr(args, "lo_path"):
        with open(args.lo_path, "r") as f:
            all_los = [line.strip() for line in f]
    else:
        all_los = list(set(q["lo"] for q in all_questions))

    # Compute PMI for questions and LOs
    dl = DataLoader(QuestionLO(all_questions, all_los, args.lo_type), batch_size=args.batch_size,
                    pin_memory=True, shuffle=False, num_workers=2,
                    collate_fn=partial(question_collate, tokenizer=tokenizer, pad_to_multiple_of=8))
    pred_writer = CustomWriter(output_dir=args.output_dir, write_interval="epoch")
    trainer = L.Trainer(accelerator="gpu", devices=-1, callbacks=[pred_writer], logger=False)
    trainer.predict(model, dataloaders=dl, return_predictions=False)

    fname = os.path.splitext(os.path.basename(args.data_path))[0]
    # Save LOs
    with open(os.path.join(args.output_dir, f"los-{fname}.json"), "w") as f:
        json.dump(all_los, f)

    # Save arguments
    with open(os.path.join(args.output_dir, f"args-classify-{fname}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    L.seed_everything(42)
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--data_path", required=True, type=str, help="Path to a file containing questions")
    parser.add_argument("--lo_type", required=True, type=str, choices=("actions", "facts"),
                        help="Type of LOs in the data")
    parser.add_argument("--lo_path", type=str, default=argparse.SUPPRESS, help="Path to a file containing LOs")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
    parser.add_argument("--output_dir", type=str, default=argparse.SUPPRESS, help="Path to the output directory")

    cl_args = parser.parse_args()
    main(cl_args)
