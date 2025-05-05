import os
from functools import cached_property

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import BasePredictionWriter

from core.model import LargeLangModel


def collate_pair(batch: list[tuple[str, str]], tokenizer,
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


class PointwiseMutualInfo:
    def __init__(self, pmi_dir: str, nrows: int, ncols: int, cond_on_dim: int = 0,
                 normalize: bool = False, symmetric: bool = True, dtype: torch.dtype = torch.float16):
        self._vec, self._mat = self.load_probs(pmi_dir, nrows, ncols, cond_on_dim, dtype)
        self._normalize, self._symmetric = normalize, symmetric

    @cached_property
    def marginals(self) -> torch.Tensor:
        # normalize the marginals
        return torch.log_softmax(self._vec, dim=-1) if self._normalize else self._vec

    @cached_property
    def conditionals(self) -> torch.Tensor:
        # use PMI to re-calculate conditionals
        return torch.transpose(self.pmi_mat.T + self.marginals, 0, 1)

    @cached_property
    def pmi_mat(self) -> torch.Tensor:
        mat = torch.log_softmax(self._mat, dim=-1) if self._normalize else self._mat
        mat = mat - self.marginals
        if self._symmetric and (mat.shape[0] == mat.shape[1]):
            mat = (mat + mat.T) / 2  # make symmetric
        return mat.T  # so that each (i, k) entry represents log P(i | k) - log P(i), if not symmetric

    @staticmethod
    def load_probs(pmi_dir: str, nrows: int, ncols: int,
                   cond_on_dim: int = 0, dtype: torch.dtype = torch.float16) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load conditional and marginal log-probabilities from a result folder
        :param pmi_dir: A path to saved log-probability data produced by 'run_pmi.py'
        :param nrows: The number of conditioning items
        :param ncols: The number of conditioned items
        :param cond_on_dim: The index of the conditioning dimension; set cond_on_dim=1 to load legacy values
        :param dtype: The data type of the loaded tensors
        :return: A Tuple consisting of
         - a `ncols` vector of marginal log-probabilities
         - a `nrows x ncols` conditional-log-probability matrix
        """
        # Prepare an empty matrix of the appropriate size to fill in
        mtx = torch.full((nrows * ncols + ncols,), torch.inf, dtype=dtype)

        # Load data from all available files
        rank = 0
        while os.path.exists(fname := os.path.join(pmi_dir, f"batch_indices_{rank}.pt")):
            [batch_inds] = torch.load(fname)
            predictions = torch.load(os.path.join(pmi_dir, f"predictions_{rank}.pt"))
            for inds, preds in zip(batch_inds, predictions, strict=True):
                mtx[inds] = preds
            rank += 1
        assert torch.isinf(mtx).sum() == 0, "The size of the matrix does not match what is saved"

        mtx = mtx.reshape(-1, ncols).float()
        # first row is the marginals; second row and below is the conditionals
        marginals, conds = mtx[0], mtx[1:]
        if cond_on_dim == 1:  # keep dim 0 as the conditioning dimension
            conds = conds.T
        return marginals, conds
