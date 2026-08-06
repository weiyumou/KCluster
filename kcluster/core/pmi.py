"""Pointwise mutual information over LLM log-prob scores.

The raw state is a marginal vector and a conditional matrix in the orientation
every scoring job produces (see tasks/congruity.py): the ROW is the
conditioning variable. All derived matrices keep that orientation:

    conditionals[i, j] = log P(item_j | context_i)
    pmi_mat[i, j]      = log P(j | i) - log P(j)
    cond_mat[i, j]     = log P(j | i)   (each row is a distribution)

Square matrices (contexts and items are the same question set, i.e. question
congruity) support the joint-based ``normalize`` estimator and symmetrization.
Rectangular matrices (e.g. learning objectives x questions) support only the
raw estimator — ``log P(Q | L) - log P(Q)``, the Relevance measure of the
LAK 2026 paper.
"""

import os
from functools import cached_property

import numpy as np
from scipy.special import log_softmax, logsumexp


class PointwiseMutualInfo:
    def __init__(self,
                 marginals: np.ndarray,
                 conditionals: np.ndarray,
                 symmetric: bool = True,
                 normalize: bool = False):
        self._vec, self._mat = marginals, conditionals
        self.symmetric = symmetric
        self.normalize = normalize

    @classmethod
    def from_array(cls, mtx: np.ndarray, **kwargs) -> "PointwiseMutualInfo":
        """Reconstruct from a stacked ``[marginals; conditionals]`` array."""
        return cls(mtx[0], mtx[1:], **kwargs)

    @classmethod
    def from_npy(cls, path: str, **kwargs) -> "PointwiseMutualInfo":
        """Reconstruct from a locally saved .npy (the layout produced by ``save``)."""
        return cls.from_array(np.load(path), **kwargs)

    @classmethod
    def from_shards(cls, pmi_dir: str, nrows: int, ncols: int, **kwargs) -> "PointwiseMutualInfo":
        """Reassemble from the rank-stamped .pt shards written by CustomWriter.

        ``nrows`` is the number of conditioning contexts and ``ncols`` the
        number of scored items; the flat grid layout is the one produced by
        tasks/congruity.py (``ncols`` marginals, then the row-major grid).
        """
        import torch  # local import: shard loading is the only torch dependency here

        mtx = torch.full((nrows * ncols + ncols,), torch.inf)
        rank = 0
        while os.path.exists(fname := os.path.join(pmi_dir, f"batch_indices_{rank}.pt")):
            [batch_inds] = torch.load(fname)
            predictions = torch.load(os.path.join(pmi_dir, f"predictions_{rank}.pt"))
            for inds, preds in zip(batch_inds, predictions, strict=True):
                mtx[inds] = preds.float()
            rank += 1
        assert not torch.isinf(mtx).any(), "Loaded shards do not cover the expected grid size"

        mtx = mtx.reshape(-1, ncols).numpy()
        # first row is the marginals; second row and below is the conditionals
        return cls(mtx[0], mtx[1:], **kwargs)

    def save(self, path: str) -> None:
        """Persist the full raw state as a ``[marginals; conditionals]`` .npy file.

        Any derived matrix (joint_mat, pmi_mat, marginal, cond_mat) can be
        reconstructed from it via ``from_npy``. ``symmetric`` / ``normalize`` are
        reconstruction-time choices, so they are supplied again when loading —
        one saved file can produce every variant.
        """
        np.save(path, np.vstack([self._vec, self._mat]))

    @property
    def _is_square(self) -> bool:
        return self._mat.shape[0] == self._mat.shape[1]

    @cached_property
    def joint_mat(self) -> np.ndarray:
        if not self._is_square:
            raise ValueError("The joint distribution needs contexts and items to be the same set "
                             "(a square matrix); rectangular relevance uses the raw estimator "
                             "(normalize=False)")
        mat = self._mat + self._vec.reshape(-1, 1)  # (i, j) = log P(j | i) + log P(i)
        if self.symmetric:
            mat = np.logaddexp(mat, mat.T) - np.log(2)
        mat = log_softmax(mat, axis=None)  # normalize the log joint probabilities
        return mat  # so that each (i, j) entry represents log P(i, j)

    @cached_property
    def marginal(self) -> np.ndarray:
        return logsumexp(self.joint_mat, axis=0)  # (j,) = log P(j)

    @cached_property
    def cond_mat(self) -> np.ndarray:
        return log_softmax(self.joint_mat, axis=1)  # (i, j) = log P(j | i)

    @cached_property
    def pmi_mat(self) -> np.ndarray:
        if self.normalize:
            # Use the joint distribution normalized over the question set (square only)
            mat = self.cond_mat - self.marginal
        else:
            # Use the raw model log-probabilities (rectangular-capable)
            mat = self._mat - self._vec
        if self.symmetric and self._is_square:
            mat = (mat + mat.T) / 2  # make it symmetric
        return mat  # so that each (i, j) entry represents log P(j | i) - log P(j), if not symmetric
