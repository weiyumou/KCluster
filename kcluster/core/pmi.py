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

import itertools
import os
from collections.abc import Sequence
from functools import cached_property

import numpy as np
from scipy.special import log_softmax, logsumexp


def residualize(sim_mtx: np.ndarray, groups: Sequence, min_pairs: int = 30,
                verbose: bool = True) -> np.ndarray:
    """Remove the part of a similarity matrix that only reflects the pair of groups.

    Question congruity depends on the *format* of the two questions as well as
    on what they test: in a mixed-format bank a fill-in/fill-in pair scores far
    above a fill-in/multiple-choice pair whether or not the two are related, so
    clustering recovers format families instead of knowledge components.

    Each unordered pair of group labels is a stratum with its own location and
    scale, estimated from the pairs it contains. Every entry is then re-expressed
    as standard deviations above what is typical *for its own stratum*::

        adjusted[i, j] = (sim[i, j] - mu[g_i, g_j]) / sd[g_i, g_j]

    Subtracting the stratum mean is exactly the residual of an OLS fit of the
    similarities on a saturated set of stratum dummies (the design is saturated,
    so the fitted values are the stratum means); dividing by the stratum standard
    deviation is a further step OLS does not perform, and is what makes strata
    with different spreads comparable.

    The strata are fit on *all* their pairs rather than on known-unrelated ones:
    genuinely related pairs are a small minority of any stratum, and requiring
    them to be identified in advance would make the correction unusable for KC
    discovery, where no reference model exists. A stratum with fewer than
    ``min_pairs`` pairs, or with no spread, is standardized with the pooled
    statistics instead — a handful of pairs cannot support its own estimate.

    A single-group matrix is rescaled globally and therefore reordered nowhere:
    with one format there is nothing to correct, which is the right no-op.

    :param sim_mtx: A square, symmetric similarity matrix (e.g. ``pmi_mat``)
    :param groups: One label per row/column, e.g. each question's ``q_type``
    :param min_pairs: Smallest stratum that may be standardized on its own
    :param verbose: Print the per-stratum table (the nuisance being removed)
    :return: A new matrix in per-stratum standard-deviation units
    """
    n = len(sim_mtx)
    if sim_mtx.shape != (n, n):
        raise ValueError(f"residualize needs a square similarity matrix, got {sim_mtx.shape}")
    if len(groups) != n:
        raise ValueError(f"got {len(groups)} group labels for a {n}x{n} matrix")

    labels = np.asarray(groups)
    # The diagonal is a question against itself, not a pair of questions; it is
    # not evidence about any stratum (and affinity propagation overwrites it
    # with the preference), so it is excluded from every fit.
    off_diag = ~np.eye(n, dtype=bool)
    pooled_mu, pooled_sd = sim_mtx[off_diag].mean(), sim_mtx[off_diag].std()

    adjusted = np.empty_like(sim_mtx, dtype=float)
    rows, thin = [], []
    for a, b in itertools.combinations_with_replacement(sorted(set(labels)), 2):
        # Unordered: {a, b} covers both the (a, b) and the (b, a) cells.
        is_a, is_b = labels == a, labels == b
        stratum = np.outer(is_a, is_b) | np.outer(is_b, is_a)
        sample = sim_mtx[stratum & off_diag]
        if sample.size == 0:
            continue
        mu, sd = sample.mean(), sample.std()
        if sample.size < min_pairs or not sd:
            mu, sd, is_thin = pooled_mu, pooled_sd, True
            thin.append(f"{a} x {b}")
        else:
            is_thin = False
        adjusted[stratum] = (sim_mtx[stratum] - mu) / sd
        rows.append((f"{a} x {b}", sample.size, sample.mean(), sample.std(), is_thin))

    if verbose:
        print(f"*** Residualizing congruity over {len(rows)} strata "
              f"({len(set(labels))} groups, {n} questions) ***")
        # Question types are long enough to break a fixed column, so size the
        # label column to the widest one actually present.
        width = max(len(name) for name, *_ in rows)
        print(f"    {'stratum':<{width}}{'pairs':>9}{'mean':>9}{'sd':>8}")
        for name, size, mu, sd, is_thin in rows:
            note = "  <- pooled (too few pairs)" if is_thin else ""
            print(f"    {name:<{width}}{size:>9,}{mu:>9.3f}{sd:>8.3f}{note}")
    if thin:
        print(f"*** WARNING: {len(thin)} stratum/strata standardized with pooled statistics: "
              f"{', '.join(thin)} ***")
    return adjusted


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
