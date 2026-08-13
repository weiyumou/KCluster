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


def _row_effects_fit(mtx: np.ndarray) -> np.ndarray:
    """Fitted values of the off-diagonal least-squares fit ``mu + a[i] + a[j]``.

    Closed form for a symmetric matrix: with ``rm`` the off-diagonal row means,
    the normal equations under the identification ``sum(a) == 0`` give
    ``mu = rm.mean()`` and ``a = (n - 1) * (rm - mu) / (n - 2)``. The fitted
    value is formed for every cell, diagonal included (``mu + 2 a[i]``).
    """
    n = len(mtx)
    rm = (mtx.sum(axis=1) - np.diagonal(mtx)) / (n - 1)
    mu = rm.mean()
    a = (n - 1) * (rm - mu) / (n - 2)
    return mu + a[:, None] + a[None, :]


def double_center(sim_mtx: np.ndarray) -> np.ndarray:
    """Remove per-item additive effects: the residual of ``mu + a[i] + a[j]``.

    The centering step of classical MDS, fit on the off-diagonal cells only and
    subtracted everywhere (the diagonal gets its fitted value ``mu + 2 a[i]``,
    consistent with :func:`residualize`). Off-diagonal row and column means of
    the result are zero, and the operation is idempotent.

    WARNING: on a mixed-format bank do not use this alone — stripping the item
    effects *promotes* the shared-format signal (measured on spacing-exp2:
    format-leakage AUC 0.695 -> 0.764, D11). Pair it with the format-mean
    correction, or use ``residualize(..., item_effects=True)``, which fits the
    item and format terms jointly.
    """
    n = len(sim_mtx)
    if sim_mtx.shape != (n, n) or n < 3:
        raise ValueError(f"double_center needs a square matrix with at least 3 rows, got {sim_mtx.shape}")
    # Fit in float64, for the reason given in residualize: a float32 input
    # otherwise leaves ~1e-6 in the row means this is supposed to zero.
    work = sim_mtx.astype(np.float64)
    return work - _row_effects_fit(work)


def residualize(sim_mtx: np.ndarray, groups: Sequence, *, item_effects: bool = False,
                min_pairs: int = 30, tol: float = 1e-10,
                max_iter: int = 50, verbose: bool = True) -> np.ndarray:
    """Remove the part of a similarity matrix that only reflects the pair of groups.

    Question congruity depends on the *format* of the two questions as well as
    on what they test: in a mixed-format bank a fill-in/fill-in pair scores far
    above a fill-in/multiple-choice pair whether or not the two are related, so
    clustering recovers format families instead of knowledge components.

    Each unordered pair of group labels is a stratum. The default subtracts each
    stratum's mean::

        adjusted[i, j] = sim[i, j] - mu[g_i, g_j]

    which is exactly the residual of an OLS fit of the similarities on a
    saturated set of stratum dummies (the design is saturated, so the fitted
    values are the stratum means). The output stays in the units of the input
    (nats for a PMI matrix) and remains spectrally comparable to it.

    ``item_effects=True`` returns the residual of the joint additive model::

        sim[i, j] ~ mu + a[i] + a[j] + gamma[g_i, g_j]

    per-item effects together with the format-pair means; it subsumes
    double-centering, and is the recommended correction for mixed-format banks
    (D11): beyond the format means it removes each question's tendency to score
    high with everything, which otherwise dominates retrieval and spectral uses.
    Fit by backfitting — alternating the exact stratum-mean and row-effect
    least-squares updates (block coordinate descent) until the change is below
    ``tol`` in relative Frobenius norm. The individual coefficients are not
    identified (``mu``, ``a`` and ``gamma`` share constants); that is harmless
    and expected — the *residual* is unique, so do not "fix" it.

    D9 additionally divided each stratum by its standard deviation. That step
    was removed in D11: it is not part of any least-squares residual, it
    measured worse than the mean-only correction on the one unbalanced bank,
    and the per-stratum rescaling distorts the spectrum. Matrices written
    before D11 are the z-scored variant and are not comparable with these.

    The strata are fit on *all* their pairs rather than on known-unrelated ones:
    genuinely related pairs are a small minority of any stratum, and requiring
    them to be identified in advance would make the correction unusable for KC
    discovery, where no reference model exists. A stratum with fewer than
    ``min_pairs`` pairs falls back to the pooled statistics — a handful of
    pairs cannot support its own estimate.

    A single-group matrix is shifted globally and therefore reordered nowhere:
    with one format there is nothing to correct, which is the right no-op
    (under ``item_effects=True`` the result equals ``double_center``).

    :param sim_mtx: A square, symmetric similarity matrix (e.g. ``pmi_mat``)
    :param groups: One label per row/column, e.g. each question's ``q_type``
    :param item_effects: Remove the joint item + format-pair model instead of
        the format-pair means alone
    :param min_pairs: Smallest stratum that may be fit on its own
    :param tol: Relative convergence threshold of the backfitting loop
    :param max_iter: Iteration cap of the backfitting loop
    :param verbose: Print the per-stratum table (the nuisance being removed)
    :return: A new float64 matrix, in the same units as the input
    """
    n = len(sim_mtx)
    if sim_mtx.shape != (n, n):
        raise ValueError(f"residualize needs a square similarity matrix, got {sim_mtx.shape}")
    if len(groups) != n:
        raise ValueError(f"got {len(groups)} group labels for a {n}x{n} matrix")
    if item_effects and n < 3:
        raise ValueError(f"item_effects=True needs at least 3 questions, got {n}")

    labels = np.asarray(groups)
    # Correct in float64 even when the input is float32 (what the local engine
    # saves): the correction is a difference of numbers around 20 nats over up
    # to ~1e6 cells per stratum, and float32 leaves ~1e-6 of rounding in the
    # very stratum means this is supposed to zero.
    work = sim_mtx.astype(np.float64)
    # The diagonal is a question against itself, not a pair of questions; it is
    # not evidence about any stratum (and affinity propagation overwrites it
    # with the preference), so it is excluded from every fit and transformed by
    # its fitted value.
    off_diag = ~np.eye(n, dtype=bool)
    pooled_mu = work[off_diag].mean()

    strata, rows, thin = [], [], []
    for a, b in itertools.combinations_with_replacement(sorted(set(labels)), 2):
        # Unordered: {a, b} covers both the (a, b) and the (b, a) cells.
        is_a, is_b = labels == a, labels == b
        stratum = np.outer(is_a, is_b) | np.outer(is_b, is_a)
        sample = work[stratum & off_diag]
        # An all-diagonal stratum (a single-question group) has nothing to
        # estimate, and is thin like any other.
        is_thin = sample.size < min_pairs
        if is_thin:
            thin.append(f"{a} x {b}")
        strata.append((stratum, is_thin, sample.mean() if sample.size else pooled_mu))
        if sample.size:
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
        print(f"*** WARNING: {len(thin)} stratum/strata fell back to pooled statistics: "
              f"{', '.join(thin)} ***")

    if not item_effects:
        adjusted = np.empty_like(work)
        for stratum, is_thin, mu in strata:
            adjusted[stratum] = work[stratum] - (pooled_mu if is_thin else mu)
        return adjusted

    # Backfitting for the joint model: each pass applies the exact
    # least-squares update of one block on the current residual, so the loop
    # is block coordinate descent and converges to the unique joint residual.
    residual = work.copy()
    scale_ref = np.linalg.norm(work[off_diag]) or 1.0
    for _ in range(max_iter):
        prev = residual.copy()
        pooled = residual[off_diag].mean()
        for stratum, is_thin, _ in strata:
            residual[stratum] -= pooled if is_thin else residual[stratum & off_diag].mean()
        residual -= _row_effects_fit(residual)
        if np.linalg.norm(residual - prev) <= tol * scale_ref:
            break
    else:
        print(f"*** WARNING: residualize(item_effects=True) did not converge "
              f"within {max_iter} iterations ***")
    return residual


def correction_variants(groups: Sequence, *, mean_only: bool = False,
                        joint: bool = False) -> list[tuple[str, dict]]:
    """The format corrections worth building for a bank with these ``groups``.

    Returns ``(artifact tag, residualize keyword arguments)`` pairs in the
    order they should be written, so both engines agree on which models a
    given pair of flags produces.

    With a single group the mean-only correction subtracts one constant from
    the whole matrix. Affinity propagation is invariant to that, so its KC
    model would be a duplicate of the uncorrected one and is dropped rather
    than written twice. The joint fit still earns its place on a single-format
    bank: removing the per-item effects is a real change with one stratum too,
    and it measured better than uncorrected on every single-format bank tried
    (D11) — with one format there is also no format contrast for it to
    promote, which is the hazard that keeps ``double_center`` out of the
    mixed-format path.
    """
    variants = []
    if mean_only and len(set(groups)) > 1:
        variants.append(("resid", {}))
    if joint:
        variants.append(("residfull", {"item_effects": True}))
    return variants


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
