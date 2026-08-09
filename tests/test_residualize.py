"""Tests for the question-format residualization of a congruity matrix.

The correction has to do three things: reproduce the OLS fit it claims to be,
actually reorder pairs so content outranks format, and stay a no-op where there
is nothing to correct. Each is pinned below.
"""

import itertools

import numpy as np
import pytest

from kcluster.core.pmi import residualize


def _blocked(groups, effects: dict, rng=None) -> np.ndarray:
    """A symmetric matrix whose value depends only on the pair of group labels."""
    labels = np.asarray(groups)
    n = len(labels)
    mat = np.zeros((n, n))
    for i, j in itertools.product(range(n), repeat=2):
        mat[i, j] = effects[frozenset((labels[i], labels[j]))]
    if rng is not None:
        noise = rng.normal(size=(n, n))
        mat = mat + (noise + noise.T) / 2
    np.fill_diagonal(mat, 0.0)
    return mat


def test_subtracting_the_stratum_mean_is_the_ols_residual():
    # The claim in the docstring: a saturated fit on stratum dummies has the
    # stratum means as its coefficients, so its residuals are what we subtract.
    rng = np.random.default_rng(0)
    groups = ["FI"] * 12 + ["S1"] * 10 + ["SA"] * 8
    mat = _blocked(groups, {frozenset(p): v for p, v in
                            zip(itertools.combinations_with_replacement(("FI", "S1", "SA"), 2),
                                [19.0, 9.0, 8.5, 11.0, 10.0, 14.5], strict=True)}, rng=rng)
    labels = np.asarray(groups)
    iu = np.triu_indices(len(groups), k=1)
    y = mat[iu]

    strata = list(itertools.combinations_with_replacement(sorted(set(groups)), 2))
    design = np.column_stack([((labels[iu[0]] == a) & (labels[iu[1]] == b))
                              | ((labels[iu[0]] == b) & (labels[iu[1]] == a)) for a, b in strata]).astype(float)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    ols_residual = y - design @ beta

    # residualize() scales as well as centres, so compare after undoing the scale.
    adjusted = residualize(mat, groups, verbose=False)[iu]
    scale = np.where(ols_residual != 0, ols_residual / np.where(adjusted != 0, adjusted, np.nan), np.nan)
    within = [scale[design[:, k].astype(bool)] for k in range(len(strata))]
    for k, col in enumerate(within):
        col = col[~np.isnan(col)]
        # One constant divisor per stratum, i.e. the only difference is the sd.
        assert np.allclose(col, col[0]), f"stratum {strata[k]} is not a pure rescale of the OLS residual"


def test_every_stratum_is_centred_and_scaled():
    rng = np.random.default_rng(1)
    groups = ["FI"] * 15 + ["S1"] * 12 + ["SA"] * 9
    mat = _blocked(groups, {frozenset(p): v for p, v in
                            zip(itertools.combinations_with_replacement(("FI", "S1", "SA"), 2),
                                [19.0, 9.0, 8.5, 11.0, 10.0, 14.5], strict=True)}, rng=rng)
    adjusted = residualize(mat, groups, verbose=False)
    labels = np.asarray(groups)
    off = ~np.eye(len(groups), dtype=bool)
    for a, b in itertools.combinations_with_replacement(sorted(set(groups)), 2):
        is_a, is_b = labels == a, labels == b
        cell = adjusted[(np.outer(is_a, is_b) | np.outer(is_b, is_a)) & off]
        assert cell.mean() == pytest.approx(0.0, abs=1e-9)
        assert cell.std() == pytest.approx(1.0, abs=1e-9)


def test_content_outranks_format_after_the_correction():
    # The failure the correction exists to fix: an UNRELATED same-format pair
    # outscoring a RELATED cross-format pair, exactly as observed on the probe.
    groups = ["FI"] * 6 + ["S1"] * 6
    mat = _blocked(groups, {frozenset(("FI", "FI")): 19.0, frozenset(("FI", "S1")): 9.0,
                            frozenset(("S1", "S1")): 11.0})
    related, unrelated = (0, 6), (1, 2)          # FI-S1 same topic; FI-FI unrelated
    mat[related] = mat[related[::-1]] = 13.0     # a real content boost, cross-format
    mat[unrelated] = mat[unrelated[::-1]] = 16.0  # below par for FI-FI, still higher raw

    assert mat[unrelated] > mat[related]          # raw: format wins
    adjusted = residualize(mat, groups, verbose=False)
    assert adjusted[related] > adjusted[unrelated]  # adjusted: content wins
    assert adjusted[unrelated] < 0                  # and the unrelated pair is below par


def test_a_single_group_corpus_is_reordered_nowhere():
    # One format means one stratum: a global affine rescale, which cannot change
    # any ranking. Single-format banks (elearning22, ScienceQA) must be untouched.
    rng = np.random.default_rng(2)
    mat = rng.normal(size=(20, 20))
    mat = (mat + mat.T) / 2
    adjusted = residualize(mat, ["Multiple Choice"] * 20, verbose=False)
    off = ~np.eye(20, dtype=bool)
    assert np.array_equal(np.argsort(mat[off]), np.argsort(adjusted[off]))


def test_symmetry_and_shape_are_preserved():
    rng = np.random.default_rng(3)
    groups = ["FI"] * 10 + ["S1"] * 10
    mat = _blocked(groups, {frozenset(("FI", "FI")): 19.0, frozenset(("FI", "S1")): 9.0,
                            frozenset(("S1", "S1")): 11.0}, rng=rng)
    adjusted = residualize(mat, groups, verbose=False)
    assert adjusted.shape == mat.shape
    assert np.allclose(adjusted, adjusted.T)
    assert not np.shares_memory(adjusted, mat)  # the input is left alone


def test_a_thin_stratum_falls_back_to_pooled_statistics(capsys):
    # Two select-all questions make a single SA-SA pair: a mean of one value and
    # a standard deviation of zero. It must not be standardized on its own.
    rng = np.random.default_rng(4)
    groups = ["FI"] * 12 + ["S1"] * 12 + ["SA"] * 2
    mat = _blocked(groups, {frozenset(p): v for p, v in
                            zip(itertools.combinations_with_replacement(("FI", "S1", "SA"), 2),
                                [19.0, 9.0, 8.5, 11.0, 10.0, 14.5], strict=True)}, rng=rng)
    adjusted = residualize(mat, groups, min_pairs=30)
    assert np.isfinite(adjusted).all(), "a zero-spread stratum produced inf/nan"
    assert "pooled statistics" in capsys.readouterr().out


@pytest.mark.parametrize("groups, message", [
    (["a"] * 5, "group labels for a 4x4 matrix"),
])
def test_mismatched_labels_are_rejected(groups, message):
    with pytest.raises(ValueError, match=message):
        residualize(np.eye(4), groups)


def test_a_non_square_matrix_is_rejected():
    with pytest.raises(ValueError, match="square similarity matrix"):
        residualize(np.zeros((3, 5)), ["a", "b", "c"])
