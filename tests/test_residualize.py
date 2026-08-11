"""Tests for the question-format residualization of a congruity matrix.

The correction has to do four things: reproduce the OLS fit it claims to be,
actually reorder pairs so content outranks format, stay a no-op where there is
nothing to correct, and — for the joint item + format model D11 added — leave
both nuisance families at zero. Each is pinned below.
"""

import itertools

import numpy as np
import pytest

from kcluster.core.pmi import double_center, residualize

SIX_EFFECTS = dict(zip([frozenset(p) for p in
                        itertools.combinations_with_replacement(("FI", "S1", "SA"), 2)],
                       [19.0, 9.0, 8.5, 11.0, 10.0, 14.5], strict=True))


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


def _with_item_effects(mat, rng) -> np.ndarray:
    """Add per-item additive effects, the second nuisance of the joint model."""
    item = rng.normal(scale=2.0, size=len(mat))
    return mat + item[:, None] + item[None, :]


def _offdiag_row_means(mat) -> np.ndarray:
    n = len(mat)
    return (mat.sum(axis=1) - np.diagonal(mat)) / (n - 1)


def _stratum_means(mat, groups):
    labels = np.asarray(groups)
    off = ~np.eye(len(mat), dtype=bool)
    for a, b in itertools.combinations_with_replacement(sorted(set(groups)), 2):
        is_a, is_b = labels == a, labels == b
        yield mat[(np.outer(is_a, is_b) | np.outer(is_b, is_a)) & off].mean()


def test_default_is_the_saturated_ols_residual():
    # The D9 anchor, promoted from an equivalence claim to a direct test: a
    # saturated fit on stratum dummies has the stratum means as coefficients,
    # so the mean-only default IS the OLS residual (D9 measured 2.84e-14).
    rng = np.random.default_rng(1)
    groups = ["FI"] * 12 + ["S1"] * 10 + ["SA"] * 8
    mat = _blocked(groups, SIX_EFFECTS, rng=rng)
    labels = np.asarray(groups)
    iu = np.triu_indices(len(groups), k=1)
    y = mat[iu]

    strata = list(itertools.combinations_with_replacement(sorted(set(groups)), 2))
    design = np.column_stack([((labels[iu[0]] == a) & (labels[iu[1]] == b))
                              | ((labels[iu[0]] == b) & (labels[iu[1]] == a)) for a, b in strata]).astype(float)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    ols_residual = y - design @ beta

    adjusted = residualize(mat, groups, verbose=False)
    assert np.abs(adjusted[iu] - ols_residual).max() < 1e-12


def test_every_stratum_is_centred():
    rng = np.random.default_rng(2)
    groups = ["FI"] * 15 + ["S1"] * 12 + ["SA"] * 9
    mat = _blocked(groups, SIX_EFFECTS, rng=rng)
    labels = np.asarray(groups)
    off = ~np.eye(len(groups), dtype=bool)

    centred = residualize(mat, groups, verbose=False)
    for a, b in itertools.combinations_with_replacement(sorted(set(groups)), 2):
        is_a, is_b = labels == a, labels == b
        stratum = (np.outer(is_a, is_b) | np.outer(is_b, is_a)) & off
        assert centred[stratum].mean() == pytest.approx(0.0, abs=1e-9)
        # The stratum spread is left alone: D11 removed the D9 sd division,
        # which is not part of any least-squares residual.
        assert centred[stratum].std() == pytest.approx(mat[stratum].std(), rel=1e-9)


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


def test_a_single_group_corpus_is_a_constant_shift():
    # One format means one stratum: the default subtracts a single constant,
    # which affinity propagation is invariant to. Single-format banks
    # (elearning22, ScienceQA) must be reordered nowhere.
    rng = np.random.default_rng(3)
    mat = rng.normal(size=(20, 20))
    mat = (mat + mat.T) / 2
    adjusted = residualize(mat, ["Multiple Choice"] * 20, verbose=False)
    off = ~np.eye(20, dtype=bool)
    assert np.allclose(adjusted, mat - mat[off].mean())
    assert np.array_equal(np.argsort(mat[off]), np.argsort(adjusted[off]))


def test_a_single_group_corpus_with_item_effects_is_double_centering():
    # With one stratum the format term degenerates to a constant, so the joint
    # model reduces to the item-effects fit alone.
    rng = np.random.default_rng(4)
    mat = rng.normal(size=(20, 20))
    mat = (mat + mat.T) / 2
    joint = residualize(mat, ["Multiple Choice"] * 20, item_effects=True, verbose=False)
    assert np.allclose(joint, double_center(mat))


@pytest.mark.parametrize("seed", [5, 6, 7])
def test_joint_residual_zeroes_stratum_and_row_means(seed):
    # The joint residual is characterized by BOTH nuisance families being
    # removed: every stratum's off-diagonal mean and every off-diagonal
    # row/column mean end up at zero, well within the backfitting iteration cap.
    rng = np.random.default_rng(seed)
    groups = ["FI"] * 20 + ["S1"] * 12 + ["SA"] * 8   # unbalanced on purpose
    mat = _with_item_effects(_blocked(groups, SIX_EFFECTS, rng=rng), rng)
    joint = residualize(mat, groups, item_effects=True, verbose=False)
    assert joint.dtype == np.float64
    assert np.allclose(joint, joint.T)
    assert not np.shares_memory(joint, mat)
    assert np.abs(_offdiag_row_means(joint)).max() < 1e-8
    assert max(abs(m) for m in _stratum_means(joint, groups)) < 1e-8


def test_joint_equals_sequential_on_a_balanced_bank():
    # Format-mean removal followed by exact double-centering lands on the same
    # residual as the joint fit when the groups are balanced (D11).
    rng = np.random.default_rng(8)
    groups = ["FI"] * 10 + ["S1"] * 10 + ["SA"] * 10
    mat = _with_item_effects(_blocked(groups, SIX_EFFECTS, rng=rng), rng)
    joint = residualize(mat, groups, item_effects=True, verbose=False)
    sequential = double_center(residualize(mat, groups, verbose=False))
    assert np.allclose(joint, sequential)


def test_joint_beats_one_pass_centering_on_an_unbalanced_bank():
    # "Sequential single passes are not exact" (D11): a single classical
    # centering pass after the format-mean step is what the evidence run
    # measured, and it leaves row means standing that the joint fit removes.
    # (Iterated to convergence the sequential route agrees with the joint fit —
    # the exact block projections commute — which is why residualize backfits.)
    rng = np.random.default_rng(9)
    groups = ["FI"] * 20 + ["S1"] * 12 + ["SA"] * 8
    mat = _with_item_effects(_blocked(groups, SIX_EFFECTS, rng=rng), rng)

    centred = residualize(mat, groups, verbose=False)
    rm = _offdiag_row_means(centred)
    one_pass = centred - rm[:, None] - rm[None, :] + rm.mean()

    joint = residualize(mat, groups, item_effects=True, verbose=False)
    off = ~np.eye(len(groups), dtype=bool)
    assert np.abs(_offdiag_row_means(one_pass)).max() > 1e-6   # nuisance left standing
    assert not np.allclose(joint, one_pass)
    assert np.sum(joint[off] ** 2) <= np.sum(one_pass[off] ** 2)  # joint is the LS residual


def test_symmetry_and_shape_are_preserved():
    rng = np.random.default_rng(10)
    groups = ["FI"] * 10 + ["S1"] * 10
    mat = _blocked(groups, {frozenset(("FI", "FI")): 19.0, frozenset(("FI", "S1")): 9.0,
                            frozenset(("S1", "S1")): 11.0}, rng=rng)
    adjusted = residualize(mat, groups, verbose=False)
    assert adjusted.shape == mat.shape
    assert np.allclose(adjusted, adjusted.T)
    assert not np.shares_memory(adjusted, mat)  # the input is left alone


def test_a_thin_stratum_falls_back_to_pooled_statistics(capsys):
    # Two select-all questions make a single SA-SA pair: it must not be fit on
    # its own, on any path.
    rng = np.random.default_rng(11)
    groups = ["FI"] * 12 + ["S1"] * 12 + ["SA"] * 2
    mat = _blocked(groups, SIX_EFFECTS, rng=rng)
    adjusted = residualize(mat, groups, min_pairs=30)
    assert np.isfinite(adjusted).all()
    assert "pooled statistics" in capsys.readouterr().out
    assert np.isfinite(residualize(mat, groups, min_pairs=30, item_effects=True, verbose=False)).all()


@pytest.mark.parametrize("item_effects", [False, True])
def test_a_float32_matrix_is_corrected_in_float64(item_effects):
    # The local engine saves congruity as float32. Correcting in that dtype
    # leaves ~1e-6 of rounding in the stratum means the correction is supposed
    # to zero, so both live paths upcast first.
    rng = np.random.default_rng(13)
    groups = ["FI"] * 60 + ["S1"] * 40
    mat = _blocked(groups, {frozenset(("FI", "FI")): 19.0, frozenset(("FI", "S1")): 9.0,
                            frozenset(("S1", "S1")): 11.0}, rng=rng).astype(np.float32)
    adjusted = residualize(mat, groups, item_effects=item_effects, verbose=False)
    assert adjusted.dtype == np.float64
    assert max(abs(m) for m in _stratum_means(adjusted, groups)) < 1e-12


def test_the_retired_zscoring_is_not_reachable():
    # D11 removed the D9 sd division outright rather than leaving it behind a
    # flag: it is not part of any least-squares residual, it measured worse
    # than the mean-only correction on the one unbalanced bank, and it
    # distorts the spectrum of the output.
    with pytest.raises(TypeError, match="scale"):
        residualize(np.zeros((4, 4)), ["a"] * 4, scale=True)


@pytest.mark.parametrize("groups, message", [
    (["a"] * 5, "group labels for a 4x4 matrix"),
])
def test_mismatched_labels_are_rejected(groups, message):
    with pytest.raises(ValueError, match=message):
        residualize(np.eye(4), groups)


def test_a_non_square_matrix_is_rejected():
    with pytest.raises(ValueError, match="square similarity matrix"):
        residualize(np.zeros((3, 5)), ["a", "b", "c"])


def test_double_center_zeroes_row_means_and_is_idempotent():
    rng = np.random.default_rng(12)
    mat = rng.normal(size=(15, 15))
    mat = (mat + mat.T) / 2
    centred = double_center(mat)
    assert np.abs(_offdiag_row_means(centred)).max() < 1e-12
    assert np.abs(_offdiag_row_means(centred.T)).max() < 1e-12
    assert np.allclose(double_center(centred), centred)
    assert np.allclose(centred, centred.T)


def test_double_center_rejects_degenerate_input():
    with pytest.raises(ValueError, match="at least 3 rows"):
        double_center(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="square matrix"):
        double_center(np.zeros((3, 5)))
