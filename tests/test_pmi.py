"""Tests for the canonical PointwiseMutualInfo.

Ports the serving repo's four tests (built on from_array construction instead
of a GCS monkeypatch) and adds fixtures pinning the row-conditioning
orientation contract, the rectangular relevance path, and shard reassembly.
"""

import numpy as np
import pytest
from scipy.special import logsumexp

from kcluster.core.pmi import PointwiseMutualInfo


@pytest.fixture
def square_probs():
    rng = np.random.default_rng(42)
    return rng.normal(-60, 10, size=5), rng.normal(-40, 10, size=(5, 5))


@pytest.fixture
def rect_probs():
    # 2 conditioning contexts (rows) x 3 scored items (columns)
    rng = np.random.default_rng(7)
    return rng.normal(-60, 10, size=3), rng.normal(-40, 10, size=(2, 3))


@pytest.mark.parametrize("normalize", [False, True])
def test_square_pmi_is_finite_and_symmetric(square_probs, normalize):
    pmi = PointwiseMutualInfo(*square_probs, symmetric=True, normalize=normalize)
    assert pmi.pmi_mat.shape == (5, 5)
    assert np.isfinite(pmi.pmi_mat).all()
    assert np.allclose(pmi.pmi_mat, pmi.pmi_mat.T)


def test_joint_is_a_normalized_log_distribution(square_probs):
    assert abs(logsumexp(PointwiseMutualInfo(*square_probs).joint_mat)) < 1e-9


def test_cond_mat_rows_are_distributions_over_items(square_probs):
    # cond_mat[i, j] = log P(j | i): each ROW must sum to one.
    pmi = PointwiseMutualInfo(*square_probs, symmetric=False)
    assert np.allclose(logsumexp(pmi.cond_mat, axis=1), 0.0)


def test_normalize_flag_changes_the_matrix(square_probs):
    raw = PointwiseMutualInfo(*square_probs, normalize=False).pmi_mat
    norm = PointwiseMutualInfo(*square_probs, normalize=True).pmi_mat
    assert not np.allclose(raw, norm)


def test_raw_pmi_orientation_is_row_conditioning(square_probs):
    # The orientation contract: pmi_mat[i, j] = conditionals[i, j] - marginals[j].
    marginals, conditionals = square_probs
    pmi = PointwiseMutualInfo(marginals, conditionals, symmetric=False, normalize=False)
    assert np.allclose(pmi.pmi_mat, conditionals - marginals)


def test_rectangular_raw_pmi_is_the_relevance_matrix(rect_probs):
    marginals, conditionals = rect_probs
    pmi = PointwiseMutualInfo(marginals, conditionals, normalize=False)
    assert pmi.pmi_mat.shape == (2, 3)
    assert np.allclose(pmi.pmi_mat, conditionals - marginals)


def test_rectangular_normalize_raises(rect_probs):
    with pytest.raises(ValueError):
        _ = PointwiseMutualInfo(*rect_probs, normalize=True).pmi_mat


def test_save_then_from_npy_reconstructs_every_matrix(tmp_path, square_probs):
    pmi = PointwiseMutualInfo(*square_probs, symmetric=True, normalize=True)
    path = str(tmp_path / "pmi.npy")
    pmi.save(path)
    loaded = PointwiseMutualInfo.from_npy(path, symmetric=True, normalize=True)
    for attr in ("joint_mat", "pmi_mat", "marginal", "cond_mat"):
        assert np.allclose(getattr(pmi, attr), getattr(loaded, attr))


def test_from_shards_reassembles_the_grid(tmp_path):
    torch = pytest.importorskip("torch")

    nrows, ncols = 2, 3
    flat = torch.arange(nrows * ncols + ncols, dtype=torch.float32) * -1.0
    # One rank, two batches, deliberately out of order.
    first, second = torch.tensor([4, 5, 6, 7, 8]), torch.tensor([0, 1, 2, 3])
    torch.save([[first, second]], tmp_path / "batch_indices_0.pt")
    torch.save([flat[first], flat[second]], tmp_path / "predictions_0.pt")

    pmi = PointwiseMutualInfo.from_shards(str(tmp_path), nrows, ncols, symmetric=False)
    assert np.allclose(pmi.pmi_mat, np.array([[-3.0, -4.0, -5.0], [-6.0, -7.0, -8.0]])
                       - np.array([0.0, -1.0, -2.0]))
