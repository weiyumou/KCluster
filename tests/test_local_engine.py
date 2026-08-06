"""Tests for the local engine substrate.

Everything needing a real model is exercised later with a tiny fixture model;
here we cover the pure helpers and that the module imports cleanly whenever
torch is available (the [local] extra).
"""

import pytest

pytest.importorskip("torch")

from kcluster.engine.local import LargeLangModel, batched  # noqa: E402


def test_batched_splits_with_remainder():
    assert list(batched("ABCDEFG", 3)) == [("A", "B", "C"), ("D", "E", "F"), ("G",)]


def test_batched_exact_multiple():
    assert list(batched([1, 2, 3, 4], 2)) == [(1, 2), (3, 4)]


def test_batched_empty_iterable_yields_nothing():
    assert list(batched([], 5)) == []


def test_batched_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        list(batched("ABC", 0))


def test_model_wrapper_exposes_the_engine_surface():
    # The method set below is the de-facto engine interface: the Vertex
    # serving container dispatches jobs to these same names by string.
    for method in ("next_logits", "next_tokens", "complete_prompts", "log_prob", "encode"):
        assert callable(getattr(LargeLangModel, method))
