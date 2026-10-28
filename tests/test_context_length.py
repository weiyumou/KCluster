"""The context-length check before scoring: the longest pair in a grid against what the model reads."""

import pytest

pytest.importorskip("torch")

from types import SimpleNamespace

from kcluster.commands.pmi import check_context
from kcluster.core.grid import ScoringGrid
from kcluster.engine.local import context_length


@pytest.fixture
def grid():
    contexts = ["one two", "one two three four"]
    return ScoringGrid([" five", " five six"], len(contexts), context=lambda row, col: contexts[row] + "\n",
                       marginal="one\n")


def test_the_longest_pair_must_fit_the_context(grid, make_tokenizer):
    tokenizer = make_tokenizer(["one", "two", "three", "four", "five", "six"])
    assert check_context(grid, tokenizer, limit=6) == 6  # "one two three four\n five six"
    assert check_context(grid, tokenizer, limit=None) == 6
    with pytest.raises(SystemExit, match="6 tokens, over the model's context length of 5"):
        check_context(grid, tokenizer, limit=5)


def test_a_bos_counts_toward_the_limit(grid, make_tokenizer):
    tokenizer = make_tokenizer(["one", "two", "three", "four", "five", "six"], bos="<s>")
    assert check_context(grid, tokenizer, limit=7) == 7
    assert check_context(grid, tokenizer, limit=6, add_bos=False) == 6


def test_the_context_length_comes_from_the_config():
    assert context_length(SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048))) == 2048
    assert context_length(SimpleNamespace(n_positions=1024)) == 1024
    assert context_length(SimpleNamespace(config=SimpleNamespace())) is None
