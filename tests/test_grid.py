"""Tests for the scoring grid: the flat layout the scorer and the matrix share, for a grid of any shape."""

import pytest

from kcluster.core.grid import ScoringGrid


@pytest.fixture
def grid():
    # Three questions in the rows, two names in the columns: the concept-scoring grid
    questions = ["q0", "q1", "q2"]
    return ScoringGrid([" flexibility.", " addition."], len(questions),
                       context=lambda row, col: f"{questions[row]}\n\nRemark:\nThe above exercise tests",
                       marginal="Remark:\nThe above exercise tests")


def test_the_first_items_are_the_marginals_then_the_rows(grid):
    assert grid.shape == (3, 2) and len(grid) == (3 + 1) * 2
    assert grid[0] == ("Remark:\nThe above exercise tests", " flexibility.")
    assert grid[1] == ("Remark:\nThe above exercise tests", " addition.")
    assert grid[2 + 2 * 2 + 0] == ("q2\n\nRemark:\nThe above exercise tests", " flexibility.")
    assert [grid[i][1] for i in range(len(grid))] == [" flexibility.", " addition."] * 4
    with pytest.raises(IndexError):
        grid[len(grid)]


def test_a_marginal_per_column_is_a_function_of_the_column():
    grid = ScoringGrid(["t0", "t1"], 1, context=lambda row, col: f"c{row}{col}", marginal=lambda col: f"m{col}")
    assert [grid[i] for i in range(len(grid))] == [("m0", "t0"), ("m1", "t1"), ("c00", "t0"), ("c01", "t1")]


def test_a_subclass_writes_its_own_contexts():
    class NameGrid(ScoringGrid):
        def marginal(self, col):
            return "no question"

        def context(self, row, col):
            return f"question {row}"

    grid = NameGrid(["a", "b"], 2)
    assert grid[1] == ("no question", "b") and grid[2 + 1 * 2 + 1] == ("question 1", "b")


def test_a_grid_without_contexts_says_so():
    with pytest.raises(NotImplementedError, match="marginal"):
        ScoringGrid(["a"], 1)[0]
    with pytest.raises(NotImplementedError, match="context"):
        ScoringGrid(["a"], 1, marginal="m")[1]


def test_contexts_may_be_conversations():
    messages = [{"role": "user", "content": "Write an exercise."}]
    grid = ScoringGrid(["t"], 1, context=lambda row, col: messages, marginal=messages)
    assert grid[0] == (messages, "t") and grid[1] == (messages, "t")


def test_longest_input_takes_the_longest_row_with_the_longest_text(grid):
    def count(context, text):
        return len(context) + len(text)

    assert grid.longest_input(count) == max(count(*grid[i]) for i in range(len(grid)))
