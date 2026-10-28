"""The scoring grid: one scoring job as a flat list of (context, text) pairs.

A grid scores ``m`` texts, its columns, after each of ``n`` contexts, its
rows, and each text once more after its *marginal* context, the one it is
scored after on its own. Flattened, the first ``m`` items are the marginals
and item ``m + i*m + j`` is text ``j`` after context ``(i, j)``: row-major,
with the row as the conditioning variable, which is the orientation of the
``PointwiseMutualInfo`` matrices reassembled from the scores
(``PointwiseMutualInfo.from_shards(shard_dir, *grid.shape)``). The grid may
be rectangular: the rows need not be the texts (learning objectives and
questions, questions and concept names).

A context is a prompt as the engine takes it (``kcluster.engine``): the text
the model reads, or a conversation as a list of ``{"role", "content"}``
messages that the engine writes in the model's chat template. The contexts
are given as functions of the cell, or by a subclass that overrides
``context`` and ``marginal`` (``kcluster.tasks.congruity.PairQuestion`` is the
question-congruity grid). The grid is torch-free: a map-style dataset for the
local scorer's DataLoader, and a sequence the Vertex engine indexes directly.
"""

from collections.abc import Callable, Sequence

#: What a model reads before a text: the text itself, or a conversation as messages.
Prompt = str | list[dict]


class ScoringGrid:
    def __init__(self, texts: Sequence[str], n_rows: int, *,
                 context: Callable[[int, int], Prompt] | None = None,
                 marginal: Callable[[int], Prompt] | Prompt | None = None):
        """``context(row, col)`` is the context text ``col`` is scored after in row ``row``, and ``marginal(col)``
        the one it is scored after on its own; ``marginal`` may be one prompt for every column. A subclass may
        override the two methods instead."""
        self.texts = list(texts)
        self.n_rows = n_rows
        self._context = context
        self._marginal = marginal if callable(marginal) or marginal is None else (lambda col: marginal)

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, columns): the shape of the conditional matrix the scores fill."""
        return self.n_rows, len(self.texts)

    def marginal(self, col: int) -> Prompt:
        """The context text ``col`` is scored after on its own."""
        if self._marginal is None:
            raise NotImplementedError("the grid has no marginal contexts: pass marginal= or override marginal()")
        return self._marginal(col)

    def context(self, row: int, col: int) -> Prompt:
        """The context text ``col`` is scored after in row ``row``."""
        if self._context is None:
            raise NotImplementedError("the grid has no contexts: pass context= or override context()")
        return self._context(row, col)

    def __getitem__(self, index: int) -> tuple[Prompt, str]:
        m = len(self.texts)
        if index < m:
            return self.marginal(index), self.texts[index]
        row, col = divmod(index - m, m)
        if row >= self.n_rows:
            raise IndexError(index)
        return self.context(row, col), self.texts[col]

    def __len__(self) -> int:
        # Conditional-prob matrix + marginal-prob vector
        return (self.n_rows + 1) * len(self.texts)

    def longest_input(self, count: Callable[[Prompt, str], int]) -> int:
        """Tokens in the longest pair the grid holds, for a check against the model's context length: the longest
        text after the row whose context is longest in the first column (contexts vary little along a row).
        ``count(context, text)`` is the engine's token count of a pair (``kcluster.engine.local.token_count``)."""
        row = max(range(self.n_rows), key=lambda i: count(self.context(i, 0), ""))
        col = max(range(len(self.texts)), key=lambda j: count("", self.texts[j]))
        return count(self.context(row, col), self.texts[col])
