"""The question-congruity scoring grid (EDM 2025), and the layout it is a case of.

``ScoringGrid`` flattens one scoring job into a linear index space of
(context, text) pairs: the first ``m`` items are the marginals — each of the
``m`` texts after its own marginal context — and item ``m + i*m + j`` scores
text ``j`` after context ``(i, j)``, for ``n`` rows. The grid is row-major
with the row as the conditioning variable, matching the orientation of the
PointwiseMutualInfo matrices reassembled from the scores
(``PointwiseMutualInfo.from_shards(shard_dir, n, m)``). A subclass says what
the contexts are.

``PairQuestion`` is the congruity grid: the questions are the texts and the
rows, the marginal context is the scored question under a bare ``Exercise 2:``
header, and context ``(i, j)`` puts question ``i`` first. Its contexts are
written by a prompt style (``kcluster.core.prompts``): the paper's plain text
unless the caller passes another.
"""

from kcluster.core.prompts import STOCK, PromptStyle
from kcluster.core.question import Question


class ScoringGrid:
    # A torch-free map-style dataset: DataLoader only needs
    # __getitem__/__len__, and the Vertex engine indexes it directly.
    def __init__(self, texts: list[str], n_rows: int):
        self.texts = texts
        self.n_rows = n_rows

    def marginal(self, col: int) -> str:
        """The context text ``col`` is scored after on its own."""
        raise NotImplementedError

    def context(self, row: int, col: int) -> str:
        """The context text ``col`` is scored after in row ``row``."""
        raise NotImplementedError

    def __getitem__(self, index):
        m = len(self.texts)
        if index < m:
            return self.marginal(index), self.texts[index]
        row, col = divmod(index - m, m)
        return self.context(row, col), self.texts[col]

    def __len__(self):
        # Conditional-prob matrix + marginal-prob vector
        return (self.n_rows + 1) * len(self.texts)

    def longest_input(self, tokenizer) -> int:
        """Tokens in the longest pair the grid holds, for a check against the model's context length: the longest
        text after the row whose context is longest in the first column (contexts vary little along a row)."""
        def length(*texts):
            return len(tokenizer(*texts)["input_ids"])

        row = max(range(self.n_rows), key=lambda i: length(self.context(i, 0)))
        col = max(range(len(self.texts)), key=lambda j: length(self.texts[j]))
        return length(self.context(row, col), self.texts[col])


class PairQuestion(ScoringGrid):
    def __init__(self, questions: list[Question], prompts: PromptStyle = STOCK):
        super().__init__([str(q) for q in questions], len(questions))
        self.questions = questions
        self.prompts = prompts

    def marginal(self, col: int) -> str:
        return self.prompts.marginal(self.questions[col])

    def context(self, row: int, col: int) -> str:
        return self.prompts.pair(self.questions[row], self.questions[col])
