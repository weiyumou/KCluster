"""The question-congruity scoring grid (EDM 2025).

``PairQuestion`` flattens one congruity job into a linear index space of
(context, text) scoring pairs: the first ``n`` items are the marginals — the
scored question under a bare ``Exercise 2:`` header — and item ``n + i*n + j``
scores question ``j`` conditioned on question ``i``. The grid is row-major
with the row as the conditioning variable, matching the orientation of the
PointwiseMutualInfo matrices reassembled from the scores.

Each item also carries the figures its two sides show, as file paths: the
conditioning question's in the context, the scored question's in the text.
A figure is context in the marginal and in the conditional alike, so the
PMI subtracts it out; what is scored is the text either way.
"""

from kcluster.core.prompts import congruity_marginal_context, congruity_pair_context
from kcluster.core.question import Question
from kcluster.io.figures import figure_paths


class PairQuestion:
    # A torch-free map-style dataset: DataLoader only needs
    # __getitem__/__len__, and the Vertex engine indexes it directly.
    def __init__(self, questions: list[Question], figure_dir: str | None = None):
        """``figure_dir``: the root the questions' figure paths resolve against; None shows no figures
        (their markers reach the model as literal text)."""
        self.questions = questions
        self.figure_dir = figure_dir

    def _figures(self, q: Question) -> list[str]:
        return figure_paths(q, self.figure_dir) if self.figure_dir else []

    def __getitem__(self, index):
        n = len(self.questions)
        if index < n:
            q = self.questions[index]
            return congruity_marginal_context(q), str(q), [], self._figures(q)
        row, col = (index - n) // n, (index - n) % n
        q1, q2 = self.questions[row], self.questions[col]
        return congruity_pair_context(q1, q2), str(q2), self._figures(q1), self._figures(q2)

    def __len__(self):
        # Conditional-prob matrix + marginal-prob vector
        return len(self.questions) ** 2 + len(self.questions)
