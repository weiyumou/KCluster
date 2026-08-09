"""The question-congruity scoring grid (EDM 2025).

``PairQuestion`` flattens one congruity job into a linear index space of
(context, text) scoring pairs: the first ``n`` items are the marginals — the
scored question under a bare ``Exercise 2:`` header — and item ``n + i*n + j``
scores question ``j`` conditioned on question ``i``. The grid is row-major
with the row as the conditioning variable, matching the orientation of the
PointwiseMutualInfo matrices reassembled from the scores.
"""

from kcluster.core.prompts import congruity_marginal_context, congruity_pair_context, congruity_scored_text
from kcluster.core.question import Question


class PairQuestion:
    # A torch-free map-style dataset: DataLoader only needs
    # __getitem__/__len__, and the Vertex engine indexes it directly.
    def __init__(self, questions: list[Question], render: str = "v1"):
        # ``render`` selects a prompts.CONGRUITY_RENDERERS variant; "v1" is the
        # published grid. Anything else is a format-leakage probe, not a run
        # whose PMI values are comparable with the paper's.
        self.questions, self.render = questions, render

    def __getitem__(self, index):
        n = len(self.questions)
        if index < n:
            q = self.questions[index]
            return congruity_marginal_context(q, self.render), congruity_scored_text(q, self.render)
        row, col = (index - n) // n, (index - n) % n
        q1, q2 = self.questions[row], self.questions[col]
        return congruity_pair_context(q1, q2, self.render), congruity_scored_text(q2, self.render)

    def __len__(self):
        # Conditional-prob matrix + marginal-prob vector
        return len(self.questions) ** 2 + len(self.questions)
