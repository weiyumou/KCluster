"""The question-congruity scoring grid (EDM 2025).

``PairQuestion`` flattens one congruity job into a linear index space of
(context, text) scoring pairs: the first ``n`` items are the marginals — the
scored question under a bare ``Exercise 2:`` header — and item ``n + i*n + j``
scores question ``j`` conditioned on question ``i``. The grid is row-major
with the row as the conditioning variable, matching the orientation of the
PointwiseMutualInfo matrices reassembled from the scores.
"""

from kcluster.core.question import Question


class PairQuestion:
    # A torch-free map-style dataset: DataLoader only needs
    # __getitem__/__len__, and the Vertex engine indexes it directly.
    def __init__(self, questions: list[Question]):
        self.questions = questions

    def __getitem__(self, index):
        n = len(self.questions)
        if index < n:
            q = self.questions[index]
            return f"{q.header(2)}\n", str(q)
        row, col = (index - n) // n, (index - n) % n
        q1, q2 = self.questions[row], self.questions[col]
        return f"{q1.header(1)}\n{str(q1)}\n\n{q2.header(2)}\n", str(q2)

    def __len__(self):
        # Conditional-prob matrix + marginal-prob vector
        return len(self.questions) ** 2 + len(self.questions)
