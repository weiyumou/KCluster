"""The question-congruity scoring grid (EDM 2025).

``PairQuestion`` is the square case of ``ScoringGrid`` (``kcluster.core.grid``)
in which the questions are both the texts and the rows: the marginal context
is the scored question under a bare ``Exercise 2:`` header, and context
``(i, j)`` puts question ``i`` first. Its contexts are written by a prompt
style (``kcluster.core.prompts``): the paper's plain text unless the caller
passes another, such as a chat style for an instruction-tuned model, whose
contexts are conversations rather than text.
"""

from kcluster.core.grid import ScoringGrid
from kcluster.core.prompts import STOCK, PromptStyle
from kcluster.core.question import Question

__all__ = ["PairQuestion", "ScoringGrid"]


class PairQuestion(ScoringGrid):
    def __init__(self, questions: list[Question], prompts: PromptStyle = STOCK):
        super().__init__([str(q) for q in questions], len(questions))
        self.questions = questions
        self.prompts = prompts

    def marginal(self, col: int):
        return self.prompts.marginal(self.questions[col])

    def context(self, row: int, col: int):
        return self.prompts.pair(self.questions[row], self.questions[col])
