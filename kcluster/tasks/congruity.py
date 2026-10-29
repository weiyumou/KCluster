"""The question-congruity scoring grid (EDM 2025).

``PairQuestion`` is the square case of ``ScoringGrid`` (``kcluster.core.grid``)
in which the questions are both the texts and the rows: the marginal context
is the scored question under a bare ``Exercise 2:`` header, and context
``(i, j)`` puts question ``i`` first. Its contexts are written by a prompt
style (``kcluster.core.prompts``): the paper's plain text unless the caller
passes another, such as a chat style for an instruction-tuned model, whose
contexts are conversations rather than text.

With a ``figure_dir`` the grid shows the questions' figures to a model that
takes images: each ``[figure]`` marker becomes the model's image token
(``image_token``, the engine's), and every item carries the two sides' figure
files, the conditioning question's in the context and the scored question's
in the text. A figure is context in the marginal and in the conditional
alike, so the PMI subtracts it out; what is scored is the text either way.
Without a root (the default) the markers reach the model as literal text and
no figure is shown.
"""

from kcluster.core.grid import ScoringGrid
from kcluster.core.prompts import STOCK, PromptStyle, show_figures
from kcluster.core.question import Question
from kcluster.io.figures import figure_paths

__all__ = ["PairQuestion", "ScoringGrid"]


class PairQuestion(ScoringGrid):
    def __init__(self, questions: list[Question], prompts: PromptStyle = STOCK, figure_dir: str | None = None,
                 image_token: str | None = None):
        """``figure_dir`` is the root the questions' figure paths resolve against; with it the figures are shown,
        at ``image_token``, the model's (``llm.image_token``)."""
        if figure_dir is not None and not image_token:
            raise ValueError("showing figures needs the model's image token (llm.image_token); a model that takes "
                             "no images scores the bank with figure_dir=None, its markers as text")
        self.figure_dir, self.image_token = figure_dir, image_token
        super().__init__([self._show(str(q)) for q in questions], len(questions))
        self.questions = questions
        self.prompts = prompts
        self.shows_images = figure_dir is not None

    def _show(self, prompt):
        return show_figures(prompt, self.image_token) if self.figure_dir is not None else prompt

    def _figures(self, question: Question) -> list[str]:
        return figure_paths(question, self.figure_dir) if self.figure_dir is not None else []

    def marginal(self, col: int):
        return self._show(self.prompts.marginal(self.questions[col]))

    def context(self, row: int, col: int):
        return self._show(self.prompts.pair(self.questions[row], self.questions[col]))

    def images(self, col: int) -> list[str]:
        return self._figures(self.questions[col])

    def context_images(self, row: int, col: int) -> list[str]:
        return self._figures(self.questions[row])
