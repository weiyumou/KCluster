"""The prompt registry: every string KCluster sends to a model, in one place.

These strings are published artifacts — the EDM 2025 paper (Tables 1, 3, 4)
quotes them verbatim — so they are versioned: any
change must bump ``PROMPT_VERSION`` and is a scientific decision, not a
wording cleanup. Golden tests (tests/test_prompts.py and the task tests)
pin the exact strings.

Templates are ``str.format`` style. Composition helpers that need a
Question take it duck-typed (``header``/``q_type``/``__str__``) to keep
this module import-free.

What fixes a prompt's *reading* sits beside it, because both engines must
agree on it as exactly as on the string: where a concept completion stops and
how it is cleaned. A style owns both halves of the reading, since a model
written to in another form answers in another form: an instruction-tuned
model ends a name at its end of turn and marks it up as it would a reply.

How the prompts are *written* is a prompt style (``PromptStyle``): the
paper's plain text, which a base model completes (``StockPrompts``, the
default everywhere), or the same requests as a conversation, which the
engine writes in a model's chat template, for an instruction-tuned model
(``ChatPrompts``). Every task that builds prompts takes a style and defaults
to ``STOCK``, so the published strings are what runs unless a caller says
otherwise.
"""

from typing import Protocol

PROMPT_VERSION = 1

SPACE = chr(32)

# --- Question rendering (EDM 2025; core/question.py) ---
EXERCISE_HEADER = "Exercise {q_num}:"
QUESTION_TYPE_LINE = "{q_type}:"
CHOICE_LINE = "{label}) {text}"
ANSWER_TRAILER = "Answer:"

# Where a question shows a figure, its text carries this marker; the
# question's ``figures`` list names the files, in marker order. A text-only
# engine reads the marker as these eight characters, exactly as the banks
# render it; to show the figure, a task puts the model's image token in its
# place (``show_figures``) and gives the file beside the prompt, as the engine
# takes images (``kcluster.engine.vision``). It is a data literal, not a prompt
# template: the text around it is unchanged, so it needs no PROMPT_VERSION bump.
FIGURE_MARKER = "[figure]"

# --- Question congruity (EDM 2025 Tables 3 and 4; decision D2: the
# marginal scores under the "Exercise 2:" header) ---


def congruity_marginal_context(question) -> str:
    return f"{question.header(2)}\n"


def congruity_pair_context(context_question, scored_question) -> str:
    return f"{context_question.header(1)}\n{context_question}\n\n{scored_question.header(2)}\n"


# --- Concept extraction (EDM 2025 Table 1) ---
CONCEPT_TRAILER_NOUN = "whether the student understands the concept of"
CONCEPT_TRAILER_VERBAL = "whether the student can"
# The remark is the sentence a concept name completes; the concept prompt
# puts it after the exercise, and the marginal prompt after no exercise. A
# question without a type gets the remark without its type clause, as its
# header goes without the type line (no published bank is typeless).
CONCEPT_REMARK = "The above exercise is a {q_type} question that tests {trailer}"
CONCEPT_REMARK_UNTYPED = "The above exercise tests {trailer}"
CONCEPT_PROMPT = "{header}\n{question}\n\nRemark:\n{remark}"
CONCEPT_MARGINAL_PROMPT = "{header}\n\nRemark:\n{remark}"


# The concept is the completion up to its first period or comma, found by beam
# search under a negative length penalty that keeps it short.
CONCEPT_STOP_TOKENS = (".", ",")
CONCEPT_NUM_BEAMS = 5
CONCEPT_LENGTH_PENALTY = -0.1


def concept_remark(question, verbal: bool = False, remark: str | None = None) -> str:
    """The remark for a question: the template with its type and the noun or verbal trailer filled in.

    ``remark`` is the published one unless given: ``CONCEPT_REMARK``, or ``CONCEPT_REMARK_UNTYPED`` for a question
    without a type. A remark given is used as written, for any question.
    """
    trailer = CONCEPT_TRAILER_VERBAL if verbal else CONCEPT_TRAILER_NOUN
    q_type = question.q_type.lower().replace(SPACE, "-")
    if remark is None:
        remark = CONCEPT_REMARK if q_type else CONCEPT_REMARK_UNTYPED
    return remark.format(q_type=q_type, trailer=trailer)


def concept_prompt(question, verbal: bool = False) -> str:
    return STOCK.concept(question, verbal=verbal)


def clean_concept(completion: str) -> str:
    """The concept a completion names: the text before its first stop token or line break after any leading
    whitespace, its whitespace collapsed to single spaces."""
    completion = completion.lstrip()
    for stop in (*CONCEPT_STOP_TOKENS, "\n"):
        completion = completion.split(stop, 1)[0]
    return SPACE.join(completion.split())


# The marks an instruction-tuned model puts around a name, none of which is ever part of one: emphasis (bold or
# italic, possibly left open where the completion stopped first), code, and quotation. Underscores stay, as a
# name in a mathematics bank may carry one.
CHAT_MARKUP = str.maketrans("", "", "*`\"\u201c\u201d")


def clean_chat_concept(completion: str) -> str:
    """``clean_concept`` of a completion with its markup (``CHAT_MARKUP``) taken out first."""
    return clean_concept(completion.translate(CHAT_MARKUP))


# --- Prompt styles: how the requests above are written for a model ---


#: What a model reads: text, or a conversation as messages (``kcluster.engine``).
Prompt = str | list[dict]


def show_figures(prompt: Prompt, image_token: str) -> Prompt:
    """The prompt with each figure marker replaced by ``image_token``, where the engine places an image: in a
    string, or in the content of each message of a conversation. The figures themselves go beside the prompt."""
    if isinstance(prompt, str):
        return prompt.replace(FIGURE_MARKER, image_token)
    return [{**m, "content": m["content"].replace(FIGURE_MARKER, image_token)}
            if isinstance(m.get("content"), str) else m for m in prompt]


class PromptStyle(Protocol):
    """The prompts of the two scoring tasks, written one way.

    A prompt is text for the model to complete, or a conversation as a list
    of messages, which the engine writes in the model's chat template.

    ``concept`` ends where a concept name of the question starts, and
    ``concept_marginal`` is the same prompt with the question's text left
    out, so a name can be scored after no question. ``pair`` is the context
    a question is scored after when another question comes first, and
    ``marginal`` the context it is scored after on its own; the scored text
    is ``str(question)`` in both. ``concept_stop_tokens`` are where a
    generated name ends, and ``clean_concept`` reads the name out of the
    completion.
    """

    concept_stop_tokens: tuple[str, ...]

    def clean_concept(self, completion: str) -> str: ...

    def concept(self, question, verbal: bool = False) -> Prompt: ...

    def concept_marginal(self, question, verbal: bool = False) -> Prompt: ...

    def pair(self, context_question, scored_question) -> Prompt: ...

    def marginal(self, scored_question) -> Prompt: ...


class StockPrompts:
    """The paper's prompts: plain text for a base model to complete.

    ``remark`` is the sentence a concept name completes, a template over
    ``{q_type}`` and ``{trailer}``; the default is the published one, in its
    typed or untyped form by the question (``concept_remark``).
    """

    concept_stop_tokens = CONCEPT_STOP_TOKENS
    clean_concept = staticmethod(clean_concept)

    def __init__(self, remark: str | None = None):
        self.remark = remark

    def concept(self, question, verbal: bool = False) -> str:
        remark = concept_remark(question, verbal, self.remark)
        return CONCEPT_PROMPT.format(header=question.header(1), question=question, remark=remark)

    def concept_marginal(self, question, verbal: bool = False) -> str:
        remark = concept_remark(question, verbal, self.remark)
        return CONCEPT_MARGINAL_PROMPT.format(header=question.header(1), remark=remark)

    def pair(self, context_question, scored_question) -> str:
        return congruity_pair_context(context_question, scored_question)

    def marginal(self, scored_question) -> str:
        return congruity_marginal_context(scored_question)


STOCK = StockPrompts()


class ChatPrompts:
    """The same requests as a conversation, for an instruction-tuned model.

    The prompts are conversations, lists of ``{"role", "content"}`` messages,
    which the engine writes in the model's own chat template with the
    model's turn opened at the end (``kcluster.engine``); a last message in
    the assistant's role is the prefill the model continues. The requests are
    the user's words, and are the caller's to choose:

        concept           user: the exercise, then ``concept_request``;
                          assistant: the remark (``remark``, as in ``StockPrompts``), which the name completes
        concept marginal  the same without the exercise
        pair              user: ``first_request``; assistant: the context question; user: ``second_request``;
                          and the scored question is what the assistant says next
        marginal          user: ``first_request``; the scored question is the assistant's reply

    ``render``, if given, writes the prompts as text here instead: a function
    of the messages before the prefill that returns the text with the model's
    turn opened, after which the prefill is written. It is for a format the
    model's template does not write; a served model renders messages with
    its own template, so a client needs none.

    ``end_of_turn`` is the token that ends the model's turn (Gemma 4's
    ``<turn|>``, Phi-4-mini's ``<|end|>``), for a model whose generation does
    not already stop there: a name ends there as well as at a period or comma.
    A name is read with the markup such a model puts around it taken out
    (``clean_chat_concept``: bold, quotation marks).
    """

    clean_concept = staticmethod(clean_chat_concept)

    def __init__(self, render=None, *, concept_request: str, first_request: str, second_request: str,
                 remark: str | None = None, end_of_turn: str | None = None):
        self.render = render
        self.concept_request, self.first_request, self.second_request = concept_request, first_request, second_request
        self.remark = remark
        self.concept_stop_tokens = (*CONCEPT_STOP_TOKENS, *([end_of_turn] if end_of_turn else []))

    def _write(self, messages: list[dict], prefill: str = ""):
        """The prompt for a conversation: the messages, the prefill last in the assistant's role, or the text
        ``render`` writes for them."""
        if self.render is None:
            return [*messages, *([{"role": "assistant", "content": prefill}] if prefill else [])]
        return self.render(messages) + prefill

    def concept(self, question, verbal: bool = False):
        user = f"{question.header(1)}\n{question}\n\n{self.concept_request}"
        return self._write([{"role": "user", "content": user}], concept_remark(question, verbal, self.remark))

    def concept_marginal(self, question, verbal: bool = False):
        user = f"{question.header(1)}\n\n{self.concept_request}"
        return self._write([{"role": "user", "content": user}], concept_remark(question, verbal, self.remark))

    def pair(self, context_question, scored_question):
        return self._write([{"role": "user", "content": self.first_request},
                            {"role": "assistant", "content": str(context_question)},
                            {"role": "user", "content": self.second_request}])

    def marginal(self, scored_question):
        return self._write([{"role": "user", "content": self.first_request}])
