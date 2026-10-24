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
how it is cleaned.

How the prompts are *written* is a prompt style (``PromptStyle``): the
paper's plain text, which a base model completes (``StockPrompts``, the
default everywhere), or the same requests as a conversation in a model's
chat template, for an instruction-tuned model (``ChatPrompts``). Every
task that builds prompts takes a style and defaults to ``STOCK``, so the
published strings are what runs unless a caller says otherwise.
"""

from typing import Protocol

PROMPT_VERSION = 1

SPACE = chr(32)

# --- Question rendering (EDM 2025; core/question.py) ---
EXERCISE_HEADER = "Exercise {q_num}:"
QUESTION_TYPE_LINE = "{q_type}:"
CHOICE_LINE = "{label}) {text}"
ANSWER_TRAILER = "Answer:"

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
# puts it after the exercise, and the marginal prompt after no exercise.
CONCEPT_REMARK = "The above exercise is a {q_type} question that tests {trailer}"
CONCEPT_PROMPT = "{header}\n{question}\n\nRemark:\n{remark}"
CONCEPT_MARGINAL_PROMPT = "{header}\n\nRemark:\n{remark}"


# The concept is the completion up to its first period or comma, found by beam
# search under a negative length penalty that keeps it short.
CONCEPT_STOP_TOKENS = (".", ",")
CONCEPT_NUM_BEAMS = 5
CONCEPT_LENGTH_PENALTY = -0.1


def concept_remark(question, verbal: bool = False, remark: str = CONCEPT_REMARK) -> str:
    """The remark for a question: the template with its type and the noun or verbal trailer filled in."""
    trailer = CONCEPT_TRAILER_VERBAL if verbal else CONCEPT_TRAILER_NOUN
    q_type = question.q_type.lower().replace(SPACE, "-")
    return remark.format(q_type=q_type, trailer=trailer)


def concept_prompt(question, verbal: bool = False) -> str:
    return STOCK.concept(question, verbal=verbal)


def clean_concept(completion: str) -> str:
    """The concept a completion names: without surrounding whitespace or the stop token."""
    return completion.strip().rstrip("".join(CONCEPT_STOP_TOKENS))


# --- Prompt styles: how the requests above are written for a model ---


class PromptStyle(Protocol):
    """The prompts of the two scoring tasks, written one way.

    ``concept`` ends where a concept name of the question starts, and
    ``concept_marginal`` is the same prompt with the question's text left
    out, so a name can be scored after no question. ``pair`` is the context
    a question is scored after when another question comes first, and
    ``marginal`` the context it is scored after on its own; the scored text
    is ``str(question)`` in both.
    """

    def concept(self, question, verbal: bool = False) -> str: ...

    def concept_marginal(self, question, verbal: bool = False) -> str: ...

    def pair(self, context_question, scored_question) -> str: ...

    def marginal(self, scored_question) -> str: ...


class StockPrompts:
    """The paper's prompts: plain text for a base model to complete.

    ``remark`` is the sentence a concept name completes, a template over
    ``{q_type}`` and ``{trailer}``; the default is the published one.
    """

    def __init__(self, remark: str = CONCEPT_REMARK):
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
    """The same requests as a conversation in a model's chat template, for an instruction-tuned model.

    ``render`` writes a list of ``{"role", "content"}`` messages as the text
    the model reads, with the assistant's turn opened at the end; for a
    Hugging Face tokenizer that is
    ``partial(tokenizer.apply_chat_template, tokenize=False, add_generation_prompt=True)``.
    The requests are the user's words, and are the caller's to choose:

        concept           user: the exercise, then ``concept_request``;
                          assistant: the remark (``remark``, as in ``StockPrompts``), which the name completes
        concept marginal  the same without the exercise
        pair              user: ``first_request``; assistant: the context question; user: ``second_request``;
                          and the scored question is what the assistant says next
        marginal          user: ``first_request``; the scored question is the assistant's reply
    """

    def __init__(self, render, *, concept_request: str, first_request: str, second_request: str,
                 remark: str = CONCEPT_REMARK):
        self.render = render
        self.concept_request, self.first_request, self.second_request = concept_request, first_request, second_request
        self.remark = remark

    def concept(self, question, verbal: bool = False) -> str:
        user = f"{question.header(1)}\n{question}\n\n{self.concept_request}"
        return self.render([{"role": "user", "content": user}]) + concept_remark(question, verbal, self.remark)

    def concept_marginal(self, question, verbal: bool = False) -> str:
        user = f"{question.header(1)}\n\n{self.concept_request}"
        return self.render([{"role": "user", "content": user}]) + concept_remark(question, verbal, self.remark)

    def pair(self, context_question, scored_question) -> str:
        return self.render([{"role": "user", "content": self.first_request},
                            {"role": "assistant", "content": str(context_question)},
                            {"role": "user", "content": self.second_request}])

    def marginal(self, scored_question) -> str:
        return self.render([{"role": "user", "content": self.first_request}])
