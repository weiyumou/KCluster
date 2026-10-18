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
"""

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
CONCEPT_PROMPT = "{header}\n{question}\n\nRemark:\nThe above exercise is a {q_type} question that tests {trailer}"


# The concept is the completion up to its first period or comma, found by beam
# search under a negative length penalty that keeps it short.
CONCEPT_STOP_TOKENS = (".", ",")
CONCEPT_NUM_BEAMS = 5
CONCEPT_LENGTH_PENALTY = -0.1


def concept_prompt(question, verbal: bool = False) -> str:
    trailer = CONCEPT_TRAILER_VERBAL if verbal else CONCEPT_TRAILER_NOUN
    q_type = question.q_type.lower().replace(SPACE, "-")
    return CONCEPT_PROMPT.format(header=question.header(1), question=question, q_type=q_type, trailer=trailer)


def clean_concept(completion: str) -> str:
    """The concept a completion names: without surrounding whitespace or the stop token."""
    return completion.strip().rstrip("".join(CONCEPT_STOP_TOKENS))
