"""The prompt registry: every string KCluster sends to a model, in one place.

These strings are published artifacts — the EDM 2025 paper (Tables 1, 3, 4)
and the LAK 2026 paper quote them verbatim — so they are versioned: any
change must bump ``PROMPT_VERSION`` and is a scientific decision, not a
wording cleanup. Golden tests (tests/test_prompts.py and the task tests)
pin the exact strings.

Templates are ``str.format`` style. Composition helpers that need a
Question take it duck-typed (``header``/``q_type``/``__str__``) to keep
this module import-free. The human-facing survey wording lives with the
survey writer in kcluster/output/qualtrics.py, not here.
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

# Rendering variants for the format-leakage probe. The scored question's own
# type line sits in both the marginal and the conditional context, so it
# cancels out of the PMI difference; the CONTEXT question's type line and
# rendering do not, and a format the two questions share makes the scored
# question's own scaffolding cheaper. That is congruity the pair earns by
# looking alike rather than by testing the same thing. Each variant removes one
# channel so the size of the effect can be measured:
#
#   v1         the published EDM 2025 grid — the default everywhere
#   no_type    drops the "<type>:" line (the explicit channel)
#   no_answer  ... and the "Answer: <key>" trailer, whose predictability under
#              a same-format context varies sharply by format
#
# These are measurement instruments, not a new default: adopting one is a
# PROMPT_VERSION bump and a scientific decision.


def _header_v1(question, q_num: int) -> str:
    return question.header(q_num)


def _header_bare(question, q_num: int) -> str:
    return EXERCISE_HEADER.format(q_num=q_num)


def _text_v1(question) -> str:
    return str(question)  # body + "Answer: <key>"


def _text_no_answer(question) -> str:
    return question.body  # stem (+ choices), no answer trailer


# name -> (how to render a header, how to render a question's text). The text
# renderer applies to the context question and the scored question alike:
# stripping the answer from only one of them would leave the cue in place.
CONGRUITY_RENDERERS = {
    "v1": (_header_v1, _text_v1),
    "no_type": (_header_bare, _text_v1),
    "no_answer": (_header_bare, _text_no_answer),
}


def _renderer(render: str):
    if render not in CONGRUITY_RENDERERS:
        raise ValueError(f"unknown congruity renderer {render!r}; "
                         f"expected one of {sorted(CONGRUITY_RENDERERS)}")
    return CONGRUITY_RENDERERS[render]


def congruity_marginal_context(question, render: str = "v1") -> str:
    header, _ = _renderer(render)
    return f"{header(question, 2)}\n"


def congruity_pair_context(context_question, scored_question, render: str = "v1") -> str:
    header, text = _renderer(render)
    return f"{header(context_question, 1)}\n{text(context_question)}\n\n{header(scored_question, 2)}\n"


def congruity_scored_text(question, render: str = "v1") -> str:
    """The text whose log-probability is scored under those contexts."""
    _, text = _renderer(render)
    return text(question)


# --- Concept extraction (EDM 2025 Table 1) ---
CONCEPT_TRAILER_NOUN = "whether the student understands the concept of"
CONCEPT_TRAILER_VERBAL = "whether the student can"
CONCEPT_PROMPT = "{header}\n{question}\n\nRemark:\nThe above exercise is a {q_type} question that tests {trailer}"


def concept_prompt(question, verbal: bool = False) -> str:
    trailer = CONCEPT_TRAILER_VERBAL if verbal else CONCEPT_TRAILER_NOUN
    q_type = question.q_type.lower().replace(SPACE, "-")
    return CONCEPT_PROMPT.format(header=question.header(1), question=question, q_type=q_type, trailer=trailer)


# --- LO alignment (LAK 2026; tasks/classify.py). The marginal is the bare
# type line — deliberately NOT the congruity grid's "Exercise 2:" header ---
LO_ALIGNMENT_MARGINAL_CONTEXT = "{q_type}:\n"
LO_ACTIONS_HEADER = "The exercise below is designed to test whether a student can {lo}."
LO_FACTS_HEADER = "The exercise below is designed to test whether a student knows:\n{lo}."

# --- MCQ generation (LAK 2026; tasks/qgen/generate.py) ---
QGEN_SEED_ACTIONS = "The exercises below are designed to test whether a student can {std}.\n\n{header}"
QGEN_SEED_FACTS = ('The exercises below are designed to test whether a student understands the following facts:\n'
                   '"{std}."\n\n{header}')
QGEN_MCQ_HEADER = "Multiple Choice (best out of {num_choices} options):\n1."
QGEN_SOLUTION_PREFIX = "\n\nSolution:\nThe correct answer is"
QGEN_EXPLANATION_PREFIX = "\n\nExplanation:\n"

# --- LLM judges (LAK 2026 study arms; papers/lak2026/scripts/) ---
GPT_JUDGE_SYSTEM_PROMPT = (
    "You are an expert at answering multiple choice questions. "
    "If none of the options a-d are correct, choose e for 'None of the above'. "
    "Provide your answer (letter a-e) and explanation in the JSON format specified."
)
JUDGE_Q1_PAIRED = "Answer the following two questions:\n\nQ1. {question}"
JUDGE_Q1_SINGLE = "Answer the following question:\n\n{question}"
JUDGE_Q2_LOGPROB = "Does the above question help teachers test whether a student can {lo}?\na) Yes\nb) No"
JUDGE_Q2_TEXT = "Does the following question test whether a student can **{lo}**?\n\n{question}"
JUDGE_PREFILL_Q1 = "The answer to Q1 is **"
JUDGE_PREFILL_Q2 = "The answer to Q2 is **"
