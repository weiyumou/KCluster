"""Golden tests for concept extraction.

The prompt assertions pin the exact string the code builds (the EDM 2025
paper's Table 1 shows the same scaffold; note the code renders the answer
line as "Answer: <label>", while the paper's illustration spells out the
choice text).
"""

import pytest

pytest.importorskip("torch")

from kcluster.core.question import Question  # noqa: E402
from kcluster.tasks.cluster import build_res_df  # noqa: E402
from kcluster.tasks.concept import extract_concepts, extract_question_embeds  # noqa: E402


class StubLLM:
    """Records prompts/kwargs and returns canned completions/embeddings."""

    def __init__(self, completion: str = " flexibility. "):
        self.completion = completion
        self.complete_calls: list[tuple[list[str], dict]] = []
        self.encode_calls: list[tuple[list[str], list[str]]] = []

    def complete_prompts(self, prompts, **kwargs):
        self.complete_calls.append((list(prompts), kwargs))
        return [self.completion] * len(prompts)

    def encode(self, texts, contexts, **kwargs):
        import torch

        self.encode_calls.append((list(texts), list(contexts)))
        return torch.zeros(len(texts), 4)


@pytest.fixture
def mcq() -> Question:
    return Question(
        {
            "id": "sciqa-0",
            "type": "Multiple Choice",
            "question": {
                "stem": "Which is the most flexible?",
                "choices": [
                    {"label": "a", "text": "paper"},
                    {"label": "b", "text": "ceramic tea cup"},
                    {"label": "c", "text": "clay tile"},
                ],
            },
            "answerKey": "a",
            "images": ["fig.png"],
        }
    )


def test_concept_prompt_matches_the_paper_template(mcq):
    llm = StubLLM()
    extract_concepts(llm, [mcq], batch_size=1)
    [(prompts, kwargs)] = llm.complete_calls
    assert prompts == [
        "Exercise 1:\n"
        "Multiple Choice:\n"
        "Which is the most flexible?\n"
        "a) paper\n"
        "b) ceramic tea cup\n"
        "c) clay tile\n"
        "Answer: a\n"
        "\n"
        "Remark:\n"
        "The above exercise is a multiple-choice question "
        "that tests whether the student understands the concept of"
    ]
    assert kwargs["stop_tokens"] == [".", ","]


def test_verbal_flag_switches_the_trailer(mcq):
    llm = StubLLM()
    extract_concepts(llm, [mcq], batch_size=1, verbal=True)
    [(prompts, _)] = llm.complete_calls
    assert prompts[0].endswith("that tests whether the student can")


def test_concepts_are_stripped_of_trailing_punctuation(mcq):
    assert extract_concepts(StubLLM(" flexibility. "), [mcq], batch_size=1) == ["flexibility"]


def test_extraction_batches_the_questions(mcq):
    llm = StubLLM()
    concepts = extract_concepts(llm, [mcq, mcq, mcq], batch_size=2)
    assert len(concepts) == 3
    assert [len(prompts) for prompts, _ in llm.complete_calls] == [2, 1]


def test_question_embeds_use_the_marginal_context(mcq):
    llm = StubLLM()
    embeds = extract_question_embeds(llm, [mcq, mcq], batch_size=1)
    assert embeds.shape == (2, 4)
    texts, contexts = llm.encode_calls[0]
    assert contexts == ["Exercise 2:\nMultiple Choice:\n"]
    assert texts == [str(mcq)]


def test_build_res_df_adds_kc_and_drops_images(mcq):
    df = build_res_df([mcq], ["flexibility"])
    assert df.loc[0, "KC"] == "flexibility"
    assert "images" not in df.columns
    assert df.loc[0, "question"] == str(mcq)
