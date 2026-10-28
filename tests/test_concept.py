"""Golden tests for concept extraction.

The prompt assertions pin the exact string the code builds (the EDM 2025
paper's Table 1 shows the same scaffold; note the code renders the answer
line as "Answer: <label>", while the paper's illustration spells out the
choice text).
"""

import pytest

pytest.importorskip("torch")

from kcluster.core.question import Question
from kcluster.tasks.cluster import build_res_df
from kcluster.tasks.concept import complete_names, embed_texts, extract_concepts, extract_question_embeds


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


def test_several_names_per_question_come_back_grouped_and_cleaned(mcq):
    class SamplingLLM(StubLLM):
        def complete_prompts(self, prompts, **kwargs):
            self.complete_calls.append((list(prompts), kwargs))
            # num_return_sequences completions per prompt, in prompt order
            return [c for _ in prompts for c in (" flexibility. ", " 42.", "elasticity, of materials", "  ")]

    llm = SamplingLLM()
    names = extract_concepts(llm, [mcq, mcq], batch_size=2, num_names=4, do_sample=True, temperature=0.7)
    assert names == [["flexibility", "elasticity"], ["flexibility", "elasticity"]]  # no letter: dropped
    [(_, kwargs)] = llm.complete_calls
    assert kwargs["num_return_sequences"] == 4 and kwargs["do_sample"] is True

    with pytest.raises(ValueError, match="no name for question sciqa-0"):
        extract_concepts(StubLLM(" 3.14"), [mcq], batch_size=1, num_names=1)


def test_a_completion_count_other_than_asked_for_raises(mcq):
    # An engine that ignores num_return_sequences gives one completion per prompt; grouped num_names at a time,
    # the names would land on the wrong questions
    with pytest.raises(ValueError, match="2 completions for 2 prompts, not 3 per prompt"):
        extract_concepts(StubLLM(), [mcq, mcq], batch_size=2, num_names=3)


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


def test_a_prompt_style_writes_the_prompts_and_the_embedding_context(mcq):
    from kcluster.core.prompts import ChatPrompts

    style = ChatPrompts(concept_request="What does it test?", first_request="Write an exercise.",
                        second_request="Write another one.", end_of_turn="</assistant>")
    llm = StubLLM()
    extract_concepts(llm, [mcq], batch_size=1, prompts=style)
    [(prompts, kwargs)] = llm.complete_calls
    assert prompts == [style.concept(mcq)]  # a conversation, for the engine to write in the model's template
    assert kwargs["stop_tokens"] == [".", ",", "</assistant>"]  # a name also ends where the model ends its turn
    assert prompts[0][0] == {"role": "user", "content": f"{mcq.header(1)}\n{mcq}\n\nWhat does it test?"}
    assert prompts[0][1]["content"].startswith("The above exercise is a multiple-choice question that tests")
    # The style reads the name too: a chat model's bold is not part of it
    assert extract_concepts(StubLLM(" **flexibility**."), [mcq], batch_size=1, prompts=style) == ["flexibility"]
    assert extract_concepts(StubLLM(" **flexibility**."), [mcq], batch_size=1) == ["**flexibility**"]
    extract_question_embeds(llm, [mcq], batch_size=1, prompts=style)
    _, contexts = llm.encode_calls[0]
    assert contexts == [[{"role": "user", "content": "Write an exercise."}]]


def test_names_are_completed_for_any_prompts():
    """The step under extract_concepts: prompts of any making, their completions read as names."""
    llm = StubLLM(" the chain rule, applied")
    prompts = ["Name it:", [{"role": "user", "content": "Name it."}, {"role": "assistant", "content": "It is"}]]
    assert complete_names(llm, prompts, batch_size=1, stop_tokens=[".", "\n"], num_beams=2) == ["the chain rule"] * 2
    assert [p for p, _ in llm.complete_calls] == [[prompts[0]], [prompts[1]]]
    assert all(kw == {"stop_tokens": [".", "\n"], "num_beams": 2} for _, kw in llm.complete_calls)
    assert complete_names(llm, prompts, batch_size=2, clean=str.upper) == [" THE CHAIN RULE, APPLIED"] * 2

    with pytest.raises(ValueError, match="no name for prompt 0"):
        complete_names(StubLLM(" 42."), ["a", "b"], batch_size=2, num_names=1, ids=["prompt 0", "prompt 1"])


def test_texts_are_embedded_after_their_contexts():
    llm = StubLLM()
    embeds = embed_texts(llm, ["t0", "t1", "t2"], ["c0", [{"role": "user", "content": "c1"}], "c2"], batch_size=2)
    assert embeds.shape == (3, 4)
    assert llm.encode_calls == [(["t0", "t1"], ["c0", [{"role": "user", "content": "c1"}]]), (["t2"], ["c2"])]
