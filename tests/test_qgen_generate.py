"""Golden tests for incremental MCQ generation, driven by a stub LLM.

The stub returns canned completions per step, so these pin the prompt
scaffolds, the stem/choice cleanup regexes, the answer wiring, and the
final assembled Question — no model weights involved.
"""

import hashlib

import pytest

pytest.importorskip("torch")

from kcluster.tasks.qgen.generate import (  # noqa: E402
    create_seed_prompts,
    generate_mcq,
    generate_mcq_from_std,
    read_standards,
)


class StubTokenizer:
    def encode(self, s):
        return [99]


class StubLLM:
    tokenizer = StubTokenizer()

    def __init__(self):
        self.calls = []

    def complete_prompts(self, prompts, **kwargs):
        self.calls.append((list(prompts), kwargs))
        out = []
        for p in prompts:
            if p.endswith("a)"):
                out.append(" paris")  # cleaned by the fallback first-line regex
            elif p.endswith("b)"):
                out.append(" london. c) junk")  # cleaned by the next-label regex
            elif p.endswith("Explanation:\n"):
                out.append("Because London is bigger.\n\nnoise")
            else:  # stem
                out.append(" What is the capital of France?\nnoise")
        return out

    def next_tokens(self, prompts, choices):
        self.calls.append(("next_tokens", list(prompts), list(choices)))
        return [(" b",) for _ in prompts]


def test_generate_mcq_assembles_a_question():
    llm = StubLLM()
    [q], [prompt] = (lambda t: (t[0], t[1]))(generate_mcq(llm, ["SEED\n1."], num_choices=2))

    assert q["question"]["stem"] == "What is the capital of France?"
    assert q["question"]["choices"] == [
        {"label": "a", "text": "Paris"},
        {"label": "b", "text": "London"},
    ]
    assert q["answerKey"] == "b"
    assert q["explanation"] == "Because London is bigger."
    # The id is a content hash of the rendered question
    assert q["id"] == hashlib.md5(str(q).encode("utf-8")).hexdigest()

    # The running prompt accumulates every step
    assert prompt == (
        "SEED\n1. What is the capital of France?"
        "\na) Paris"
        "\nb) London"
        "\n\nSolution:\nThe correct answer is b) London."
        "\n\nExplanation:\nBecause London is bigger."
    )

    # Generation always stops at line breaks and never starts with one
    stem_call_kwargs = llm.calls[0][1]
    assert stem_call_kwargs["stop_tokens"] == ["\n"]
    assert stem_call_kwargs["begin_suppress_tokens"] == [99]
    # The answer step restricts next tokens to the label tokens
    assert ("next_tokens" == llm.calls[3][0]) and (llm.calls[3][2] == [" a", " b"])
    # The explanation step stops at a blank line
    assert llm.calls[4][1]["stop_strings"] == ["\n\n"]


def test_create_seed_prompts():
    [actions] = create_seed_prompts(["explain how levers work"], "actions", "HDR\n1.")
    assert actions == ("The exercises below are designed to test whether a student can "
                       "explain how levers work.\n\nHDR\n1.")

    [facts] = create_seed_prompts(["Water boils at 100 C"], "facts", "HDR\n1.")
    assert facts == ("The exercises below are designed to test whether a student understands "
                     'the following facts:\n"Water boils at 100 C."\n\nHDR\n1.')

    with pytest.raises(ValueError, match="std_type"):
        create_seed_prompts(["x"], "verbs", "HDR")


def test_read_standards_normalizes_like_classify(tmp_path):
    path = tmp_path / "standards.txt"
    path.write_text("Explain how levers work.\n\nIdentify the states of matter.\n")

    assert read_standards(str(path), "actions") == [
        "explain how levers work", "identify the states of matter"]
    assert read_standards(str(path), "facts") == [
        "Explain how levers work", "Identify the states of matter"]


def test_generate_mcq_from_std_assigns_los():
    llm = StubLLM()
    questions, prompts = generate_mcq_from_std(
        llm, ["standard one", "standard two"], "actions",
        stds_per_batch=2, qs_per_std=1, configs={"choice": {}}, num_choices=2)

    assert len(questions) == len(prompts) == 2
    assert [q["lo"] for q in questions] == ["standard one", "standard two"]
    # Every seed prompt embeds its standard and the shared MCQ header
    assert "whether a student can standard one" in prompts[0]
    assert "Multiple Choice (best out of 2 options):" in prompts[0]
