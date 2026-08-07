"""Tests for MCQ validation: the completeness filters, the permutation-
averaged answer-confidence check, and the perplexity sort — all against
stub LLMs."""

import random

import pytest

torch = pytest.importorskip("torch")

from kcluster.core.question import Question  # noqa: E402
from kcluster.tasks.qgen.validate import shuffle_choices, sort_questions, validate_mcq  # noqa: E402


class StubTokenizer:
    def __call__(self, texts, **kwargs):
        # Token id = position: " a" -> 0, " b" -> 1, " c" -> 2, ...
        return {"input_ids": [[i] for i in range(len(texts))]}


class ConfidentLLM:
    """Always assigns ~all probability to the label whose text is `answer`."""

    tokenizer = StubTokenizer()
    device = torch.device("cpu")

    def __init__(self, answer="Paris", vocab=8):
        self.answer, self.vocab = answer, vocab
        self.prompts = []

    def next_logits(self, prompts):
        self.prompts.extend(prompts)
        logits = torch.zeros(len(prompts), self.vocab)
        for i, p in enumerate(prompts):
            for j in range(4):
                if f"{chr(ord('a') + j)}) {self.answer}" in p:
                    logits[i, j] = 12.0
        return logits


def _mcq(qid="q-0", stem="What is the capital of France?", choices=("Paris", "London"),
         answer="a", explanation="Because it just is, historically.", lo="geo"):
    return Question(
        {
            "id": qid,
            "type": "Multiple Choice",
            "question": {"stem": stem,
                         "choices": [{"label": chr(ord("a") + i), "text": t} for i, t in enumerate(choices)]},
            "answerKey": answer,
            "explanation": explanation,
            "lo": lo,
        }
    )


def test_validate_mcq_confirms_a_confident_answer():
    llm = ConfidentLLM(answer="Paris")
    q = _mcq(answer="b")  # a wrong stored key: validation re-keys it

    result = validate_mcq(llm, {"geo": [q]}, batch_size=4, prob_thd=0.9, num_choices=2, shuffle=False)

    [valid] = result["geo"]
    # The choices end up in the last scored permutation's order, and the
    # answer key points at the confident text within that ordering
    [ans_text] = [c["text"] for c in valid["question"]["choices"] if c["label"] == valid["answerKey"]]
    assert ans_text == "Paris"
    # Both permutations of the two choices were scored, with the appended
    # "None of the above" distractor visible to the model
    assert len(llm.prompts) == 2
    assert all("c) None of the above" in p for p in llm.prompts)


def test_validate_mcq_rejects_none_of_the_above_wins():
    llm = ConfidentLLM(answer="None of the above")
    result = validate_mcq(llm, {"geo": [_mcq()]}, batch_size=4, prob_thd=0.9, num_choices=2, shuffle=False)
    assert result == {}


class ExplodingLLM:
    tokenizer = StubTokenizer()
    device = torch.device("cpu")

    def next_logits(self, prompts):
        raise AssertionError("incomplete questions must be filtered before scoring")


@pytest.mark.parametrize("q", [
    _mcq(stem="2+2?"),                                        # trivial stem
    _mcq(choices=("Paris", "Paris")),                         # duplicate choices
    _mcq(choices=("Paris", "Abc")),                           # trivial choice
    _mcq(explanation="short"),                                # trivial explanation
    _mcq(choices=("Paris", "False")),                         # undesired choice
    _mcq(choices=("Paris", "Both A and B")),                  # both/neither artifact
], ids=["stem", "dup", "short-choice", "explanation", "undesired", "both"])
def test_validate_mcq_filters_incomplete_questions(q):
    assert validate_mcq(ExplodingLLM(), {"geo": [q]}, num_choices=2, shuffle=False) == {}


class PplLLM:
    def __init__(self, ppls):
        self.ppls, self.contexts = ppls, []

    def log_prob(self, texts, contexts, return_ppl):
        assert return_ppl
        self.contexts.extend(contexts)
        return torch.tensor(self.ppls[:len(texts)])


def test_sort_questions_orders_by_perplexity():
    q1, q2 = _mcq(qid="q-1"), _mcq(qid="q-2")
    prompts = ["SEED HDR\n1. first prompt", "SEED HDR\n1. second prompt"]
    llm = PplLLM([5.0, 1.0])  # q-1 is the more perplexing question

    questions, sorted_prompts = sort_questions(llm, {"geo": [q1, q2]}, prompts, batch_size=4)

    assert [q["id"] for q in questions["geo"]] == ["q-2", "q-1"]
    assert sorted_prompts == ["SEED HDR\n1. second prompt", "SEED HDR\n1. first prompt"]
    # Perplexity conditions on the seeding prompt up to the question number
    assert llm.contexts == ["SEED HDR\n1. ", "SEED HDR\n1. "]


def test_shuffle_choices_preserves_the_answer():
    random.seed(0)
    questions = [_mcq(choices=("Paris", "London", "Berlin", "Madrid"), answer="c")]  # Berlin
    shuffle_choices(questions)

    [q] = questions
    labels = [c["label"] for c in q["question"]["choices"]]
    assert labels == ["a", "b", "c", "d"]  # relabeled in order
    [ans_text] = [c["text"] for c in q["question"]["choices"] if c["label"] == q["answerKey"]]
    assert ans_text == "Berlin"


def test_cli_lists_qgen_commands(capsys):
    from kcluster.cli import main

    main(["--help"])
    out = capsys.readouterr().out
    assert "qgen-generate" in out and "qgen-validate" in out and "classify" in out
