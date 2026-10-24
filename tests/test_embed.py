"""Offline end-to-end tests of the embed command.

Fake encoders return planted, cosine-separable vectors; real affinity
propagation runs; correctly labeled cosine KC models land in an existing
result dir. No model weights are involved.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

from kcluster.cli import main as cli_main
from kcluster.commands import embed
from kcluster.io.jsonl import dump_questions

GROUPS = ["alpha"] * 3 + ["beta"] * 3

# Two directions, cosine-separable along the planted groups
PLANTED = np.array(
    [[1.0, 0.01], [1.0, 0.02], [1.0, 0.03],
     [0.01, 1.0], [0.02, 1.0], [0.03, 1.0]]
)


class FakeSentenceTransformer:
    def __init__(self, path, **kwargs):
        assert path == "st-path"

    def encode(self, texts):
        assert len(texts) == 6
        return PLANTED.copy()


@pytest.fixture
def result_dir(make_questions, make_result_dir):
    """A result dir as the concept step (or vertex-build-kc) leaves it."""
    return make_result_dir(make_questions(), GROUPS)


def _assert_model(rd, name):
    # concept-cosine clusters the concept phrases, so it files under concept/;
    # the two question encoders are embed-family (D15)
    sub = "concept" if name == "concept" else "embed"
    assert np.load(rd / "mat" / "embed" / f"questions_{name}-embed.npy").shape == (6, 2)
    kc = pd.read_csv(rd / "kc" / sub / f"questions_{name}-cosine-kc.csv")
    assert kc["KC"].tolist() == GROUPS, name
    assert kc["KC-raw"].str.fullmatch(r"KC-\d+").all(), name


def test_embed_sbert_builds_question_and_concept_models(result_dir, monkeypatch):
    monkeypatch.setattr(embed, "SentenceTransformer", FakeSentenceTransformer)
    cli_main(["embed", "--result_dir", str(result_dir), "--sent_path", "st-path"])

    # One SentenceTransformer, two models: questions and concept phrases
    _assert_model(result_dir, "sbert")
    _assert_model(result_dir, "concept")

    breadcrumb = json.loads((result_dir / "args-embed-questions.json").read_text())
    assert breadcrumb["data_path"].endswith("questions.jsonl")


def test_embed_llm_builds_the_llm_model(result_dir, monkeypatch):
    monkeypatch.setattr(embed, "LargeLangModel", lambda path, **kwargs: object())
    monkeypatch.setattr(embed, "extract_question_embeds",
                        lambda llm, questions, batch_size, figure_dir=None: torch.tensor(PLANTED))
    cli_main(["embed", "--result_dir", str(result_dir), "--llm_path", "llm-path", "--batch_size", "4"])

    _assert_model(result_dir, "llm")
    assert not (result_dir / "kc" / "embed" / "questions_sbert-cosine-kc.csv").exists()


def test_embed_works_on_a_vertex_course_dir(result_dir, monkeypatch):
    """Vertex course dirs carry an args-kc breadcrumb instead of args-concept;
    the data path is recovered all the same (engine-agnostic by design)."""
    (result_dir / "args-concept-questions.json").rename(result_dir / "args-kc-questions.json")
    monkeypatch.setattr(embed, "SentenceTransformer", FakeSentenceTransformer)
    cli_main(["embed", "--result_dir", str(result_dir), "--sent_path", "st-path"])
    _assert_model(result_dir, "sbert")


def test_embed_data_path_overrides_an_unreachable_recorded_path(result_dir, tmp_path, monkeypatch):
    """A Vertex run launched from a laptop records a path a cluster does not
    have; --data_path points at the copy that is actually reachable."""
    moved = tmp_path / "elsewhere" / "questions.jsonl"
    moved.parent.mkdir()
    original = json.loads((result_dir / "args-concept-questions.json").read_text())["data_path"]
    os.rename(original, moved)

    monkeypatch.setattr(embed, "SentenceTransformer", FakeSentenceTransformer)
    with pytest.raises(SystemExit, match="Question file not found"):
        cli_main(["embed", "--result_dir", str(result_dir), "--sent_path", "st-path"])

    cli_main(["embed", "--result_dir", str(result_dir), "--sent_path", "st-path", "--data_path", str(moved)])
    _assert_model(result_dir, "sbert")


def test_embed_rejects_a_question_file_of_the_wrong_size(result_dir, tmp_path, monkeypatch, make_questions):
    wrong = tmp_path / "short.jsonl"
    dump_questions(make_questions(4), str(wrong))
    monkeypatch.setattr(embed, "SentenceTransformer", FakeSentenceTransformer)
    with pytest.raises(SystemExit, match="wrong question file"):
        cli_main(["embed", "--result_dir", str(result_dir), "--sent_path", "st-path", "--data_path", str(wrong)])


def test_embed_requires_an_encoder(result_dir):
    with pytest.raises(SystemExit, match="--sent_path"):
        cli_main(["embed", "--result_dir", str(result_dir)])


def test_embed_requires_a_result_dir_without_a_run_dir(monkeypatch):
    monkeypatch.delenv("KCLUSTER_RUN_DIR", raising=False)
    with pytest.raises(SystemExit, match="--result_dir is required"):
        cli_main(["embed", "--sent_path", "st-path"])
