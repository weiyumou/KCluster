"""Offline test of the pmi command's figure plumbing: which root and adapter
reach the scoring grid under each flag. The GPU work is stubbed out."""

import json

import pytest

pytest.importorskip("torch")

from kcluster.cli import main as cli_main
from kcluster.commands import pmi


@pytest.fixture
def bank(tmp_path, make_questions):
    questions = make_questions(3)
    questions[0]["question"]["stem"] = "[figure]\nStem 0?"
    questions[0]["figures"] = ["figs/q0.png"]
    (tmp_path / "figs").mkdir()
    (tmp_path / "figs" / "q0.png").write_bytes(b"png")
    path = tmp_path / "bank.jsonl"
    path.write_text("".join(json.dumps(q.data) + "\n" for q in questions))
    return path


@pytest.fixture
def stubbed(monkeypatch):
    seen = {}
    monkeypatch.setattr(pmi.LargeLangModel, "load_tokenizer", staticmethod(lambda path: "tokenizer"))
    monkeypatch.setattr(pmi, "LogProbScorer", lambda path, **kwargs: seen.setdefault("model_kwargs", kwargs))
    monkeypatch.setattr(pmi, "score_pairs", lambda model, tokenizer, pairs, shard_dir, **kwargs: seen.update(
        pairs=pairs, scoring_kwargs=kwargs))
    return seen


def test_figures_reach_the_grid_with_the_models_adapter(bank, tmp_path, stubbed, monkeypatch):
    monkeypatch.setattr(pmi, "load_adapter", lambda llm_path, root: ("adapter", llm_path, root))
    cli_main(["pmi", "--llm_path", "gemma", "--data_path", str(bank), "--run_dir", str(tmp_path / "run"),
              "--dtype", "bfloat16"])
    assert stubbed["pairs"].figure_dir == str(tmp_path)
    assert stubbed["scoring_kwargs"]["adapter"] == ("adapter", "gemma", str(tmp_path))
    assert stubbed["model_kwargs"]["dtype"] == pmi.DTYPES["bfloat16"]


def test_ignore_figures_scores_the_markers_as_text(bank, tmp_path, stubbed):
    cli_main(["pmi", "--llm_path", "phi-2", "--data_path", str(bank), "--run_dir", str(tmp_path / "run"),
              "--ignore_figures"])
    assert stubbed["pairs"].figure_dir is None and stubbed["scoring_kwargs"]["adapter"] is None
    assert stubbed["model_kwargs"]["dtype"] == pmi.DTYPES["float16"]


def test_a_text_only_model_refuses_a_figure_bank(bank, tmp_path, stubbed, monkeypatch):
    from kcluster.engine import local

    monkeypatch.setattr(local.ImageAdapter, "load", classmethod(lambda cls, path, **kw: None))
    with pytest.raises(SystemExit, match="takes no images.*--ignore_figures"):
        cli_main(["pmi", "--llm_path", "phi-2", "--data_path", str(bank), "--run_dir", str(tmp_path / "run")])
