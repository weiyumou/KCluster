"""Offline test of the pmi command's figure plumbing: which root and adapter
reach the scoring grid under each flag. The GPU work is stubbed out."""

import json
from types import SimpleNamespace

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
    monkeypatch.setattr(pmi, "LogProbScorer", lambda path, **kwargs: seen.update(model_kwargs=kwargs)
                        or SimpleNamespace(model=None))
    monkeypatch.setattr(pmi, "check_context", lambda grid, tokenizer, limit, **kwargs: seen.update(
        context_kwargs=kwargs) or 0)
    monkeypatch.setattr(pmi, "context_length", lambda model: None)
    monkeypatch.setattr(pmi, "score_pairs", lambda model, tokenizer, pairs, shard_dir, **kwargs: seen.update(
        pairs=pairs, scoring_kwargs=kwargs))
    return seen


def test_figures_reach_the_grid_with_the_models_adapter(bank, tmp_path, stubbed, monkeypatch):
    vision = SimpleNamespace(image_token="<|image|>")
    monkeypatch.setattr(pmi.figures, "vision_for", lambda llm_path, root: vision)
    cli_main(["pmi", "--llm_path", "gemma", "--data_path", str(bank), "--run_dir", str(tmp_path / "run"),
              "--dtype", "bfloat16"])
    grid = stubbed["pairs"]
    assert grid.figure_dir == str(tmp_path) and grid.image_token == "<|image|>"
    assert grid[0][1].startswith("<|image|>\n") and grid[0][3] == [str(tmp_path / "figs" / "q0.png")]
    assert stubbed["scoring_kwargs"]["vision"] is vision and stubbed["context_kwargs"]["vision"] is vision
    assert stubbed["model_kwargs"]["dtype"] == pmi.DTYPES["bfloat16"]


def test_ignore_figures_scores_the_markers_as_text(bank, tmp_path, stubbed):
    cli_main(["pmi", "--llm_path", "phi-2", "--data_path", str(bank), "--run_dir", str(tmp_path / "run"),
              "--ignore_figures"])
    grid = stubbed["pairs"]
    assert grid.figure_dir is None and not grid.shows_images and "[figure]" in grid[0][1]
    assert stubbed["scoring_kwargs"]["vision"] is None
    assert stubbed["model_kwargs"]["dtype"] == pmi.DTYPES["float16"]


def test_a_text_only_model_refuses_a_figure_bank(bank, tmp_path, stubbed, monkeypatch):
    from kcluster.engine import vision

    monkeypatch.setattr(vision.ImageAdapter, "load", classmethod(lambda cls, path, **kw: None))
    with pytest.raises(SystemExit, match="takes no images.*--ignore_figures"):
        cli_main(["pmi", "--llm_path", "phi-2", "--data_path", str(bank), "--run_dir", str(tmp_path / "run")])
