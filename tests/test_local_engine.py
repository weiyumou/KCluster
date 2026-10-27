"""Tests for the local engine substrate.

Everything needing a real model is exercised later with a tiny fixture model;
here we cover the pure helpers and that the module imports cleanly whenever
torch is available (the [local] extra).
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from kcluster.engine.local import LargeLangModel, LogProbScorer


def test_log_prob_scorer_is_a_lightning_module():
    import lightning as L

    assert issubclass(LogProbScorer, L.LightningModule)


def test_model_wrapper_exposes_the_engine_surface():
    # The method set below is the de-facto engine interface: the Vertex
    # serving container dispatches jobs to these same names by string.
    for method in ("next_logits", "next_tokens", "complete_prompts", "log_prob", "encode"):
        assert callable(getattr(LargeLangModel, method))


def test_collate_pair_masks_context_and_padding_out_of_labels():
    import torch

    from kcluster.engine.local import collate_pair

    def fake_tokenizer(text, text_pair, **kwargs):
        # Two rows: token_type_ids mark text tokens with 1; context and
        # right-padding carry 0 and must be ignored in the labels.
        return {
            "input_ids": torch.tensor([[11, 12, 21, 22], [13, 23, 24, 0]]),
            "token_type_ids": torch.tensor([[0, 0, 1, 1], [0, 1, 1, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        }

    inputs, labels = collate_pair([("ctx a", "txt a"), ("ctx b", "txt b")], tokenizer=fake_tokenizer)
    assert "token_type_ids" not in inputs
    assert labels.tolist() == [[-100, -100, 21, 22], [-100, 23, 24, -100]]


def test_custom_writer_shard_filenames_are_rank_stamped(tmp_path):
    # PointwiseMutualInfo.load_probs discovers shards by these exact names.
    import torch

    from kcluster.engine.local import CustomWriter

    class FakeTrainer:
        global_rank = 0

    writer = CustomWriter(output_dir=str(tmp_path), write_interval="epoch")
    writer.write_on_epoch_end(FakeTrainer(), None, [torch.tensor([1.0])], [[torch.tensor([0])]])
    assert (tmp_path / "predictions_0.pt").exists()
    assert (tmp_path / "batch_indices_0.pt").exists()


@pytest.mark.filterwarnings("ignore:.*pin_memory.*:UserWarning")  # no CUDA on a dev machine
def test_score_pairs_feeds_the_grid_to_a_gpu_trainer_in_order(tmp_path, monkeypatch):
    import torch

    from kcluster.engine import local

    seen = {"pad": []}

    class FakeTrainer:
        is_global_zero = True
        strategy = SimpleNamespace(barrier=lambda: seen.update(barrier=True))

        def __init__(self, **kwargs):
            seen["trainer"] = kwargs

        def predict(self, model, dataloaders, return_predictions):
            seen.update(model=model, batches=list(dataloaders), return_predictions=return_predictions)

    def fake_tokenizer(text, text_pair, **kwargs):
        seen["pad"].append(kwargs["pad_to_multiple_of"])
        return {"input_ids": torch.tensor([[int(t[1:]), 7] for t in text_pair]),
                "token_type_ids": torch.tensor([[0, 1]] * len(text))}

    monkeypatch.setattr(local.L, "Trainer", FakeTrainer)
    pairs = [("c0", "t0"), ("c1", "t1"), ("c2", "t2")]
    first = local.score_pairs("scorer", fake_tokenizer, pairs, str(tmp_path), batch_size=2, pad_to_multiple_of=8,
                              strategy="ddp")
    assert first is True and seen["barrier"]  # every rank has written before the first goes on

    kwargs = seen["trainer"]
    assert (kwargs["accelerator"], kwargs["devices"], kwargs["strategy"], kwargs["logger"]) == ("gpu", -1, "ddp", False)
    [writer] = kwargs["callbacks"]
    assert isinstance(writer, local.CustomWriter) and writer.output_dir == str(tmp_path)
    assert seen["model"] == "scorer" and seen["return_predictions"] is False
    # Unshuffled, in dataset order: from_shards puts each score back by its flat index
    assert [inputs["input_ids"][:, 0].tolist() for inputs, _ in seen["batches"]] == [[0, 1], [2]]
    assert [labels.tolist() for _, labels in seen["batches"]] == [[[-100, 7], [-100, 7]], [[-100, 7]]]
    assert seen["pad"] == [8, 8]

    # Trainer arguments pass through: one device of any kind, and no strategy unless asked for
    local.score_pairs("scorer", fake_tokenizer, pairs, str(tmp_path), devices=1, accelerator="auto")
    assert (seen["trainer"]["devices"], seen["trainer"]["accelerator"]) == (1, "auto")
    assert "strategy" not in seen["trainer"]


@pytest.mark.filterwarnings("ignore:.*pin_memory.*:UserWarning")  # no CUDA on a dev machine
@pytest.mark.filterwarnings("ignore:GPU available but not used", "ignore:.*does not have many workers",
                            "ignore:.*LeafSpec.*")  # Lightning's notices on a CPU run
def test_a_grid_scored_in_shards_comes_back_cell_by_cell(tmp_path):
    """ScoringGrid's flat order, the shards the scorer writes and from_shards agree: every score lands in its own
    cell, for a grid with more rows than columns, scored over several batches by a real Trainer."""
    import lightning as L
    import numpy as np
    import torch

    from kcluster.core.pmi import PointwiseMutualInfo
    from kcluster.engine.local import score_pairs
    from kcluster.tasks.congruity import ScoringGrid

    class NameGrid(ScoringGrid):
        def marginal(self, col):
            return "no question"

        def context(self, row, col):
            return f"question {row}"

    grid = NameGrid([" flexibility.", " addition."], n_rows=3)
    contexts = ["no question", "question 0", "question 1", "question 2"]

    def tokenizer(text, text_pair, **kwargs):
        # One token for the context and one for the text, each its index
        return {"input_ids": torch.tensor([[contexts.index(c), grid.texts.index(t)] for c, t in zip(text, text_pair)]),
                "token_type_ids": torch.tensor([[0, 1]] * len(text))}

    class CellScorer(L.LightningModule):
        def forward(self, batch):
            inputs, _ = batch
            return 10.0 * inputs["input_ids"][:, 0] + inputs["input_ids"][:, 1]

    assert score_pairs(CellScorer(), tokenizer, grid, str(tmp_path), batch_size=3, accelerator="cpu", devices=1)
    PointwiseMutualInfo.from_shards(str(tmp_path), grid.n_rows, len(grid.texts)).save(str(tmp_path / "scores.npy"))
    # Row 0: each text after no question; row 1 + r: after question r
    assert np.array_equal(np.load(tmp_path / "scores.npy"), 10.0 * np.arange(4)[:, None] + np.arange(2))
