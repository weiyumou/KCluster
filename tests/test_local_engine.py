"""Tests for the local engine substrate.

Everything needing a real model is exercised later with a tiny fixture model;
here we cover the pure helpers and that the module imports cleanly whenever
torch is available (the [local] extra).
"""

import pytest

pytest.importorskip("torch")

from kcluster.engine.local import LargeLangModel, LogProbScorer  # noqa: E402


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
