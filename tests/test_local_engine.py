"""Tests for the local engine substrate.

Everything needing a real model is exercised later with a tiny fixture model;
here we cover the pure helpers and that the module imports cleanly whenever
torch is available (the [local] extra).
"""

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


class _TinyLM:
    """A deterministic stand-in for a causal LM: logits from cumulative embeddings, kept or all."""

    def __init__(self, vocab: int = 7, dim: int = 4):
        import torch

        gen = torch.Generator().manual_seed(0)
        self.emb = torch.randn(vocab, dim, generator=gen)
        self.head = torch.randn(dim, vocab, generator=gen)

    def forward(self, input_ids, attention_mask=None, logits_to_keep: int = 0):
        from types import SimpleNamespace

        hidden = self.emb[input_ids].cumsum(dim=1)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return SimpleNamespace(logits=hidden @ self.head)

    __call__ = forward


class _TinyLMAllLogits(_TinyLM):
    def forward(self, input_ids, attention_mask=None):
        return super().forward(input_ids)

    __call__ = forward


def test_token_loss_keeps_only_the_logits_the_labels_need():
    import torch

    from kcluster.engine.local import _logits_to_keep, _token_loss

    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [0, 0, 2, 3, 4, 5]])
    # Left-padded scoring: labelled tokens are a suffix, with a gap where a
    # figure's soft tokens would sit (row 0, position 4)
    labels = torch.tensor([[-100, -100, -100, 4, -100, 6], [-100, -100, -100, -100, 4, 5]])
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    # From just before the first labelled column (3) to the end: 4 positions
    assert _logits_to_keep(_TinyLM(), labels, -100) == 4
    assert _logits_to_keep(_TinyLMAllLogits(), labels, -100) == 0
    assert _logits_to_keep(_TinyLM(), torch.full_like(labels, -100), -100) == 0
    # A label in the first or second column needs every logit anyway
    assert _logits_to_keep(_TinyLM(), torch.where(labels == -100, 0, labels)[:, :], -100) == 0

    kept = _token_loss(_TinyLM(), inputs, labels)
    full = _token_loss(_TinyLMAllLogits(), inputs, labels)
    assert kept.shape == (2, 3) and full.shape == (2, 5)
    assert torch.allclose(kept.sum(dim=-1), full.sum(dim=-1))
    assert (full[:, :2] == 0).all()  # ignored positions contribute nothing either way


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
    local.score_pairs("scorer", fake_tokenizer, pairs, str(tmp_path), batch_size=2, pad_to_multiple_of=8,
                      strategy="ddp")

    kwargs = seen["trainer"]
    assert (kwargs["accelerator"], kwargs["devices"], kwargs["strategy"], kwargs["logger"]) == ("gpu", -1, "ddp", False)
    [writer] = kwargs["callbacks"]
    assert isinstance(writer, local.CustomWriter) and writer.output_dir == str(tmp_path)
    assert seen["model"] == "scorer" and seen["return_predictions"] is False
    # Unshuffled, in dataset order: from_shards puts each score back by its flat index
    assert [inputs["input_ids"][:, 0].tolist() for inputs, _ in seen["batches"]] == [[0, 1], [2]]
    assert [labels.tolist() for _, labels in seen["batches"]] == [[[-100, 7], [-100, 7]], [[-100, 7]]]
    assert seen["pad"] == [8, 8]

    # Trainer arguments pass through: one device, and no strategy unless asked for
    local.score_pairs("scorer", fake_tokenizer, pairs, str(tmp_path), devices=1)
    assert seen["trainer"]["devices"] == 1 and "strategy" not in seen["trainer"]


def test_token_loss_is_computed_in_float32_whatever_the_models_dtype():
    import torch

    from kcluster.engine.local import _token_loss

    class _Bf16LM(_TinyLM):
        def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
            out = super().forward(input_ids, attention_mask, logits_to_keep)
            out.logits = out.logits.to(torch.bfloat16)
            return out

        __call__ = forward

    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[-100, -100, 3, 4, 5, 6]])
    loss = _token_loss(_Bf16LM(), {"input_ids": ids}, labels)
    assert loss.dtype == torch.float32
