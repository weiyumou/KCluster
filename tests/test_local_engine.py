"""Tests for the local engine substrate.

The tokenization is exercised on a tiny real fast tokenizer (conftest's
``make_tokenizer``): what the engine reads for a prompt, where a text's
tokens are, and the BOS rule. Everything needing a real model is left to the
GPU runs.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from kcluster.engine.local import LargeLangModel, LogProbScorer, pair_inputs, render_prompt, token_count


def test_log_prob_scorer_is_a_lightning_module():
    import lightning as L

    assert issubclass(LogProbScorer, L.LightningModule)


def test_model_wrapper_exposes_the_engine_surface():
    # The method set below is the de-facto engine interface: the Vertex
    # serving container dispatches jobs to these same names by string.
    for method in ("next_logits", "next_tokens", "complete_prompts", "log_prob", "encode"):
        assert callable(getattr(LargeLangModel, method))


# --- prompts: strings as they are, conversations in the chat template ---

def test_a_string_prompt_is_read_as_it_is(make_tokenizer, chat_template):
    tokenizer = make_tokenizer(["a", "b"], chat_template=chat_template)
    assert render_prompt(tokenizer, "a b") == "a b"


def test_a_conversation_is_written_in_the_chat_template_with_the_turn_opened(make_tokenizer, chat_template):
    tokenizer = make_tokenizer(["a"], chat_template=chat_template)
    messages = [{"role": "user", "content": "Write an exercise."}, {"role": "assistant", "content": "Exercise 1"},
                {"role": "user", "content": "Write another one."}]
    assert render_prompt(tokenizer, messages) == (
        "<user>Write an exercise.</user><assistant>Exercise 1</assistant><user>Write another one.</user><assistant>")


def test_a_last_assistant_message_is_a_prefill_after_the_opened_turn(make_tokenizer, chat_template):
    tokenizer = make_tokenizer(["a"], chat_template=chat_template)
    messages = [{"role": "user", "content": "What does it test?"},
                {"role": "assistant", "content": "The above exercise tests whether the student can"}]
    assert render_prompt(tokenizer, messages) == (
        "<user>What does it test?</user><assistant>The above exercise tests whether the student can")
    assert render_prompt(tokenizer, messages[:1]) == "<user>What does it test?</user><assistant>"


# --- pairs: context and text read together, the text's tokens found by their offsets ---

def test_pair_inputs_mask_the_text_alone_and_pad_on_the_left(make_tokenizer):
    tokenizer = make_tokenizer(["ctx", "a", "txt", "b", "c"])
    inputs, is_text = pair_inputs(tokenizer, ["ctx a", "ctx"], [" txt b", " c"], pad_to_multiple_of=8)
    ids = {w: tokenizer.convert_tokens_to_ids(w) for w in ("ctx", "a", "txt", "b", "c")}
    pad = tokenizer.pad_token_id
    assert inputs["input_ids"].tolist() == [[pad] * 4 + [ids["ctx"], ids["a"], ids["txt"], ids["b"]],
                                            [pad] * 6 + [ids["ctx"], ids["c"]]]
    assert inputs["attention_mask"].tolist() == [[0] * 4 + [1] * 4, [0] * 6 + [1] * 2]
    assert is_text.tolist() == [[False] * 6 + [True] * 2, [False] * 7 + [True]]


def test_prompts_alone_have_no_text_tokens(make_tokenizer):
    tokenizer = make_tokenizer(["a", "b"])
    inputs, is_text = pair_inputs(tokenizer, ["a b", "a"])
    assert inputs["input_ids"].shape == (2, 2) and not is_text.any()


def test_the_split_owes_nothing_to_the_tokenizers_pair_template(make_tokenizer):
    # Llama's pair template would put a second <s> between context and text; the engine reads the pair as one
    # string, so the only <s> is the leading one, and it counts as context
    tokenizer = make_tokenizer(["a", "b"], bos="<s>")
    inputs, is_text = pair_inputs(tokenizer, ["a"], [" b"])
    assert tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist()) == ["<s>", "a", "b"]
    assert is_text.tolist() == [[False, False, True]]


def test_a_text_must_start_at_a_token_boundary(make_tokenizer):
    tokenizer = make_tokenizer(["ab", "a", "b"])
    with pytest.raises(ValueError, match="does not start at a token boundary"):
        pair_inputs(tokenizer, ["a"], ["b"])  # "ab" is one token, half of it context
    with pytest.raises(ValueError, match="tokenizes to nothing"):
        pair_inputs(tokenizer, ["a"], [" "])


def test_contexts_and_texts_must_pair_up(make_tokenizer):
    with pytest.raises(ValueError, match="equal length"):
        pair_inputs(make_tokenizer(["a"]), ["a", "a"], ["a"])


# --- the BOS rule ---

def test_a_bos_adding_tokenizer_adds_it_once(make_tokenizer):
    tokenizer = make_tokenizer(["a"], bos="<s>")
    # The tokenizer's default: one <s> before a bare prompt, none more before one that carries its own
    for prompt in ("a", "<s> a"):
        inputs, _ = pair_inputs(tokenizer, [prompt])
        assert tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist()) == ["<s>", "a"], prompt
    # add_bos=False: no special tokens at all
    inputs, _ = pair_inputs(tokenizer, ["a"], add_bos=False)
    assert tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist()) == ["a"]


def test_a_tokenizer_adding_no_bos_adds_it_on_request(make_tokenizer):
    from tokenizers import processors

    tokenizer = make_tokenizer(["a"], bos="<s>")
    tokenizer._tokenizer.post_processor = processors.TemplateProcessing(single="$A", pair="$A $B:1")  # Gemma's way
    inputs, _ = pair_inputs(tokenizer, ["a"])
    assert tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist()) == ["a"]
    inputs, _ = pair_inputs(tokenizer, ["a"], add_bos=True)
    assert tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist()) == ["<s>", "a"]
    inputs, _ = pair_inputs(tokenizer, ["<s> a"], add_bos=True)  # and not twice
    assert tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist()) == ["<s>", "a"]


def test_token_count_is_what_the_model_reads(make_tokenizer, chat_template):
    tokenizer = make_tokenizer(["a", "b"], bos="<s>", chat_template=chat_template)
    assert token_count(tokenizer, "a", " b") == 3
    assert token_count(tokenizer, "a", " b", add_bos=False) == 2
    messages = [{"role": "user", "content": "a"}]
    assert token_count(tokenizer, messages) == len(tokenizer(render_prompt(tokenizer, messages))["input_ids"]) == 2


# --- the methods, over a fake model ---

def _engine(tokenizer, model):
    llm = LargeLangModel.__new__(LargeLangModel)
    llm.tokenizer, llm.model, llm.add_bos = tokenizer, model, None
    return llm


def test_a_text_with_nothing_before_it_cannot_be_scored(make_tokenizer):
    import torch

    tokenizer = make_tokenizer(["a", "b"])
    llm = _engine(tokenizer, SimpleNamespace(device=torch.device("cpu")))
    with pytest.raises(ValueError, match="nothing before it"):
        llm.log_prob(["a b"])
    with pytest.raises(ValueError, match="nothing before it"):
        llm.log_prob([" b", " b"], ["a", ""])  # one row is enough to refuse


def test_log_prob_sums_the_text_tokens_losses(make_tokenizer):
    import torch

    tokenizer = make_tokenizer(["a", "b", "c"])
    vocab = len(tokenizer)

    class UniformModel:
        device = torch.device("cpu")

        def __call__(self, input_ids, attention_mask):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, vocab))

    llm = _engine(tokenizer, UniformModel())
    # Every token is equally likely: -log(V) per text token, the context's and padding's left out
    scores = llm.log_prob([" b c", " c"], ["a", "a b"])
    assert torch.allclose(scores, torch.tensor([-2.0, -1.0]) * torch.log(torch.tensor(float(vocab))))
    assert torch.allclose(llm.log_prob([" b c"], ["a"], return_ppl=True), torch.tensor([float(vocab)]))


def test_complete_prompts_stops_at_stop_strings_and_the_models_own_end(make_tokenizer, chat_template):
    import torch

    tokenizer = make_tokenizer(["a", "b", "."], chat_template=chat_template)
    seen = {}

    class EchoModel:
        device = torch.device("cpu")

        def generate(self, input_ids, attention_mask, **kwargs):
            seen.update(kwargs)
            new = torch.tensor([[tokenizer.convert_tokens_to_ids("b")]] * len(input_ids))
            return torch.cat([input_ids, new], dim=1)

    llm = _engine(tokenizer, EchoModel())
    assert llm.complete_prompts(["a", [{"role": "user", "content": "a"}]], stop_tokens=[".", ","],
                                stop_strings=["\n\n"], max_new_tokens=5, num_beams=2) == ["b", "b"]
    # Stop strings are matched on the generated text, beside the model's own end of sequence, which stays
    assert seen["stop_strings"] == [".", ",", "\n\n"] and seen["tokenizer"] is tokenizer
    assert "eos_token_id" not in seen and seen["num_beams"] == 2 and seen["max_new_tokens"] == 5


# --- the batch scorer ---

def test_collate_pair_masks_context_and_padding_out_of_labels(make_tokenizer):
    from kcluster.engine.local import collate_pair

    tokenizer = make_tokenizer(["a", "b", "c"])
    inputs, labels = collate_pair([("a", " b c"), ("a b", " c")], tokenizer=tokenizer)
    ids = {w: tokenizer.convert_tokens_to_ids(w) for w in "abc"}
    assert inputs["input_ids"].tolist() == [[ids["a"], ids["b"], ids["c"]], [ids["a"], ids["b"], ids["c"]]]
    assert labels.tolist() == [[-100, ids["b"], ids["c"]], [-100, -100, ids["c"]]]


class _TinyLM:
    """A deterministic stand-in for a causal LM: logits from cumulative embeddings, kept or all."""

    def __init__(self, vocab: int = 7, dim: int = 4):
        import torch

        gen = torch.Generator().manual_seed(0)
        self.emb = torch.randn(vocab, dim, generator=gen)
        self.head = torch.randn(dim, vocab, generator=gen)

    def forward(self, input_ids, attention_mask=None, logits_to_keep: int = 0):
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
    # Left-padded scoring: labelled tokens are a suffix, with a gap where an
    # image's soft tokens would sit (row 0, position 4)
    labels = torch.tensor([[-100, -100, -100, 4, -100, 6], [-100, -100, -100, -100, 4, 5]])
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    # From just before the first labelled column (3) to the end: 4 positions
    assert _logits_to_keep(_TinyLM(), labels, -100) == 4
    assert _logits_to_keep(_TinyLMAllLogits(), labels, -100) == 0
    assert _logits_to_keep(_TinyLM(), torch.full_like(labels, -100), -100) == 0
    # A label in the first or second column needs every logit anyway
    assert _logits_to_keep(_TinyLM(), torch.where(labels == -100, 0, labels), -100) == 0

    kept = _token_loss(_TinyLM(), inputs, labels)
    full = _token_loss(_TinyLMAllLogits(), inputs, labels)
    assert kept.shape == (2, 3) and full.shape == (2, 5)
    assert torch.allclose(kept.sum(dim=-1), full.sum(dim=-1))
    assert (full[:, :2] == 0).all()  # ignored positions contribute nothing either way


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


def test_custom_writer_shard_filenames_are_rank_stamped(tmp_path):
    # PointwiseMutualInfo.from_shards discovers shards by these exact names.
    import torch

    from kcluster.engine.local import CustomWriter

    class FakeTrainer:
        global_rank = 0

    writer = CustomWriter(output_dir=str(tmp_path), write_interval="epoch")
    writer.write_on_epoch_end(FakeTrainer(), None, [torch.tensor([1.0])], [[torch.tensor([0])]])
    assert (tmp_path / "predictions_0.pt").exists()
    assert (tmp_path / "batch_indices_0.pt").exists()


@pytest.mark.filterwarnings("ignore:.*pin_memory.*:UserWarning")  # no CUDA on a dev machine
def test_score_pairs_feeds_the_grid_to_a_gpu_trainer_in_order(tmp_path, monkeypatch, make_tokenizer):
    from kcluster.engine import local

    seen = {}

    class FakeTrainer:
        is_global_zero = True
        strategy = SimpleNamespace(barrier=lambda: seen.update(barrier=True))

        def __init__(self, **kwargs):
            seen["trainer"] = kwargs

        def predict(self, model, dataloaders, return_predictions):
            seen.update(model=model, batches=list(dataloaders), return_predictions=return_predictions)

    monkeypatch.setattr(local.L, "Trainer", FakeTrainer)
    tokenizer = make_tokenizer(["c0", "c1", "c2", "t"])
    pairs = [("c0", " t"), ("c1", " t"), ("c2", " t")]
    first = local.score_pairs("scorer", tokenizer, pairs, str(tmp_path), batch_size=2, pad_to_multiple_of=4,
                              strategy="ddp")
    assert first is True and seen["barrier"]  # every rank has written before the first goes on

    kwargs = seen["trainer"]
    assert (kwargs["accelerator"], kwargs["devices"], kwargs["strategy"], kwargs["logger"]) == ("gpu", -1, "ddp", False)
    [writer] = kwargs["callbacks"]
    assert isinstance(writer, local.CustomWriter) and writer.output_dir == str(tmp_path)
    assert seen["model"] == "scorer" and seen["return_predictions"] is False
    # Unshuffled, in dataset order: from_shards puts each score back by its flat index
    ids = {w: tokenizer.convert_tokens_to_ids(w) for w in ("c0", "c1", "c2", "t")}
    assert [inputs["input_ids"][:, -2].tolist() for inputs, _ in seen["batches"]] == [[ids["c0"], ids["c1"]], [ids["c2"]]]
    assert [labels[:, -1].tolist() for _, labels in seen["batches"]] == [[ids["t"], ids["t"]], [ids["t"]]]
    assert all(inputs["input_ids"].shape[1] == 4 for inputs, _ in seen["batches"])  # padded to the multiple

    # Trainer arguments pass through: one device of any kind, and no strategy unless asked for
    local.score_pairs("scorer", tokenizer, pairs, str(tmp_path), devices=1, accelerator="auto")
    assert (seen["trainer"]["devices"], seen["trainer"]["accelerator"]) == (1, "auto")
    assert "strategy" not in seen["trainer"]


@pytest.mark.filterwarnings("ignore:.*pin_memory.*:UserWarning")  # no CUDA on a dev machine
@pytest.mark.filterwarnings("ignore:GPU available but not used", "ignore:.*does not have many workers",
                            "ignore:.*LeafSpec.*")  # Lightning's notices on a CPU run
def test_a_grid_scored_in_shards_comes_back_cell_by_cell(tmp_path, make_tokenizer):
    """ScoringGrid's flat order, the shards the scorer writes and from_shards agree: every score lands in its own
    cell, for a grid with more rows than columns, scored over several batches by a real Trainer."""
    import lightning as L
    import numpy as np

    from kcluster.core.grid import ScoringGrid
    from kcluster.core.pmi import PointwiseMutualInfo
    from kcluster.engine.local import score_pairs

    contexts = ["q0", "q1", "q2", "q3"]  # the no-question row, then one per question
    texts = [" flexibility", " addition"]
    grid = ScoringGrid(texts, n_rows=3, context=lambda row, col: contexts[row + 1], marginal=contexts[0])
    tokenizer = make_tokenizer([*contexts, *(t.strip() for t in texts)])

    class CellScorer(L.LightningModule):
        def forward(self, batch):
            # One token for the context and one for the text: the cell's row and column, as a number
            inputs, _ = batch
            ids = inputs["input_ids"][:, -2:].tolist()
            return 10.0 * self._t([contexts.index(tokenizer.convert_ids_to_tokens(c)) for c, _ in ids]) \
                + self._t([texts.index(" " + tokenizer.convert_ids_to_tokens(t)) for _, t in ids])

        @staticmethod
        def _t(values):
            import torch
            return torch.tensor(values, dtype=torch.float32)

    assert score_pairs(CellScorer(), tokenizer, grid, str(tmp_path), batch_size=3, accelerator="cpu", devices=1)
    PointwiseMutualInfo.from_shards(str(tmp_path), *grid.shape).save(str(tmp_path / "scores.npy"))
    # Row 0: each text after no question; row 1 + r: after question r
    assert np.array_equal(np.load(tmp_path / "scores.npy"), 10.0 * np.arange(4)[:, None] + np.arange(2))


class _PositionedLM(_TinyLM):
    """A stand-in whose forward takes position ids, and records what it was given."""

    def __init__(self):
        super().__init__()
        self.seen = {}

    def forward(self, input_ids, attention_mask=None, position_ids=None, logits_to_keep: int = 0, **kwargs):
        self.seen = dict(position_ids=position_ids, **kwargs)
        return super().forward(input_ids, attention_mask, logits_to_keep)

    __call__ = forward


def test_a_padded_row_is_scored_at_its_own_positions():
    import torch

    from kcluster.engine.local import _forward_inputs, _token_loss

    ids = torch.tensor([[0, 0, 2, 3, 4], [1, 2, 3, 4, 5]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    inputs = {"input_ids": ids, "attention_mask": mask}
    model = _PositionedLM()
    # The padded row counts from its first real token; the pads get position 0, which the mask hides anyway
    assert _forward_inputs(model, inputs)["position_ids"].tolist() == [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]
    # Unpadded rows need none, and a model whose forward takes none is called as before
    assert "position_ids" not in _forward_inputs(model, {"input_ids": ids, "attention_mask": torch.ones_like(mask)})
    assert "position_ids" not in _forward_inputs(_TinyLM(), inputs)
    # The scoring pass carries them
    _token_loss(model, inputs, torch.masked_fill(ids, mask == 0, -100))
    assert model.seen["position_ids"].tolist() == [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]


def test_next_logits_and_encode_score_a_padded_row_at_its_own_positions(make_tokenizer):
    import torch

    from kcluster.engine.local import LargeLangModel

    class Recorder(_PositionedLM):
        device = torch.device("cpu")

        def forward(self, input_ids, attention_mask=None, position_ids=None, output_hidden_states=False, **kwargs):
            self.seen = dict(position_ids=position_ids)
            out = super().forward(input_ids, attention_mask, position_ids)
            out.hidden_states = (torch.ones(*input_ids.shape, 2),)
            return out

        __call__ = forward

    llm = LargeLangModel.__new__(LargeLangModel)
    llm.tokenizer, llm.model, llm.add_bos = make_tokenizer(["ctx", "a", "b"]), Recorder(), None
    llm.next_logits(["ctx a b", "ctx"])
    assert llm.model.seen["position_ids"].tolist() == [[0, 1, 2], [0, 0, 0]]
    llm.encode([" b"], ["ctx a"], pad_to_multiple_of=4)
    assert llm.model.seen["position_ids"].tolist() == [[0, 0, 1, 2]]
