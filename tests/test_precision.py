"""The precision a model is scored in: its own dtype for the forward, float32 at the output head, and the
``--dtype`` arguments that choose the former."""

import pytest

pytest.importorskip("torch")

import torch
import torch.nn.functional as F
from torch import nn

from kcluster.engine import local
from kcluster.engine.local import LargeLangModel, config_dtype, float32_head, model_kwargs


class _HeadedModel(nn.Module):
    """A model with an output head over a vocabulary, in a given dtype."""

    def __init__(self, vocab: int = 10, dim: int = 4, dtype=torch.bfloat16, bias: bool = False):
        super().__init__()
        self.lm_head = nn.Linear(dim, vocab, bias=bias).to(dtype)

    def get_output_embeddings(self):
        return self.lm_head


def test_the_head_projects_in_float32_a_chunk_of_the_vocabulary_at_a_time():
    torch.manual_seed(0)
    model = float32_head(_HeadedModel(bias=True), chunk=4)  # 10 rows: chunks of 4, 4, 2
    hidden = torch.randn(2, 3, 4).to(torch.bfloat16)
    logits = model.lm_head(hidden)
    assert logits.dtype == torch.float32 and logits.shape == (2, 3, 10)
    exact = F.linear(hidden.float(), model.lm_head.weight.float(), model.lm_head.bias.float())
    assert torch.equal(logits, exact)
    assert model.lm_head.weight.dtype == torch.bfloat16  # the weights stay as loaded


def test_the_head_is_left_alone_when_there_is_nothing_to_gain():
    fp32 = float32_head(_HeadedModel(dtype=torch.float32))
    assert "forward" not in fp32.lm_head.__dict__  # the module's own forward, unpatched

    class Headless(nn.Module):
        def get_output_embeddings(self):
            return None

    assert "forward" not in float32_head(Headless()).__dict__

    quantized = _HeadedModel()
    quantized.lm_head.weight = nn.Parameter(torch.zeros(10, 4, dtype=torch.uint8), requires_grad=False)
    assert "forward" not in float32_head(quantized).lm_head.__dict__


def test_a_model_without_get_output_embeddings_is_patched_through_lm_head():
    class Bare(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(4, 10).to(torch.bfloat16)

    model = float32_head(Bare())
    assert model.lm_head(torch.ones(1, 4, dtype=torch.bfloat16)).dtype == torch.float32


def test_load_model_applies_the_float32_head(monkeypatch):
    loaded = _HeadedModel()
    monkeypatch.setattr(local.AutoModelForCausalLM, "from_pretrained", lambda path, **kw: loaded)
    model = LargeLangModel.load_model("m", dtype=torch.bfloat16)
    assert model is loaded and model.lm_head(torch.ones(1, 4, dtype=torch.bfloat16)).dtype == torch.float32


def test_encode_asks_a_model_that_can_for_one_positions_logits(make_tokenizer):
    seen = {}

    class Recorder:
        device = torch.device("cpu")

        def forward(self, input_ids, attention_mask=None, output_hidden_states=False, logits_to_keep=0):
            seen["logits_to_keep"] = logits_to_keep
            hidden = torch.ones(*input_ids.shape, 3)
            return type("Out", (), {"hidden_states": (hidden,)})()

        __call__ = forward

    llm = LargeLangModel.__new__(LargeLangModel)
    llm.tokenizer, llm.model, llm.add_bos = make_tokenizer(["Exercise:", "which", "is"]), Recorder(), None
    llm.encode([" which is"], ["Exercise:"])
    assert seen["logits_to_keep"] == 1


def test_the_default_precision_is_the_models_own():
    assert model_kwargs("auto", trust_remote_code=True) == {"dtype": "auto", "trust_remote_code": True}
    assert model_kwargs("bfloat16", "none", True)["dtype"] is torch.bfloat16


def test_quantization_reads_the_compute_dtype_from_the_config(monkeypatch, tmp_path):
    pytest.importorskip("bitsandbytes")
    from transformers import AutoConfig

    class Config:
        dtype = "bfloat16"

    monkeypatch.setattr(AutoConfig, "from_pretrained", classmethod(lambda cls, path, **kw: Config()))
    kwargs = model_kwargs("auto", "4bit", False, str(tmp_path))
    assert kwargs["quantization_config"].bnb_4bit_compute_dtype is torch.bfloat16
    assert model_kwargs("float16", "4bit", False)["quantization_config"].bnb_4bit_compute_dtype is torch.float16


def test_a_config_naming_no_dtype_asks_for_the_flag(monkeypatch, tmp_path):
    from transformers import AutoConfig

    class Config:
        dtype = None

    monkeypatch.setattr(AutoConfig, "from_pretrained", classmethod(lambda cls, path, **kw: Config()))
    with pytest.raises(SystemExit, match="pass --dtype"):
        config_dtype(str(tmp_path))


@pytest.mark.parametrize("command", ["pmi", "concept", "embed"])
def test_the_scoring_commands_load_a_model_in_its_own_precision_by_default(command):
    import argparse
    import importlib

    parser = argparse.ArgumentParser()
    importlib.import_module(f"kcluster.commands.{command}").add_arguments(parser)
    required = ["--llm_path", "m"] + (["--data_path", "d"] if command != "embed" else [])
    assert parser.parse_args(required).dtype == "auto"
    assert parser.parse_args([*required, "--dtype", "float16"]).dtype == "float16"
