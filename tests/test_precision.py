"""The precision a model is scored in: its own dtype for the forward (the ``--dtype`` arguments that choose it),
and one position's logits where none are read."""

import pytest

pytest.importorskip("torch")

import torch

from kcluster.engine.local import LargeLangModel, config_dtype, model_kwargs


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
