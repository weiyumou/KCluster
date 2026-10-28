"""Tests for the ``kcluster serve`` command's model loading: the precision a model is served in."""

import argparse

import pytest

pytest.importorskip("torch")
pytest.importorskip("uvicorn")

import torch

from kcluster.commands import serve


def test_the_default_precision_is_the_models_own():
    parser = argparse.ArgumentParser()
    serve.add_arguments(parser)
    args = parser.parse_args(["--llm_path", "m"])
    assert args.dtype == "auto"
    assert serve.model_kwargs(args.dtype, args.quantize, args.trust_remote_code, args.llm_path) == {
        "dtype": "auto", "trust_remote_code": False}  # from_pretrained reads the config's dtype
    assert serve.model_kwargs("bfloat16", "none", True)["dtype"] is torch.bfloat16


def test_quantization_reads_the_compute_dtype_from_the_config(monkeypatch, tmp_path):
    pytest.importorskip("bitsandbytes")
    from transformers import AutoConfig

    class Config:
        dtype = "bfloat16"

    monkeypatch.setattr(AutoConfig, "from_pretrained", classmethod(lambda cls, path, **kw: Config()))
    kwargs = serve.model_kwargs("auto", "4bit", False, str(tmp_path))
    assert kwargs["quantization_config"].bnb_4bit_compute_dtype is torch.bfloat16
    assert serve.model_kwargs("float16", "4bit", False)["quantization_config"].bnb_4bit_compute_dtype is torch.float16


def test_a_config_naming_no_dtype_asks_for_the_flag(monkeypatch, tmp_path):
    from transformers import AutoConfig

    class Config:
        dtype = None

    monkeypatch.setattr(AutoConfig, "from_pretrained", classmethod(lambda cls, path, **kw: Config()))
    with pytest.raises(SystemExit, match="pass --dtype"):
        serve.config_dtype(str(tmp_path))
