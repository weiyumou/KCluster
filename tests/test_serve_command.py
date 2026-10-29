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
