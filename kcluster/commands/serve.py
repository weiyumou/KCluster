"""Serve a local model over HTTP for interactive experiments.

Loads one causal LM through ``LargeLangModel`` and exposes the engine's
methods as routes (see ``kcluster.engine.serve``); ``kcluster.engine.http``
is the matching client. Meant for a GPU box reached through an SSH tunnel —
it binds to localhost by default and does no authentication of its own.
``deploy/vm/README.md`` walks through the L4 VM setup.

The model is a startup argument, not a property of the server: swap
``--llm_path`` (and ``--quantize`` for models that do not fit on the GPU) and
the same routes serve the new weights, in the precision the model's config
names unless ``--dtype`` says otherwise. Startup refuses a tokenizer the
engine cannot separate a text from its context with, since ``log_prob`` and
``encode`` depend on that.
"""

import argparse
import os

import torch
import uvicorn

from kcluster.engine.local import DTYPES, LargeLangModel
from kcluster.engine.serve import check_tokenizer, create_app

#: The precision a model is loaded in unless asked otherwise: the one its config names (``dtype="auto"``).
DEFAULT_DTYPE = "auto"


def config_dtype(llm_path: str, trust_remote_code: bool = False) -> torch.dtype:
    """The precision a model's config names (Phi-2 float16, the Phi-4 and Gemma families bfloat16)."""
    from transformers import AutoConfig

    dtype = AutoConfig.from_pretrained(llm_path, local_files_only=True, trust_remote_code=trust_remote_code).dtype
    if dtype is None:
        raise SystemExit(f"{llm_path}'s config names no dtype; pass --dtype")
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def model_kwargs(dtype: str, quantize: str, trust_remote_code: bool, llm_path: str | None = None) -> dict:
    """``from_pretrained`` arguments for the requested precision: ``auto`` is the model config's own, which
    quantization reads from ``llm_path`` for its compute dtype."""
    kwargs = {"dtype": "auto" if dtype == "auto" else DTYPES[dtype], "trust_remote_code": trust_remote_code}
    if quantize != "none":
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise SystemExit(f"--quantize {quantize} needs bitsandbytes (pip install bitsandbytes): {e}") from None
        if quantize == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            compute = config_dtype(llm_path, trust_remote_code) if dtype == "auto" else DTYPES[dtype]
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=compute, bnb_4bit_quant_type="nf4")
    return kwargs


def main(args):
    model_id = args.model_id or os.path.basename(os.path.normpath(args.llm_path))
    llm = LargeLangModel(args.llm_path, device=args.device, add_bos=args.add_bos,
                         **model_kwargs(args.dtype, args.quantize, args.trust_remote_code, args.llm_path))
    check_tokenizer(llm.tokenizer)
    images = f"images at {llm.image_token}" if llm.image_token else "text only"
    print(f"Serving {model_id!r} ({llm.model.dtype}, {llm.device}, {images}) on http://{args.host}:{args.port} "
          f"— OpenAPI schema at /openapi.json, docs at /docs")
    uvicorn.run(create_app(llm, model_id), host=args.host, port=args.port, log_level=args.log_level)


def add_arguments(parser):
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded causal LM")
    parser.add_argument("--host", default="127.0.0.1", type=str,
                        help="Bind address; keep localhost and reach it through an SSH tunnel")
    parser.add_argument("--port", default=8080, type=int, help="Port to listen on")
    parser.add_argument("--device", default="auto", type=str, help="device_map for from_pretrained")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=["auto", *sorted(DTYPES)],
                        help="Model precision (default: the one the model's config names)")
    parser.add_argument("--quantize", default="none", choices=["none", "8bit", "4bit"],
                        help="bitsandbytes quantization, for models that do not fit in --dtype")
    parser.add_argument("--add_bos", action=argparse.BooleanOptionalAction, default=None,
                        help="Put the model's BOS token before every string prompt that lacks one (--no-add_bos: "
                             "add no special tokens; default: the tokenizer's own rule). A tokenizer that adds "
                             "none, such as Gemma 4's instruction-tuned one, wants --add_bos for string prompts")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Allow the checkpoint's own modeling code (older Phi-2 checkpoints need it)")
    parser.add_argument("--model_id", default=None, type=str,
                        help="Name reported in every response (default: the last path component of --llm_path)")
    parser.add_argument("--log_level", default="info", type=str, help="uvicorn log level")
