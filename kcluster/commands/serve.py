"""Serve a local model over HTTP for interactive experiments.

Loads one causal LM through ``LargeLangModel`` and exposes the engine's
methods as routes (see ``kcluster.engine.serve``); ``kcluster.engine.http``
is the matching client. Meant for a GPU box reached through an SSH tunnel —
it binds to localhost by default and does no authentication of its own.
``deploy/vm/README.md`` walks through the L4 VM setup.

The model is a startup argument, not a property of the server: swap
``--llm_path`` (and ``--quantize`` for models that do not fit in fp16) and
the same routes serve the new weights. Startup refuses a tokenizer whose
paired encoding cannot separate context from text, since ``log_prob`` and
``encode`` depend on that.
"""

import argparse
import os

import torch
import uvicorn

from kcluster.engine.local import LargeLangModel
from kcluster.engine.serve import check_tokenizer, create_app

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def model_kwargs(dtype: str, quantize: str, trust_remote_code: bool) -> dict:
    """``from_pretrained`` arguments for the requested precision."""
    kwargs = {"torch_dtype": DTYPES[dtype], "trust_remote_code": trust_remote_code}
    if quantize != "none":
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise SystemExit(f"--quantize {quantize} needs bitsandbytes (pip install bitsandbytes): {e}") from None
        if quantize == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=DTYPES[dtype], bnb_4bit_quant_type="nf4")
    return kwargs


def main(args):
    model_id = args.model_id or os.path.basename(os.path.normpath(args.llm_path))
    llm = LargeLangModel(args.llm_path, device=args.device,
                         **model_kwargs(args.dtype, args.quantize, args.trust_remote_code))
    check_tokenizer(llm.tokenizer)
    print(f"Serving {model_id!r} ({llm.model.dtype}, {llm.device}) on http://{args.host}:{args.port} "
          f"— OpenAPI schema at /openapi.json, docs at /docs")
    uvicorn.run(create_app(llm, model_id), host=args.host, port=args.port, log_level=args.log_level)


def add_arguments(parser):
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded causal LM")
    parser.add_argument("--host", default="127.0.0.1", type=str,
                        help="Bind address; keep localhost and reach it through an SSH tunnel")
    parser.add_argument("--port", default=8080, type=int, help="Port to listen on")
    parser.add_argument("--device", default="auto", type=str, help="device_map for from_pretrained")
    parser.add_argument("--dtype", default="float16", choices=sorted(DTYPES), help="Model precision")
    parser.add_argument("--quantize", default="none", choices=["none", "8bit", "4bit"],
                        help="bitsandbytes quantization, for models that do not fit in --dtype")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Allow the checkpoint's own modeling code (older Phi-2 checkpoints need it)")
    parser.add_argument("--model_id", default=None, type=str,
                        help="Name reported in every response (default: the last path component of --llm_path)")
    parser.add_argument("--log_level", default="info", type=str, help="uvicorn log level")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
