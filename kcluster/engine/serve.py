"""HTTP serving for the local engine: a FastAPI app over ``LargeLangModel``.

``kcluster serve`` loads one model and exposes each engine method as a POST
route of the same name — ``/complete_prompts``, ``/log_prob``,
``/next_tokens``, ``/next_logits``, ``/encode`` — plus ``/tokenize`` and a
``GET /health``. Request bodies are the method's keyword arguments;
``/complete_prompts`` additionally forwards any unknown fields to HF
``generate`` (``num_beams``, ``do_sample``, ``guidance_scale``, ...), so a
prompt experiment never needs a server change. Every response carries the
served ``model_id`` so logged results are self-describing.

The app is model-agnostic: anything ``LargeLangModel`` loads can be served,
and ``check_tokenizer`` refuses at startup a tokenizer that cannot mark the
context/text split the scoring methods rely on. One GPU, one model, one
request at a time (a lock serializes inference; ``/health`` stays live).

Requires fastapi + uvicorn (the ``serve`` extra); the model itself needs the
``local`` extra. ``kcluster.engine.http.HttpLangModel`` is the matching client.
"""

import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict


def check_tokenizer(tokenizer) -> None:
    """Raise unless paired encoding yields token_type_ids that separate the two texts.

    ``log_prob`` and ``encode`` tokenize ``(context, text)`` pairs and use
    ``token_type_ids`` to score or pool only the text. A tokenizer that
    returns all zeros there would silently treat the whole sequence as text,
    so a swapped-in model must pass this before it is served.
    """
    encoded = tokenizer(text=["alpha"], text_pair=["beta"], return_token_type_ids=True)
    type_ids = encoded.get("token_type_ids")
    if not type_ids or set(type_ids[0]) != {0, 1}:
        raise RuntimeError(
            f"{type(tokenizer).__name__} does not mark context/text pairs with token_type_ids "
            f"(got {type_ids}); log_prob and encode cannot separate the context from the text with it")


def to_jsonable(value: Any) -> Any:
    """Tensors/arrays to nested lists; tuples to lists; everything else as is."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompleteRequest(BaseModel):
    """Unknown fields are forwarded to ``generate`` (e.g. ``num_beams``)."""
    model_config = ConfigDict(extra="allow")
    prompts: list[str]
    stop_tokens: list[str] | None = None
    max_new_tokens: int = 200
    pad_to_multiple_of: int | None = None


class LogProbRequest(_Body):
    texts: list[str]
    contexts: list[str] | None = None
    ignore_idx: int = -100
    pad_to_multiple_of: int | None = None
    return_ppl: bool = False


class NextTokensRequest(_Body):
    prompts: list[str]
    choices: list[str] | None = None
    top_k: int = 1
    pad_to_multiple_of: int | None = None


class NextLogitsRequest(_Body):
    prompts: list[str]
    normalize: bool = False
    pad_to_multiple_of: int | None = None


class EncodeRequest(_Body):
    texts: list[str]
    contexts: list[str] | None = None
    pad_to_multiple_of: int | None = None


class TokenizeRequest(_Body):
    texts: list[str]


def create_app(llm, model_id: str) -> FastAPI:
    """Build the app around a loaded ``LargeLangModel`` (or any object with its methods)."""
    app = FastAPI(title="kcluster serve", description=f"kcluster local engine serving `{model_id}`")
    state = {"lock": threading.Lock(), "started": time.time(), "last_request": time.time(), "requests": 0}

    def run(method: str, **kwargs):
        """Call one engine method under the lock; runs on the threadpool."""
        state["last_request"] = time.time()
        state["requests"] += 1
        fn = (lambda texts: llm.tokenizer(texts)["input_ids"]) if method == "tokenize" else getattr(llm, method)
        with state["lock"], _inference_mode():
            try:
                result = fn(**kwargs)
            except Exception as e:  # surface the engine's message, not a bare 500
                raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
        return {"model_id": model_id, "result": to_jsonable(result)}

    @app.get("/health")
    def health() -> dict:
        now = time.time()
        model = getattr(llm, "model", None)
        return {
            "status": "ok",
            "model_id": model_id,
            "device": str(getattr(llm, "device", "unknown")),
            "dtype": str(getattr(model, "dtype", "unknown")),
            "uptime_seconds": round(now - state["started"], 1),
            "idle_seconds": round(now - state["last_request"], 1),
            "requests": state["requests"],
        }

    @app.post("/complete_prompts")
    async def complete_prompts(body: CompleteRequest) -> dict:
        """``result``: one completion string per prompt."""
        return await run_in_threadpool(run, "complete_prompts", **body.model_dump())

    @app.post("/log_prob")
    async def log_prob(body: LogProbRequest) -> dict:
        """``result``: one log-probability (or perplexity) per text, shape (N,)."""
        return await run_in_threadpool(run, "log_prob", **body.model_dump())

    @app.post("/next_tokens")
    async def next_tokens(body: NextTokensRequest) -> dict:
        """``result``: per prompt, the ``top_k`` most likely next tokens."""
        return await run_in_threadpool(run, "next_tokens", **body.model_dump())

    @app.post("/next_logits")
    async def next_logits(body: NextLogitsRequest) -> dict:
        """``result``: next-token logits (or log-probs), shape (N, V)."""
        return await run_in_threadpool(run, "next_logits", **body.model_dump())

    @app.post("/encode")
    async def encode(body: EncodeRequest) -> dict:
        """``result``: mean-pooled last-layer embeddings, shape (N, H)."""
        return await run_in_threadpool(run, "encode", **body.model_dump())

    @app.post("/tokenize")
    async def tokenize(body: TokenizeRequest) -> dict:
        """``result``: token id lists, unpadded, one per text."""
        return await run_in_threadpool(run, "tokenize", texts=body.texts)

    return app


def _inference_mode():
    """``torch.inference_mode()`` when torch is present; a no-op for stub engines in tests."""
    try:
        import torch
    except ImportError:
        from contextlib import nullcontext
        return nullcontext()
    return torch.inference_mode()
