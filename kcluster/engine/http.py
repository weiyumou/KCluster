"""HTTP client for ``kcluster serve``: ``LargeLangModel``'s surface over the wire.

``HttpLangModel`` exposes the same method names and signatures as the local
engine, so tasks that only use the engine surface — ``extract_concepts``,
``extract_question_embeds``, ``validate_mcq``, ``sort_questions`` — run
unmodified against a served model::

    llm = HttpLangModel("http://localhost:8080")
    concepts = extract_concepts(llm, questions, batch_size=16)

Scoring methods return torch tensors when torch is importable (what the
tasks expect) and numpy arrays otherwise, so the client itself needs neither
extra. ``tokenizer`` is a thin shim over ``/tokenize`` that covers what the
tasks call (``encode`` and ``__call__`` for id lists); anything needing the
real tokenizer object — ``generate_mcq`` passes it into ``generate`` — must
run the model in-process.

Only the standard library is used; the server side is ``kcluster.engine.serve``.
"""

import json
import urllib.error
import urllib.request
from typing import Any


class HttpTokenizer:
    """The slice of the tokenizer API the tasks use, answered by ``/tokenize``."""

    def __init__(self, client: "HttpLangModel"):
        self._client = client

    def encode(self, text: str) -> list[int]:
        return self._client._post("/tokenize", {"texts": [text]})[0]

    def __call__(self, texts: str | list[str], **kwargs) -> dict[str, list[list[int]]]:
        if kwargs:
            raise NotImplementedError(
                f"HttpTokenizer only returns id lists; unsupported arguments {sorted(kwargs)} — "
                "run the model in-process for anything needing the real tokenizer")
        single = isinstance(texts, str)
        ids = self._client._post("/tokenize", {"texts": [texts] if single else list(texts)})
        return {"input_ids": ids[0] if single else ids}


class HttpLangModel:
    """Client mirroring ``LargeLangModel`` against a ``kcluster serve`` endpoint."""

    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 600.0, tensors: str = "auto"):
        """``tensors``: ``"torch"``, ``"numpy"``, ``"list"``, or ``"auto"`` (torch if importable, else numpy)."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if tensors == "auto":
            try:
                import torch  # noqa: F401
                tensors = "torch"
            except ImportError:
                tensors = "numpy"
        if tensors not in ("torch", "numpy", "list"):
            raise ValueError(f"tensors must be 'torch', 'numpy', 'list' or 'auto', got {tensors!r}")
        self.tensors = tensors
        self._model_id: str | None = None

    # -- transport -----------------------------------------------------------

    def _post(self, path: str, payload: dict) -> Any:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as rsp:
                body = json.load(rsp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise RuntimeError(f"{path} failed ({e.code}): {detail}") from None
        self._model_id = body.get("model_id", self._model_id)
        return body["result"]

    def _tensor(self, value):
        if self.tensors == "torch":
            import torch
            return torch.tensor(value)
        if self.tensors == "numpy":
            import numpy as np
            return np.asarray(value)
        return value

    # -- engine surface --------------------------------------------------------

    def health(self) -> dict:
        with urllib.request.urlopen(self.base_url + "/health", timeout=self.timeout) as rsp:
            body = json.load(rsp)
        self._model_id = body.get("model_id", self._model_id)
        return body

    @property
    def model_id(self) -> str:
        """The served model's id, as reported by the server."""
        if self._model_id is None:
            self.health()
        return self._model_id

    @property
    def device(self):
        """Results arrive on the CPU; tasks call ``.to(llm.device)`` on their index tensors."""
        if self.tensors == "torch":
            import torch
            return torch.device("cpu")
        return "cpu"

    @property
    def tokenizer(self) -> HttpTokenizer:
        return HttpTokenizer(self)

    def complete_prompts(self, prompts: list[str], stop_tokens: list[str] | None = None, max_new_tokens: int = 200,
                         pad_to_multiple_of: int | None = None, **kwargs) -> list[str]:
        """Completes a batch of prompts; ``kwargs`` are HF ``generate`` arguments."""
        return self._post("/complete_prompts", {"prompts": list(prompts), "stop_tokens": stop_tokens,
                                                "max_new_tokens": max_new_tokens,
                                                "pad_to_multiple_of": pad_to_multiple_of, **kwargs})

    def log_prob(self, texts: list[str], contexts: list[str] | None = None,
                 ignore_idx: int = -100, pad_to_multiple_of: int | None = None, return_ppl: bool = False):
        """Log-prob of each text given its context, shape (N,)."""
        return self._tensor(self._post("/log_prob", {"texts": list(texts), "contexts": contexts,
                                                     "ignore_idx": ignore_idx,
                                                     "pad_to_multiple_of": pad_to_multiple_of,
                                                     "return_ppl": return_ppl}))

    def next_tokens(self, prompts: list[str], choices: list[str] | None = None,
                    top_k: int = 1, pad_to_multiple_of: int | None = None) -> list[tuple[str, ...]]:
        """The ``top_k`` most likely next tokens (optionally restricted to ``choices``) per prompt."""
        result = self._post("/next_tokens", {"prompts": list(prompts), "choices": choices, "top_k": top_k,
                                             "pad_to_multiple_of": pad_to_multiple_of})
        return [tuple(r) for r in result]

    def next_logits(self, prompts: list[str], normalize: bool = False, pad_to_multiple_of: int | None = None):
        """Next-token logits (log-probs with ``normalize``), shape (N, V)."""
        return self._tensor(self._post("/next_logits", {"prompts": list(prompts), "normalize": normalize,
                                                        "pad_to_multiple_of": pad_to_multiple_of}))

    def encode(self, texts: list[str], contexts: list[str] | None = None, pad_to_multiple_of: int | None = None):
        """Mean-pooled last-layer embeddings, shape (N, H)."""
        return self._tensor(self._post("/encode", {"texts": list(texts), "contexts": contexts,
                                                   "pad_to_multiple_of": pad_to_multiple_of}))
