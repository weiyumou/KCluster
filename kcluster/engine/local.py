"""Local (in-process) execution substrate: a HuggingFace causal-LM wrapper and
the Lightning batch scorer that shards large scoring jobs across GPUs.

The engine is general: it knows nothing of questions or of KCluster's prompts.
What it takes is a *prompt*, the text a model reads, in either of two forms:

- a string, read verbatim. The tokenizer's own leading special tokens (a
  ``<bos>``) are added as its default says, unless the string already begins
  with the BOS token, so a prompt rendered from a chat template, which carries
  its own, is never given a second one. ``add_bos`` overrides the default
  either way: a tokenizer that adds none (Gemma 4's instruction-tuned one; its
  base one adds its own) wants ``True`` for a string prompt.
- a conversation, as a list of ``{"role", "content"}`` messages, written in
  the model's chat template with the model's turn opened at the end. A last
  message in the assistant's role is a *prefill*: it is written after the
  opened turn, as the words the model continues.

The scoring methods take a prompt as the context of each *text*, the string
whose tokens are scored (``log_prob``) or pooled (``encode``). Context and
text are tokenized together, as the model reads them, and the text's tokens
are found by their character offsets, so the split is the engine's and owes
nothing to a tokenizer's pair template (which may insert a BOS between the
two, as Llama's does). A text must start at a token boundary: give it the
leading space or line break it needs, as a completion would have.

Images: a model with a processor (``ImageAdapter``, ``kcluster.engine.vision``)
takes them. A prompt or a text places each image with the model's image
token (``image_token``), and every method takes the images at those places
(``images``, and ``context_images`` for the contexts of the paired methods),
one list of sources per prompt or text, in reading order. The image's tokens
are context, never scored or pooled. A model without a processor refuses
images; a prompt without any reads as before.

Precision: a model runs in the dtype it was loaded in, and the loss runs in
float32: the logits are upcast before the log-softmax, so each token's
log-prob and their sum over a text are exact to float32 given the logits
(in bfloat16 the sum alone rounds to whole nats at the magnitude of a
question). The logits themselves carry the model's precision: a bfloat16
logit near 25 is good to 0.125 nats, which adds a few tenths of a nat per
text that move with the batch shape, as the bfloat16 forward's own hidden
states do. A padded row is scored at its own positions (``position_ids``
counted from its first real token, for a model that takes them): without
that a left-padded row's tokens sit at shifted positions, which is the same
text to a rotary model in exact arithmetic but not in bfloat16. A run
repeats exactly when every batch has the same shape.
"""

import inspect
import itertools
import os
from functools import lru_cache, partial

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import BasePredictionWriter
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from kcluster.engine.vision import ImageAdapter

#: What a model reads before a text: the text itself, or a conversation as messages.
Prompt = str | list[dict]

#: The precisions a command loads a model in (``--dtype``). The published
#: runs scored in float16 (Phi-2); Gemma overflows in it and wants bfloat16.
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

#: The precision a command loads a model in unless ``--dtype`` says otherwise: the one its config names
#: (``dtype="auto"``; Phi-2 float16, which the published runs scored in, the Phi-4 and Gemma families bfloat16).
DEFAULT_DTYPE = "auto"


def config_dtype(llm_path: str, trust_remote_code: bool = False) -> torch.dtype:
    """The precision a model's config names (Phi-2 float16, the Phi-4 and Gemma families bfloat16)."""
    from transformers import AutoConfig

    dtype = AutoConfig.from_pretrained(llm_path, local_files_only=True, trust_remote_code=trust_remote_code).dtype
    if dtype is None:
        raise SystemExit(f"{llm_path}'s config names no dtype; pass --dtype")
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def model_kwargs(dtype: str, quantize: str = "none", trust_remote_code: bool = False,
                 llm_path: str | None = None) -> dict:
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


def render_prompt(tokenizer, prompt: Prompt) -> str:
    """The text a model reads for a prompt: a string as it is, messages in the tokenizer's chat template with the
    model's turn opened, and a last assistant message written after it as a prefill."""
    if isinstance(prompt, str):
        return prompt
    messages = list(prompt)
    prefill = messages.pop()["content"] if messages and messages[-1].get("role") == "assistant" else ""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill


def _adds_bos(tokenizer) -> bool:
    """Whether the tokenizer puts its BOS token before a text on its own."""
    bos_id = tokenizer.bos_token_id
    return bos_id is not None and tokenizer("x")["input_ids"][:1] == [bos_id]


def _with_specials(tokenizer, text: str, add_bos: bool | None) -> tuple[str, bool]:
    """The text to tokenize and whether to let the tokenizer add its special tokens, under the BOS rule above."""
    bos = tokenizer.bos_token
    if bos and text.startswith(bos):
        return text, False
    if add_bos is None:
        return text, True
    if add_bos and bos and not _adds_bos(tokenizer):
        return bos + text, True
    return text, add_bos


def _encode_pair(tokenizer, context: str, text: str, add_bos: bool | None) -> tuple[list[int], list[bool]]:
    """Token ids of a rendered context and its text read together, and the mask of the text's tokens."""
    context, specials = _with_specials(tokenizer, context, add_bos)
    enc = tokenizer(context + text, add_special_tokens=specials, return_offsets_mapping=True)
    boundary = len(context)
    if any(start < boundary < end for start, end in enc["offset_mapping"]):
        raise ValueError(f"the text {text[:30]!r} does not start at a token boundary after its context: give it the "
                         "leading space or line break a completion would have")
    # Special tokens the tokenizer adds carry an empty offset, and count as context.
    is_text = [start >= boundary and end > start for start, end in enc["offset_mapping"]]
    if text and not any(is_text):
        raise ValueError(f"the text {text[:30]!r} tokenizes to nothing")
    return enc["input_ids"], is_text


def _rows(tokenizer, contexts: list[Prompt], texts: list[str], vision: ImageAdapter | None,
          context_images, images) -> tuple[list[tuple[str, str]], dict]:
    """The (context, text) strings of a batch as the model reads them, their images in place, and the pixel
    inputs that go with them (empty without images)."""
    rendered = [render_prompt(tokenizer, c) for c in contexts]
    given = [per_row for per_row in (context_images, images) if per_row is not None and any(per_row)]
    if vision is None:
        if given:
            raise ValueError("images were given, but this model takes no images (it has no processor)")
        return list(zip(rendered, texts, strict=True)), {}
    n = len(texts)
    rows = [[(c, ci), (t, ti)] for c, t, ci, ti in zip(rendered, texts, context_images or [[]] * n,
                                                       images or [[]] * n, strict=True)]
    expanded, extra = vision.prepare(rows)
    return [(c, t) for c, t in expanded], extra


def pair_inputs(tokenizer, contexts: list[Prompt], texts: list[str] | None = None,
                pad_to_multiple_of: int = None, device=None, *, add_bos: bool | None = None,
                vision: ImageAdapter | None = None, context_images=None, images=None):
    """Model inputs for a batch of prompts, or of (context, text) pairs, and the mask of each text's tokens.

    Rows are padded on the left to one length, a multiple of ``pad_to_multiple_of`` if given. The mask covers the
    text alone, leaving out the context, the special tokens and the padding; with no texts it is all False. With
    images, ``vision`` puts each in its place and adds the pixel inputs, and the mask also leaves out the image
    soft tokens: a text's own tokens are what is scored.
    """
    texts = texts if texts is not None else [""] * len(contexts)
    if len(texts) != len(contexts):
        raise ValueError("Contexts and texts must have the equal length")
    pairs, extra = _rows(tokenizer, contexts, texts, vision, context_images, images)
    rows = [_encode_pair(tokenizer, c, t, add_bos) for c, t in pairs]
    length = max(len(ids) for ids, _ in rows)
    if pad_to_multiple_of:
        length = -(-length // pad_to_multiple_of) * pad_to_multiple_of
    pad_id = tokenizer.pad_token_id

    def padded(values, fill):
        return [[fill] * (length - len(v)) + list(v) for v in values]

    inputs = {"input_ids": torch.tensor(padded([ids for ids, _ in rows], pad_id)),
              "attention_mask": torch.tensor(padded([[1] * len(ids) for ids, _ in rows], 0))}
    is_text = torch.tensor(padded([mask for _, mask in rows], False))
    if extra:
        inputs.update(extra)
        if (mm_ids := vision.mm_token_type_ids(inputs["input_ids"])) is not None:
            inputs["mm_token_type_ids"] = mm_ids
        is_text &= ~vision.soft_token_mask(inputs["input_ids"])
    if device is not None:
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        is_text = is_text.to(device)
    return inputs, is_text


def token_count(tokenizer, context: Prompt, text: str = "", add_bos: bool | None = None,
                vision: ImageAdapter | None = None) -> int:
    """Tokens a model reads for a text after its context, as ``pair_inputs`` tokenizes them.

    An image at its place counts at most ``vision.tokens_per_image``, the processor's budget, in place of its token:
    the count is an upper bound for a prompt with images, exact without.
    """
    rendered = render_prompt(tokenizer, context)
    ids, _ = _encode_pair(tokenizer, rendered, text, add_bos)
    count = len(ids)
    if vision is not None and (per_image := vision.tokens_per_image) is not None:
        count += (per_image - 1) * vision.count(rendered + text)
    return count


def context_length(model) -> int | None:
    """Tokens the model reads at most, as its config names them; None when the config names no limit."""
    config = getattr(model, "config", model)
    for name in ("max_position_embeddings", "n_positions"):
        limit = getattr(config, name, None)
        if limit is not None:
            return int(limit)
    return None


@lru_cache(maxsize=None)
def _takes(model_class, name: str) -> bool:
    """Whether the model class's forward takes the keyword ``name``."""
    forward = getattr(model_class, "forward", model_class.__call__)
    return name in inspect.signature(forward).parameters


def _takes_logits_to_keep(model_class) -> bool:
    return _takes(model_class, "logits_to_keep")


def _forward_inputs(model, inputs: dict) -> dict:
    """The inputs for a forward pass: with ``position_ids`` counting each row's tokens from its first real one,
    when a row is padded and the model takes them.

    Left padding puts a row's tokens at positions shifted by the pad count. A rotary model reads the same text
    there in exact arithmetic (its attention depends on position differences), but not in bfloat16: the
    rotations differ, and a padded row's score moved with the pad count (Llama-2-7b, bf16: by up to 0.07
    nats; with the positions given, a padded row's offset no longer depends on the padding). A model with
    absolute position embeddings reads a different text altogether. ``generate`` derives the positions itself,
    so this is for the scoring passes.
    """
    mask = inputs.get("attention_mask")
    if mask is None or bool(mask.all()) or not _takes(type(model), "position_ids"):
        return inputs
    return {**inputs, "position_ids": (mask.cumsum(dim=-1) - 1).clamp(min=0)}


def _logits_to_keep(model, labels, ignore_idx: int) -> int:
    """How many trailing positions' logits the loss needs, or 0 for all of them.

    Scoring pads on the left, so the labelled tokens sit at the end of every
    row; only the logits from just before the first labelled column onward
    enter the loss. A vocabulary of 262k tokens (Gemma 4) makes the full
    ``(N, S, V)`` logits the largest tensor of a scoring job, so a model whose
    forward takes ``logits_to_keep`` is asked for that suffix only.
    """
    if not _takes_logits_to_keep(type(model)):
        return 0
    labelled = (labels != ignore_idx).any(dim=0)
    if not labelled.any():
        return 0
    keep = labels.size(1) - int(labelled.nonzero()[0]) + 1
    return keep if keep < labels.size(1) else 0


def _token_loss(model, inputs, labels, ignore_idx: int = -100):
    """Per-token cross-entropy of each label given the tokens before it, 0 where ignored.

    One column per position the loss needs (see ``_logits_to_keep``): the
    trailing ones, or all of them. Each row's sum is the same either way.
    """
    keep = _logits_to_keep(model, labels, ignore_idx)
    logits = model(**_forward_inputs(model, inputs),
                   **({"logits_to_keep": keep} if keep else {})).logits.transpose(-1, -2)  # (N, V, K)
    kept = logits.size(-1)
    # The loss in float32: a bfloat16 log-softmax is good to ~0.4% per token
    # and a bfloat16 sum of them to whole nats at the magnitudes of a scored
    # question (ulp 2 at 300), which is the resolution the PMI lives on. The
    # upcast is cheap on the kept logits.
    return F.cross_entropy(logits[..., :-1].float(), labels[:, -(kept - 1):], reduction="none",
                           ignore_index=ignore_idx)  # (N, K - 1)


def _check_preceded(inputs, is_text) -> None:
    """Raise if any text starts the sequence: nothing predicts its first token, so it cannot be scored."""
    first = inputs["attention_mask"].int().argmax(dim=1)  # the first real token of each left-padded row
    if is_text[torch.arange(len(first)), first].any():
        raise ValueError("a text cannot be scored with nothing before it: give it a context, or a tokenizer that "
                         "adds a BOS token (add_bos=True)")


class LargeLangModel:
    """Encapsulates a downloaded LLM and its tokenizer, and the processor of a model that takes images.

    ``add_bos`` is the engine's BOS rule for string prompts (module docstring): None follows the tokenizer's default,
    True puts the model's BOS token before every prompt that lacks one, False adds no special tokens. ``vision``
    is the model's ``ImageAdapter``, or None for a text-only model; ``image_token`` is where a prompt places an
    image.
    """

    vision: ImageAdapter | None = None

    def __init__(self, model_path: str, device: str = "auto", add_bos: bool | None = None, **model_args):
        # Load tokenizer, and the processor of a model that takes images
        self.tokenizer = LargeLangModel.load_tokenizer(model_path)
        self.add_bos = add_bos
        self.vision = ImageAdapter.load(model_path, **({"trust_remote_code": model_args["trust_remote_code"]}
                                                       if "trust_remote_code" in model_args else {}))

        # Load model
        self.model = LargeLangModel.load_model(model_path, device_map=device, **model_args).eval()

    @staticmethod
    def load_tokenizer(path: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side="left", **kwargs)
        if not tokenizer.is_fast:
            raise ValueError(f"{path} has no fast tokenizer; the engine splits a text from its context by "
                             "character offsets, which only a fast tokenizer reports")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @staticmethod
    def load_model(path: str, **kwargs):
        return AutoModelForCausalLM.from_pretrained(path, local_files_only=True, **kwargs)

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def pad_token(self) -> str:
        return self.tokenizer.pad_token

    @property
    def image_token(self) -> str | None:
        """The token a prompt places an image with, or None for a model that takes no images."""
        return self.vision.image_token if self.vision is not None else None

    def _inputs(self, contexts, texts=None, pad_to_multiple_of=None, images=None, context_images=None):
        return pair_inputs(self.tokenizer, contexts, texts, pad_to_multiple_of, self.device, add_bos=self.add_bos,
                           vision=self.vision, context_images=context_images, images=images)

    def next_logits(self, prompts: list[Prompt], normalize: bool = False, pad_to_multiple_of: int = None,
                    images: list[list] | None = None):
        """Returns the next-token logits for a batch of prompts"""
        inputs, _ = self._inputs(prompts, pad_to_multiple_of=pad_to_multiple_of, context_images=images)
        logits = self.model(**_forward_inputs(self.model, inputs)).logits[:, -1]  # (N, V)
        return torch.log_softmax(logits, dim=-1) if normalize else logits

    def next_tokens(self, prompts: list[Prompt], choices: list[str] = None,
                    top_k: int = 1, pad_to_multiple_of: int = None,
                    images: list[list] | None = None) -> list[tuple[str, ...]]:
        """Returns the most likely next tokens (optionally restricted) for a batch of prompts"""
        assert (not choices) or (top_k <= len(choices)), \
            f"{len(choices)} choices are insufficient for Top-{top_k} selection"

        logits = self.next_logits(prompts, pad_to_multiple_of=pad_to_multiple_of, images=images)

        # if only some tokens are allowed, mask out the other logits
        if choices:
            chc_ids = list(itertools.chain.from_iterable(self.tokenizer(choices, add_special_tokens=False)["input_ids"]))
            mask = torch.ones_like(logits[0], dtype=torch.bool)
            mask[chc_ids] = 0
            logits = torch.masked_fill(logits, mask, -torch.inf)

        token_ids = torch.topk(logits, top_k, dim=-1).indices
        next_tokens = self.tokenizer.batch_decode(token_ids.reshape(-1, 1), skip_special_tokens=True)
        return list(itertools.batched(next_tokens, top_k))

    def complete_prompts(self, prompts: list[Prompt], stop_tokens: list[str] = None, max_new_tokens: int = 200,
                         pad_to_multiple_of: int = None, images: list[list] | None = None, **kwargs) -> list[str]:
        """Completes a batch of prompts.

        Generation ends at the model's own end of sequence, or where the completion ends with one of the
        ``stop_tokens``, strings matched on the text as generated (``generate``'s ``stop_strings``); the stop stays
        in the completion. ``kwargs`` reach ``generate``.
        """
        stops = [*(stop_tokens or []), *(kwargs.pop("stop_strings", None) or [])]
        if stops:
            kwargs.update(stop_strings=stops, tokenizer=kwargs.get("tokenizer", self.tokenizer))
        inputs, _ = self._inputs(prompts, pad_to_multiple_of=pad_to_multiple_of, context_images=images)
        outputs = self.model.generate(**inputs,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      max_new_tokens=max_new_tokens, **kwargs)
        num_new_tokens = outputs.size(1) - inputs["input_ids"].size(1)
        return self.tokenizer.batch_decode(outputs[:, -num_new_tokens:], skip_special_tokens=True)

    def log_prob(self, texts: list[str], contexts: list[Prompt] = None,
                 ignore_idx: int = -100, pad_to_multiple_of: int = None, return_ppl: bool = False,
                 images: list[list] | None = None, context_images: list[list] | None = None):
        """Computes the log-prob of a batch of texts, each given its context.

        With no contexts a text follows the BOS token alone, on a tokenizer that adds one; a text with nothing
        before it cannot be scored and raises. ``images`` are the texts' images and ``context_images`` the contexts'.
        """
        contexts = contexts if contexts is not None else [""] * len(texts)
        inputs, is_text = self._inputs(contexts, texts, pad_to_multiple_of, images, context_images)
        _check_preceded(inputs, is_text)
        labels = torch.masked_fill(inputs["input_ids"], ~is_text, ignore_idx)
        loss = _token_loss(self.model, inputs, labels, ignore_idx)  # (N, S)

        if return_ppl:  # calculate perplexity
            return torch.exp(torch.sum(loss, dim=-1) / torch.sum(is_text, dim=-1))  # (N,)

        return -torch.sum(loss, dim=-1)  # (N,)

    def encode(self, texts: list[str], contexts: list[Prompt] = None, pad_to_multiple_of: int = None,
               images: list[list] | None = None, context_images: list[list] | None = None) -> torch.Tensor:
        """Encodes a batch of texts (each after its context, if any) as mean-pooled last-layer hidden states"""
        contexts = contexts if contexts is not None else [""] * len(texts)
        inputs, mask = self._inputs(contexts, texts, pad_to_multiple_of, images, context_images)  # mask: (N, S)
        indices = torch.cumsum(torch.sum(mask, dim=-1), dim=-1).tolist()  # (N,)

        # The logits are not needed: ask a model that can for one position's, not every position's
        keep = {"logits_to_keep": 1} if _takes_logits_to_keep(type(self.model)) else {}
        last_states = self.model(**_forward_inputs(self.model, inputs), output_hidden_states=True,
                                 **keep).hidden_states[-1]  # (N, S, H)

        chunks = torch.vsplit(last_states[mask], indices)[:-1]  # (Nc, H) * N
        embeddings = torch.stack([torch.mean(c, dim=0) for c in chunks], dim=0)  # (N, H)
        return embeddings


def collate_pair(batch: list[tuple], tokenizer, pad_to_multiple_of: int = None, ignore_idx: int = -100,
                 add_bos: bool | None = None, vision: ImageAdapter | None = None):
    """A batch of ``(context, text)`` items, or of ``(context, text, context_images, images)`` ones."""
    contexts, texts, *per_side = zip(*batch)
    context_images, images = per_side if per_side else (None, None)
    inputs, is_text = pair_inputs(tokenizer, list(contexts), list(texts), pad_to_multiple_of, add_bos=add_bos,
                                  vision=vision, context_images=context_images, images=images)
    return inputs, torch.masked_fill(inputs["input_ids"], ~is_text, ignore_idx)


class LogProbScorer(L.LightningModule):
    def __init__(self, llm_path: str, **model_args):
        super().__init__()
        self.model = LargeLangModel.load_model(llm_path, device_map=self.device, **model_args)

    def forward(self, batch):
        inputs, labels = batch
        return -torch.sum(_token_loss(self.model, inputs, labels), dim=-1)  # (N,)


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in output_dir, each containing the predictions of its respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

        # save `batch_indices` to get the information about the data index from prediction data
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))


def score_pairs(model: LogProbScorer, tokenizer, pairs, shard_dir: str, *, batch_size: int = 16,
                num_workers: int = 0, pad_to_multiple_of: int = None, add_bos: bool | None = None,
                vision: ImageAdapter | None = None, **trainer_kwargs) -> None:
    """Score a grid of (context, text) pairs on the GPUs, as shards in ``shard_dir``.

    ``pairs`` is a map-style dataset of (context, text) tuples — a
    ``ScoringGrid`` (``kcluster.core.grid``) such as ``PairQuestion`` — each
    optionally followed by the two sides' images, and each rank writes the
    log-prob of every text given its context to its own shard pair, which
    ``PointwiseMutualInfo.from_shards`` reassembles. ``add_bos`` is the
    engine's BOS rule for string contexts; with images, ``vision``
    (``ImageAdapter.load``) shows them to the model. ``trainer_kwargs`` reach
    ``L.Trainer``: ``accelerator`` (``"gpu"`` unless given), ``devices``
    (every GPU unless given) and ``strategy``. Returns once every rank has
    written its shard, True in the first process: what comes after the
    scores (the reassembly, the KC models) is its to do alone.
    """
    dl = DataLoader(pairs, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers,
                    collate_fn=partial(collate_pair, tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of,
                                       add_bos=add_bos, vision=vision))
    trainer = L.Trainer(callbacks=[CustomWriter(output_dir=shard_dir, write_interval="epoch")], logger=False,
                        **{"accelerator": "gpu", "devices": -1, **trainer_kwargs})
    trainer.predict(model, dataloaders=dl, return_predictions=False)
    trainer.strategy.barrier()
    return trainer.is_global_zero
