"""Local (in-process) execution substrate: a HuggingFace causal-LM wrapper and
the Lightning batch scorer that shards large scoring jobs across GPUs."""

import itertools
import os
from functools import partial

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import BasePredictionWriter
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def _pair_inputs(tokenizer, contexts, texts, pad_to_multiple_of: int = None, device=None):
    """Model inputs for a batch of (context, text) pairs, and the mask of each text's tokens.

    The tokenizer marks the second sequence of a pair in ``token_type_ids``;
    that marker is what lets scoring and pooling cover the text alone, leaving
    out the context and the padding.
    """
    inputs = tokenizer(text=contexts, text_pair=texts,
                       return_tensors="pt", return_token_type_ids=True,
                       padding=True, pad_to_multiple_of=pad_to_multiple_of)
    if device is not None:
        inputs = inputs.to(device)
    return inputs, inputs.pop("token_type_ids").bool()


def _token_loss(model, inputs, labels, ignore_idx: int = -100):
    """Per-token cross-entropy of each label given the tokens before it, 0 where ignored."""
    logits = model(**inputs).logits.transpose(-1, -2)  # (N, V, S)
    return F.cross_entropy(logits[..., :-1], labels[:, 1:], reduction="none", ignore_index=ignore_idx)  # (N, S)


class LargeLangModel:
    """Encapsulates a downloaded LLM and its tokenizer"""

    def __init__(self, model_path: str, device: str = "auto", **model_args):
        # Load tokenizer
        self.tokenizer = LargeLangModel.load_tokenizer(model_path)

        # Load model
        self.model = LargeLangModel.load_model(model_path, device_map=device, **model_args).eval()

    @staticmethod
    def load_tokenizer(path: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side="left", **kwargs)
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

    def next_logits(self, prompts: list[str], normalize: bool = False, pad_to_multiple_of: int = None):
        """Returns the next-token logits for a batch of prompts"""
        inputs = self.tokenizer(prompts, return_tensors="pt",
                                padding=True, pad_to_multiple_of=pad_to_multiple_of).to(self.device)
        logits = self.model(**inputs).logits[:, -1]  # (N, V)
        return torch.log_softmax(logits, dim=-1) if normalize else logits

    def next_tokens(self, prompts: list[str], choices: list[str] = None,
                    top_k: int = 1, pad_to_multiple_of: int = None) -> list[tuple[str, ...]]:
        """Returns the most likely next tokens (optionally restricted) for a batch of prompts"""
        assert (not choices) or (top_k <= len(choices)), \
            f"{len(choices)} choices are insufficient for Top-{top_k} selection"

        logits = self.next_logits(prompts, pad_to_multiple_of=pad_to_multiple_of)

        # if only some tokens are allowed, mask out the other logits
        if choices:
            chc_ids = list(itertools.chain.from_iterable(self.tokenizer(choices)["input_ids"]))
            mask = torch.ones_like(logits[0], dtype=torch.bool)
            mask[chc_ids] = 0
            logits = torch.masked_fill(logits, mask, -torch.inf)

        token_ids = torch.topk(logits, top_k, dim=-1).indices
        next_tokens = self.tokenizer.batch_decode(token_ids.reshape(-1, 1), skip_special_tokens=True)
        return list(itertools.batched(next_tokens, top_k))

    def complete_prompts(self, prompts: list[str], stop_tokens: list[str] = None, max_new_tokens: int = 200,
                         pad_to_multiple_of: int = None, **kwargs) -> list[str]:
        """Completes a batch of prompts"""
        if stop_ids := stop_tokens:
            stop_ids = list(itertools.chain.from_iterable(self.tokenizer(stop_tokens)["input_ids"]))
        inputs = self.tokenizer(prompts, return_tensors="pt",
                                padding=True, pad_to_multiple_of=pad_to_multiple_of).to(self.device)
        outputs = self.model.generate(**inputs,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      eos_token_id=stop_ids,
                                      max_new_tokens=max_new_tokens, **kwargs)
        num_new_tokens = outputs.size(1) - inputs["input_ids"].size(1)
        return self.tokenizer.batch_decode(outputs[:, -num_new_tokens:], skip_special_tokens=True)

    def log_prob(self, texts: list[str], contexts: list[str] = None,
                 ignore_idx: int = -100, pad_to_multiple_of: int = None, return_ppl: bool = False):
        """Computes the log-prob of a batch of texts, optionally given contexts"""
        if not contexts:
            contexts = [self.pad_token] * len(texts)
        inputs, is_text = _pair_inputs(self.tokenizer, contexts, texts, pad_to_multiple_of, self.device)
        labels = torch.masked_fill(inputs["input_ids"], ~is_text, ignore_idx)
        loss = _token_loss(self.model, inputs, labels, ignore_idx)  # (N, S)

        if return_ppl:  # calculate perplexity
            return torch.exp(torch.sum(loss, dim=-1) / torch.sum(is_text, dim=-1))  # (N,)

        return -torch.sum(loss, dim=-1)  # (N,)

    def encode(self, texts: list[str], contexts: list[str] = None, pad_to_multiple_of: int = None) -> torch.Tensor:
        """Encodes a batch of texts (with optional contexts) using mean-pooled last-layer hidden states"""
        if not contexts:
            contexts = [self.pad_token] * len(texts)
        inputs, mask = _pair_inputs(self.tokenizer, contexts, texts, pad_to_multiple_of, self.device)  # mask: (N, S)
        indices = torch.cumsum(torch.sum(mask, dim=-1), dim=-1).tolist()  # (N,)

        last_states = self.model(**inputs, output_hidden_states=True).hidden_states[-1]  # (N, S, H)

        chunks = torch.vsplit(last_states[mask], indices)[:-1]  # (Nc, H) * N
        embeddings = torch.stack([torch.mean(c, dim=0) for c in chunks], dim=0)  # (N, H)
        return embeddings


def collate_pair(batch: list[tuple[str, str]], tokenizer,
                 pad_to_multiple_of: int = None, ignore_idx: int = -100):
    contexts, texts = list(zip(*batch))
    inputs, is_text = _pair_inputs(tokenizer, contexts, texts, pad_to_multiple_of)
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
                num_workers: int = 0, pad_to_multiple_of: int = None, **trainer_kwargs) -> None:
    """Score a grid of (context, text) pairs on the GPUs, as shards in ``shard_dir``.

    ``pairs`` is a map-style dataset of (context, text) tuples — a
    ``ScoringGrid`` such as ``PairQuestion`` — and each rank writes the
    log-prob of every text given its context to its own shard pair, which
    ``PointwiseMutualInfo.from_shards`` reassembles. ``trainer_kwargs`` reach
    ``L.Trainer``: ``accelerator`` (``"gpu"`` unless given), ``devices`` (every
    GPU unless given) and ``strategy``. Returns once every rank has written
    its shard, True in the first process: what comes after the scores (the
    reassembly, the KC models) is its to do alone.
    """
    dl = DataLoader(pairs, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers,
                    collate_fn=partial(collate_pair, tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of))
    trainer = L.Trainer(callbacks=[CustomWriter(output_dir=shard_dir, write_interval="epoch")], logger=False,
                        **{"accelerator": "gpu", "devices": -1, **trainer_kwargs})
    trainer.predict(model, dataloaders=dl, return_predictions=False)
    trainer.strategy.barrier()
    return trainer.is_global_zero
