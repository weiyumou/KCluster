"""Figures for an image-taking model: the engine's side of the figures contract.

``ImageAdapter`` wraps a HuggingFace processor (``AutoProcessor``) and turns
the ``[figure]`` markers of a prompt into the model's image tokens, and the
figure files into the pixel inputs that go with them, by the processor's own
rules: ``replace_image_token`` for the expansion (Gemma 4: ``<|image>``, one
placeholder per soft token, ``<image|>``), ``image_processor`` for the
pixels, ``create_mm_token_type_ids`` for the modality mask Gemma 4 attends
with. The processor's ``__call__`` is not used: it expands markers in ``text``
only, never in ``text_pair``, and the scoring methods need both sides of a
(context, text) pair to show figures. Built this way, the inputs match what
the processor would produce for the same text token for token (verified
against Gemma 4's), with the pair mask on top.

The soft tokens are never scored or pooled: ``soft_token_mask`` takes them
out of the text mask, so a figure is context in the marginal and in the
conditional alike and the PMI subtracts it out. The tokens that bracket a
figure stay text.

A figure source is a path, raw image bytes, a ``data:`` URI (what the HTTP
client sends), or a PIL image. Pillow is imported on use, so the module
imports without the ``multimodal`` extra.
"""

import base64
import io
import mimetypes
import os
from functools import lru_cache

from kcluster.core.prompts import FIGURE_MARKER


def open_figure(source):
    """A figure as an RGB PIL image, from a path, bytes, a ``data:`` URI or a PIL image."""
    from PIL import Image

    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif isinstance(source, str) and source.startswith("data:"):
        data = base64.b64decode(source.split(",", 1)[1])
    elif isinstance(source, str):
        return _open_file(source)
    else:
        raise TypeError(f"a figure is a path, bytes, a data: URI or a PIL image, not {type(source).__name__}")
    with Image.open(io.BytesIO(data)) as img:
        return img.convert("RGB")


@lru_cache(maxsize=1024)
def _open_file(path: str):
    # A scoring grid shows each figure once per pair it is in; decode it once per process
    from PIL import Image

    with Image.open(path) as img:
        return img.convert("RGB")


def data_uri(source) -> str:
    """A figure as a ``data:`` URI, for sending over HTTP (a path or bytes are sent as they are; a PIL image as PNG)."""
    if isinstance(source, str) and source.startswith("data:"):
        return source
    if isinstance(source, str):
        mime = mimetypes.guess_type(source)[0] or "application/octet-stream"
        with open(source, "rb") as f:
            data = f.read()
    elif isinstance(source, (bytes, bytearray)):
        mime, data = "application/octet-stream", bytes(source)
    else:  # a PIL image
        buf = io.BytesIO()
        source.save(buf, format="PNG")
        mime, data = "image/png", buf.getvalue()
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def expand_markers(text: str, replacements) -> str:
    """``text`` with each figure marker replaced by the next of ``replacements`` (an iterator)."""
    head, *rest = text.split(FIGURE_MARKER)
    return head + "".join(next(replacements) + part for part in rest)


class ImageAdapter:
    """Figures in, a model's image tokens and pixel inputs out, by a processor's rules."""

    def __init__(self, processor):
        self.processor = processor
        self.image_token = processor.image_token
        self.image_token_id = getattr(processor, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = processor.tokenizer.convert_tokens_to_ids(self.image_token)

    @classmethod
    def load(cls, model_path: str, **kwargs) -> "ImageAdapter | None":
        """The adapter for a downloaded model, or None for one that takes no images (no processor config)."""
        if not os.path.isfile(os.path.join(model_path, "processor_config.json")):
            return None
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, **kwargs)
        if getattr(processor, "image_token", None) is None or not hasattr(processor, "image_processor"):
            return None
        return cls(processor)

    @property
    def soft_tokens_per_figure(self) -> int | None:
        """The processor's image budget in tokens per figure, when it names one."""
        return getattr(self.processor.image_processor, "max_soft_tokens", None)

    def prepare(self, rows: list[list[tuple[str, list]]]) -> tuple[list[list[str]], dict]:
        """Expand the markers of every segment of every row, and process their figures.

        ``rows[i]`` is a row's segments in order — ``[(context, context_figures),
        (text, text_figures)]`` for a scoring pair — each a text and the figures
        its markers show. Returns the expanded segment texts, row by row, and the
        pixel inputs for the model (empty when no row shows a figure). Figures
        are processed in reading order, which is the order the model matches
        them to the image-token blocks of a left-padded batch.
        """
        sources = []
        for row in rows:
            for text, figures in row:
                if (n := text.count(FIGURE_MARKER)) != len(figures):
                    raise ValueError(f"a text shows {n} figure marker(s) but {len(figures)} figure(s) were given: "
                                     f"{text[:80]!r}")
                sources.extend(figures)
        if not sources:
            return [[text for text, _ in row] for row in rows], {}

        inputs = self.processor.image_processor([open_figure(s) for s in sources], return_tensors="pt")
        replacements = iter(self.processor.replace_image_token(inputs, i) for i in range(len(sources)))
        expanded = [[expand_markers(text, replacements) for text, _ in row] for row in rows]
        unused = set(self.processor.unused_input_names)
        return expanded, {k: v for k, v in inputs.items() if k not in unused}

    def mm_token_type_ids(self, input_ids):
        """The modality mask of a batch of token ids, when the model's forward takes one; else None."""
        if "mm_token_type_ids" not in self.processor.model_input_names:
            return None
        import torch

        return torch.tensor(self.processor.create_mm_token_type_ids(input_ids.tolist()), device=input_ids.device)

    def soft_token_mask(self, input_ids):
        """True at the image soft tokens: the positions a figure's features fill, which are never scored."""
        return input_ids == self.image_token_id
