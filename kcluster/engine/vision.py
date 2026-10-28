"""Images for a model that takes them: the engine's side of a multimodal prompt.

A prompt places an image with the model's own image token (``image_token``;
Gemma 4's ``<|image|>``): in a string, once where the image goes; in a
conversation, as an image part (``{"type": "image"}``) in a message's
content, which the chat template writes as that token, or as the token in
the content's text. The images themselves are given beside the prompt, in
reading order, one source per place: a path, raw bytes, a ``data:`` URI
(what the HTTP client sends), or a PIL image.

``ImageAdapter`` wraps a HuggingFace processor (``AutoProcessor``) and turns
each place into the model's run of image tokens, and the images into the
pixel inputs that go with them, by the processor's own rules:
``replace_image_token`` for the expansion (Gemma 4: ``<|image>``, one
placeholder per soft token, ``<image|>``), ``image_processor`` for the
pixels, ``create_mm_token_type_ids`` for the modality mask Gemma 4 attends
with. The processor's ``__call__`` is not used: the engine reads a context
and its text as one string and finds the text by character offsets
(``kcluster.engine.local``), so the expansion is done on the strings before
they are tokenized. Built this way, the inputs match what the processor
would produce for the same text token for token (verified against Gemma 4's).

The soft tokens are never scored or pooled: ``soft_token_mask`` takes them
out of a text's mask, so an image is context wherever it stands, and the
tokens that bracket it stay text.

Pillow is imported on use, so the module imports without the ``multimodal``
extra.
"""

import base64
import io
import mimetypes
import os
import warnings
from collections.abc import Iterator
from functools import lru_cache


def open_image(source):
    """An image as an RGB PIL image, from a path, bytes, a ``data:`` URI or a PIL image."""
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
        raise TypeError(f"an image is a path, bytes, a data: URI or a PIL image, not {type(source).__name__}")
    with Image.open(io.BytesIO(data)) as img:
        return img.convert("RGB")


@lru_cache(maxsize=1024)
def _open_file(path: str):
    # A scoring grid shows each image once per pair it is in; decode it once per process
    from PIL import Image

    with Image.open(path) as img:
        return img.convert("RGB")


def data_uri(source) -> str:
    """An image as a ``data:`` URI, for sending over HTTP (a path or bytes are sent as they are; a PIL image as PNG)."""
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


def expand_places(text: str, token: str, replacements: Iterator[str]) -> str:
    """``text`` with each occurrence of ``token`` replaced by the next of ``replacements``."""
    head, *rest = text.split(token)
    return head + "".join(next(replacements) + part for part in rest)


class ImageAdapter:
    """Images in, a model's image tokens and pixel inputs out, by a processor's rules."""

    def __init__(self, processor):
        self.processor = processor
        self.image_token = processor.image_token
        self.image_token_id = getattr(processor, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = processor.tokenizer.convert_tokens_to_ids(self.image_token)

    @classmethod
    def load(cls, model_path: str, **kwargs) -> "ImageAdapter | None":
        """The adapter for a downloaded model, or None for one that takes no images (no processor config).

        A processor that cannot be loaded for want of the ``multimodal`` extra leaves the model text-only, with a
        warning, rather than failing a text-only job.
        """
        if not os.path.isfile(os.path.join(model_path, "processor_config.json")):
            return None
        from transformers import AutoProcessor

        try:
            processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, **kwargs)
        except ImportError as e:
            warnings.warn(f"{model_path} takes images, but its processor could not be loaded ({e}); install the "
                          "multimodal extra to show it any. Serving it text-only.", stacklevel=2)
            return None
        if getattr(processor, "image_token", None) is None or not hasattr(processor, "image_processor"):
            return None
        return cls(processor)

    @property
    def tokens_per_image(self) -> int | None:
        """At most how many tokens an image takes in place of its token: the processor's soft-token budget plus
        the two that bracket it, when the processor names a budget; None otherwise."""
        budget = getattr(self.processor.image_processor, "max_soft_tokens", None)
        if budget is None:
            budget = getattr(self.processor, "image_seq_length", None)
        return None if budget is None else int(budget) + 2

    def count(self, text: str) -> int:
        """How many images a text places."""
        return text.count(self.image_token)

    def prepare(self, rows: list[list[tuple[str, list]]]) -> tuple[list[list[str]], dict]:
        """Expand the image places of every segment of every row, and process their images.

        ``rows[i]`` is a row's segments in order — ``[(context, context_images),
        (text, images)]`` for a scoring pair — each a text and the images at
        its places. Returns the expanded segment texts, row by row, and the
        pixel inputs for the model (empty when no row shows an image). Images
        are processed in reading order, which is the order the model matches
        them to the image-token blocks of a left-padded batch.
        """
        sources = []
        for row in rows:
            for text, images in row:
                if (n := self.count(text)) != len(images):
                    raise ValueError(f"a prompt places {n} image(s) ({self.image_token}) but {len(images)} were "
                                     f"given: {text[:80]!r}")
                sources.extend(images)
        if not sources:
            return [[text for text, _ in row] for row in rows], {}

        inputs = self.processor.image_processor([open_image(s) for s in sources], return_tensors="pt")
        replacements = iter(self.processor.replace_image_token(inputs, i) for i in range(len(sources)))
        expanded = [[expand_places(text, self.image_token, replacements) for text, _ in row] for row in rows]
        unused = set(getattr(self.processor, "unused_input_names", ()))
        return expanded, {k: v for k, v in inputs.items() if k not in unused}

    def mm_token_type_ids(self, input_ids):
        """The modality mask of a batch of token ids, when the model's forward takes one; else None."""
        if "mm_token_type_ids" not in self.processor.model_input_names:
            return None
        import torch

        return torch.tensor(self.processor.create_mm_token_type_ids(input_ids.tolist()), device=input_ids.device)

    def soft_token_mask(self, input_ids):
        """True at the image soft tokens: the positions an image's features fill, which are never scored."""
        return input_ids == self.image_token_id
