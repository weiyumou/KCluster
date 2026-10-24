"""The figure adapter, offline: a stub processor stands in for Gemma 4's, with
the hooks the adapter uses (``replace_image_token``, ``image_processor``,
``create_mm_token_type_ids``), and a word-level fast tokenizer with the image
tokens as added tokens plays the model's tokenizer. What is pinned is the
layout the real processor was verified to produce: each marker becomes a
bracketed run of soft tokens, the pair mask covers the text's hard tokens
only, and the figures reach the model in reading order."""

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from kcluster.engine.local import _pair_inputs, _prompt_inputs, collate_pair
from kcluster.engine.vision import ImageAdapter, data_uri, expand_markers, open_figure

BOI, IMG, EOI = "<boi>", "<img>", "<eoi>"


class StubProcessor:
    """Gemma-4-shaped: two soft tokens per figure (the real one has 256-280), bracketed."""

    image_token = IMG
    model_input_names = ["input_ids", "attention_mask", "mm_token_type_ids"]
    unused_input_names = ["num_soft_tokens_per_image"]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMG)
        self.sizes = []

    def image_processor(self, images, return_tensors=None):
        self.sizes.append([im.size for im in images])
        n = len(images)
        return {"pixel_values": torch.arange(n, dtype=torch.float).reshape(n, 1),
                "num_soft_tokens_per_image": [2] * n}

    def replace_image_token(self, inputs, image_idx):
        return BOI + IMG * inputs["num_soft_tokens_per_image"][image_idx] + EOI

    def create_mm_token_type_ids(self, input_ids):
        return [[1 if t == self.image_token_id else 0 for t in row] for row in input_ids]


@pytest.fixture
def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["[PAD]", "[UNK]", "alpha", "beta", "gamma", "[figure]"]
    raw = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", padding_side="left")
    tok.add_special_tokens({"additional_special_tokens": [BOI, IMG, EOI]})
    return tok


@pytest.fixture
def adapter(tokenizer):
    return ImageAdapter(StubProcessor(tokenizer))


@pytest.fixture
def figures(tmp_path):
    paths = []
    for name, size in (("a", (8, 6)), ("b", (5, 5)), ("c", (3, 4))):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", size, "white").save(path)
        paths.append(str(path))
    return paths


def test_open_figure_takes_every_source_form(figures):
    path = figures[0]
    from_path = open_figure(path)
    with open(path, "rb") as f:
        raw = f.read()
    assert from_path.mode == "RGB" and from_path.size == (8, 6)
    assert open_figure(raw).size == (8, 6)
    assert open_figure(data_uri(path)).size == (8, 6)
    assert open_figure(Image.new("L", (2, 2))).mode == "RGB"
    assert data_uri(path).startswith("data:image/png;base64,")
    assert data_uri(data_uri(path)) == data_uri(path)
    with pytest.raises(TypeError, match="a figure is"):
        open_figure(3)


def test_expand_markers_consumes_replacements_in_order():
    assert expand_markers("x [figure] y [figure]", iter(["<1>", "<2>"])) == "x <1> y <2>"
    assert expand_markers("no markers", iter([])) == "no markers"


def test_prepare_expands_every_segment_and_orders_figures_by_reading(adapter, figures):
    rows = [[("alpha [figure]", figures[:1]), ("beta [figure] gamma", figures[1:2])],
            [("alpha", []), ("[figure] beta", figures[2:])]]
    expanded, inputs = adapter.prepare(rows)
    assert expanded == [[f"alpha {BOI}{IMG}{IMG}{EOI}", f"beta {BOI}{IMG}{IMG}{EOI} gamma"],
                        ["alpha", f"{BOI}{IMG}{IMG}{EOI} beta"]]
    # Row 0's context figure, then its text figure, then row 1's: the order the
    # model fills the image-token blocks of the batch in
    assert adapter.processor.sizes == [[(8, 6), (5, 5), (3, 4)]]
    assert inputs["pixel_values"].tolist() == [[0.0], [1.0], [2.0]]
    assert "num_soft_tokens_per_image" not in inputs


def test_prepare_without_figures_touches_nothing(adapter):
    expanded, inputs = adapter.prepare([[("alpha", []), ("beta", [])]])
    assert expanded == [["alpha", "beta"]] and inputs == {} and adapter.processor.sizes == []


def test_prepare_rejects_a_marker_count_mismatch(adapter, figures):
    with pytest.raises(ValueError, match="shows 1 figure marker.*2 figure"):
        adapter.prepare([[("alpha [figure]", figures[:2])]])


def test_pair_inputs_score_the_texts_hard_tokens_only(tokenizer, adapter, figures):
    contexts, texts = ["alpha [figure]", "alpha"], ["beta [figure] gamma", "beta"]
    inputs, is_text = _pair_inputs(tokenizer, contexts, texts, adapter=adapter,
                                   context_figures=[figures[:1], []], figures=[figures[1:2], []])
    ids = inputs["input_ids"]
    soft = ids == adapter.image_token_id
    decode = lambda row, mask: tokenizer.decode(row[mask]).split()  # noqa: E731
    # The whole text is marked, except its soft tokens: the brackets stay text
    assert decode(ids[0], is_text[0]) == ["beta", BOI, EOI, "gamma"]
    assert decode(ids[1], is_text[1]) == ["beta"]
    assert not (is_text & soft).any()
    # The model's extra inputs ride along, and the modality mask marks the soft tokens
    assert inputs["pixel_values"].shape == (2, 1)
    assert torch.equal(inputs["mm_token_type_ids"], soft.long())
    assert "token_type_ids" not in inputs
    # Left-padded: the text-only row is padded in front
    assert ids[1, 0] == tokenizer.pad_token_id and inputs["attention_mask"][1, 0] == 0


def test_pair_inputs_without_figures_are_the_plain_pair_encoding(tokenizer, adapter):
    plain, plain_mask = _pair_inputs(tokenizer, ["alpha"], ["beta gamma"])
    same, same_mask = _pair_inputs(tokenizer, ["alpha"], ["beta gamma"], adapter=adapter, figures=[[]],
                                   context_figures=[[]])
    assert torch.equal(plain["input_ids"], same["input_ids"]) and torch.equal(plain_mask, same_mask)
    assert set(same.keys()) == {"input_ids", "attention_mask"}


def test_figures_without_an_adapter_are_refused(tokenizer, figures):
    with pytest.raises(ValueError, match="takes no images"):
        _pair_inputs(tokenizer, ["alpha [figure]"], ["beta"], context_figures=[figures[:1]], figures=[[]])
    with pytest.raises(ValueError, match="takes no images"):
        _prompt_inputs(tokenizer, ["alpha [figure]"], figures=[figures[:1]])


def test_prompt_inputs_show_figures_too(tokenizer, adapter, figures):
    inputs = _prompt_inputs(tokenizer, ["alpha [figure]", "beta"], adapter=adapter, figures=[figures[:1], []])
    soft = inputs["input_ids"] == adapter.image_token_id
    assert soft.sum() == 2 and inputs["pixel_values"].shape == (1, 1)
    assert torch.equal(inputs["mm_token_type_ids"], soft.long())


def test_collate_pair_takes_items_with_figures(tokenizer, adapter, figures):
    batch = [("alpha", "beta [figure]", [], figures[:1]), ("alpha", "gamma", [], [])]
    inputs, labels = collate_pair(batch, tokenizer=tokenizer, adapter=adapter)
    soft = inputs["input_ids"] == adapter.image_token_id
    assert (labels[soft] == -100).all()
    assert labels[0].tolist().count(-100) == labels.size(1) - 3  # beta, <boi>, <eoi>
    # Two-tuples still collate as before
    inputs, labels = collate_pair([("alpha", "beta"), ("alpha", "gamma")], tokenizer=tokenizer)
    assert set(inputs.keys()) == {"input_ids", "attention_mask"} and (labels != -100).sum() == 2


def test_adapter_load_is_none_without_a_processor_config(tmp_path):
    assert ImageAdapter.load(str(tmp_path)) is None
