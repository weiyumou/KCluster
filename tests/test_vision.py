"""The image adapter, offline: a stub processor stands in for Gemma 4's, with
the hooks the adapter uses (``replace_image_token``, ``image_processor``,
``create_mm_token_type_ids``), and a word-level fast tokenizer with the image
tokens as added tokens plays the model's tokenizer. What is pinned is the
layout the real processor was verified to produce: each image's place becomes
a bracketed run of soft tokens, the text mask covers the text's own tokens
only, and the images reach the model in reading order."""

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from kcluster.engine.local import LargeLangModel, collate_pair, pair_inputs, token_count
from kcluster.engine.vision import ImageAdapter, data_uri, expand_places, open_image

BOI, IMG, EOI = "<boi>", "<img>", "<eoi>"


class StubProcessor:
    """Gemma-4-shaped: two soft tokens per image (the real one has up to 280), bracketed."""

    image_token = IMG
    model_input_names = ["input_ids", "attention_mask", "mm_token_type_ids"]
    unused_input_names = ["num_soft_tokens_per_image"]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMG)
        self.sizes = []

        class ImageProcessor:
            max_soft_tokens = 2

            def __call__(_, images, return_tensors=None):
                self.sizes.append([im.size for im in images])
                n = len(images)
                return {"pixel_values": torch.arange(n, dtype=torch.float).reshape(n, 1),
                        "num_soft_tokens_per_image": [2] * n}

        self.image_processor = ImageProcessor()

    def replace_image_token(self, inputs, image_idx):
        return BOI + IMG * inputs["num_soft_tokens_per_image"][image_idx] + EOI

    def create_mm_token_type_ids(self, input_ids):
        return [[1 if t == self.image_token_id else 0 for t in row] for row in input_ids]


@pytest.fixture
def tokenizer(make_tokenizer, chat_template):
    tok = make_tokenizer(["alpha", "beta", "gamma", "[figure]"], chat_template=chat_template)
    tok.add_special_tokens({"additional_special_tokens": [BOI, IMG, EOI]})
    return tok


@pytest.fixture
def vision(tokenizer):
    return ImageAdapter(StubProcessor(tokenizer))


@pytest.fixture
def images(tmp_path):
    paths = []
    for name, size in (("a", (8, 6)), ("b", (5, 5)), ("c", (3, 4))):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", size, "white").save(path)
        paths.append(str(path))
    return paths


def test_open_image_takes_every_source_form(images):
    path = images[0]
    from_path = open_image(path)
    with open(path, "rb") as f:
        raw = f.read()
    assert from_path.mode == "RGB" and from_path.size == (8, 6)
    assert open_image(raw).size == (8, 6)
    assert open_image(data_uri(path)).size == (8, 6)
    assert open_image(Image.new("L", (2, 2))).mode == "RGB"
    assert data_uri(path).startswith("data:image/png;base64,")
    assert data_uri(data_uri(path)) == data_uri(path)
    with pytest.raises(TypeError, match="an image is"):
        open_image(3)


def test_expand_places_consumes_replacements_in_order():
    assert expand_places("x <i> y <i>", "<i>", iter(["<1>", "<2>"])) == "x <1> y <2>"
    assert expand_places("no places", "<i>", iter([])) == "no places"


def test_prepare_expands_every_segment_and_orders_images_by_reading(vision, images):
    rows = [[(f"alpha {IMG}", images[:1]), (f"beta {IMG} gamma", images[1:2])],
            [("alpha", []), (f"{IMG} beta", images[2:])]]
    expanded, inputs = vision.prepare(rows)
    assert expanded == [[f"alpha {BOI}{IMG}{IMG}{EOI}", f"beta {BOI}{IMG}{IMG}{EOI} gamma"],
                        ["alpha", f"{BOI}{IMG}{IMG}{EOI} beta"]]
    # Row 0's context image, then its text image, then row 1's: the order the
    # model fills the image-token blocks of the batch in
    assert vision.processor.sizes == [[(8, 6), (5, 5), (3, 4)]]
    assert inputs["pixel_values"].tolist() == [[0.0], [1.0], [2.0]]
    assert "num_soft_tokens_per_image" not in inputs


def test_prepare_without_images_touches_nothing(vision):
    expanded, inputs = vision.prepare([[("alpha", []), ("beta", [])]])
    assert expanded == [["alpha", "beta"]] and inputs == {} and vision.processor.sizes == []


def test_prepare_rejects_a_place_count_mismatch(vision, images):
    with pytest.raises(ValueError, match=r"places 1 image\(s\).*but 2 were given"):
        vision.prepare([[(f"alpha {IMG}", images[:2])]])
    with pytest.raises(ValueError, match=r"places 1 image\(s\).*but 0 were given"):
        vision.prepare([[(f"alpha {IMG}", [])]])


def test_pair_inputs_score_the_texts_own_tokens_only(tokenizer, vision, images):
    contexts, texts = [f"alpha {IMG}", "alpha"], [f" beta {IMG} gamma", " beta"]
    inputs, is_text = pair_inputs(tokenizer, contexts, texts, vision=vision,
                                  context_images=[images[:1], []], images=[images[1:2], []])
    ids = inputs["input_ids"]
    soft = ids == vision.image_token_id
    decode = lambda row, mask: tokenizer.decode(row[mask]).split()  # noqa: E731
    # The whole text is marked, except its soft tokens: the brackets stay text
    assert decode(ids[0], is_text[0]) == ["beta", BOI, EOI, "gamma"]
    assert decode(ids[1], is_text[1]) == ["beta"]
    assert not (is_text & soft).any()
    # The model's extra inputs ride along, and the modality mask marks the soft tokens
    assert inputs["pixel_values"].shape == (2, 1)
    assert torch.equal(inputs["mm_token_type_ids"], soft.long())
    # Left-padded: the text-only row is padded in front
    assert ids[1, 0] == tokenizer.pad_token_id and inputs["attention_mask"][1, 0] == 0


def test_a_conversation_places_images_through_the_template(tokenizer, vision, images):
    # The template writes the content as it is, so the image token in a message is the image's place
    messages = [{"role": "user", "content": f"alpha {IMG}"}]
    inputs, is_text = pair_inputs(tokenizer, [messages], [" beta"], vision=vision, context_images=[images[:1]])
    soft = inputs["input_ids"] == vision.image_token_id
    assert soft.sum() == 2 and inputs["pixel_values"].shape == (1, 1)
    assert tokenizer.decode(inputs["input_ids"][0][is_text[0]]) == "beta"


def test_pair_inputs_without_images_are_the_plain_pair_encoding(tokenizer, vision):
    plain, plain_mask = pair_inputs(tokenizer, ["alpha"], [" beta gamma"])
    same, same_mask = pair_inputs(tokenizer, ["alpha"], [" beta gamma"], vision=vision, images=[[]],
                                  context_images=[[]])
    assert torch.equal(plain["input_ids"], same["input_ids"]) and torch.equal(plain_mask, same_mask)
    assert set(same.keys()) == {"input_ids", "attention_mask"}


def test_images_without_a_processor_are_refused(tokenizer, images):
    with pytest.raises(ValueError, match="takes no images"):
        pair_inputs(tokenizer, ["alpha [figure]"], [" beta"], context_images=[images[:1]], images=[[]])
    with pytest.raises(ValueError, match="takes no images"):
        pair_inputs(tokenizer, ["alpha [figure]"], images=[images[:1]])
    # A text-only model reads a bank's marker as the characters it is
    inputs, _ = pair_inputs(tokenizer, ["alpha [figure]"], images=[[]])
    assert tokenizer.decode(inputs["input_ids"][0]) == "alpha [figure]"


def test_prompts_alone_show_images_too(tokenizer, vision, images):
    inputs, is_text = pair_inputs(tokenizer, [f"alpha {IMG}", "beta"], vision=vision, context_images=[images[:1], []])
    soft = inputs["input_ids"] == vision.image_token_id
    assert soft.sum() == 2 and inputs["pixel_values"].shape == (1, 1) and not is_text.any()
    assert torch.equal(inputs["mm_token_type_ids"], soft.long())


def test_collate_pair_takes_items_with_images(tokenizer, vision, images):
    batch = [("alpha", f" beta {IMG}", [], images[:1]), ("alpha", " gamma", [], [])]
    inputs, labels = collate_pair(batch, tokenizer=tokenizer, vision=vision)
    soft = inputs["input_ids"] == vision.image_token_id
    assert (labels[soft] == -100).all()
    assert labels[0].tolist().count(-100) == labels.size(1) - 3  # beta, <boi>, <eoi>
    # Two-tuples still collate as before
    inputs, labels = collate_pair([("alpha", " beta"), ("alpha", " gamma")], tokenizer=tokenizer)
    assert set(inputs.keys()) == {"input_ids", "attention_mask"} and (labels != -100).sum() == 2


def test_token_count_bounds_an_image_by_the_processors_budget(tokenizer, vision):
    assert vision.tokens_per_image == 4  # two soft tokens and the two that bracket them
    assert token_count(tokenizer, "alpha", " beta") == 2
    # Each place is one token in the string and at most four when shown: three more
    assert token_count(tokenizer, f"alpha {IMG}", f" beta {IMG}", vision=vision) == 4 + 2 * 3
    assert token_count(tokenizer, "alpha", " beta", vision=vision) == 2


def test_the_engine_names_its_image_token(tokenizer, vision):
    llm = LargeLangModel.__new__(LargeLangModel)
    llm.tokenizer, llm.add_bos = tokenizer, None
    assert llm.image_token is None  # text-only: no processor
    llm.vision = vision
    assert llm.image_token == IMG


def test_adapter_load_is_none_without_a_processor_config(tmp_path):
    assert ImageAdapter.load(str(tmp_path)) is None


def test_image_bytes_reads_every_source_form_without_decoding(images):
    """``image_bytes`` is what an engine sends over the wire: the file as it is and its MIME type."""
    from kcluster.engine.vision import image_bytes

    path = images[0]
    with open(path, "rb") as f:
        raw = f.read()
    assert image_bytes(path) == (raw, "image/png")
    assert image_bytes(raw) == (raw, "image/png")  # sniffed from the magic
    assert image_bytes(data_uri(path)) == (raw, "image/png")
    data, mime = image_bytes(Image.new("L", (2, 2)))
    assert mime == "image/png" and data.startswith(b"\x89PNG")
    assert image_bytes(b"\xff\xd8\xff" + b"\x00" * 4)[1] == "image/jpeg"
    assert image_bytes(b"RIFF\x00\x00\x00\x00WEBP")[1] == "image/webp"
    with pytest.raises(ValueError, match="no image format"):
        image_bytes(b"not an image")
    with pytest.raises(ValueError, match="not an image file"):
        image_bytes("notes.txt")
