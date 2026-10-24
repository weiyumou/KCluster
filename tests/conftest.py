"""Fixtures shared across test modules: the synthetic bank, planted scores,
the result-dir scaffold, and the fakes more than one module builds on.

Only what was being copied between files lives here. Questions whose content a
test pins — the papers' golden examples — stay
with the tests that assert on them.
"""

import contextlib
import io
import json

import numpy as np
import pytest

from kcluster.core.question import Question
from kcluster.io.jsonl import dump_questions
from kcluster.tasks.cluster import build_res_df


@pytest.fixture
def make_questions():
    """Factory for a bank of ``n`` interchangeable one-choice MCQs, ``q-0`` onwards.

    ``extra(i)`` supplies question ``i``'s passthrough fields (an ``lo``, a
    ``ds-step-name``), which land after the required ones.
    """
    def make(n: int = 6, extra=None) -> list[Question]:
        return [
            Question(
                {
                    "id": f"q-{i}",
                    "type": "Multiple Choice",
                    "question": {"stem": f"Stem {i}?", "choices": [{"label": "a", "text": "x"}]},
                    "answerKey": "a",
                    **(extra(i) if extra else {}),
                }
            )
            for i in range(n)
        ]

    return make


@pytest.fixture
def planted_scores():
    """Factory for raw scores with planted clusters, stacked ``[marginals; conditionals]``.

    Two questions sharing a group label score 5 nats above the marginal given
    each other, any other pair 5 below — the array ``PointwiseMutualInfo.from_array``
    takes, and, flattened, the grid the local engine's shards cover.
    """
    def make(groups) -> np.ndarray:
        same = np.array([[g1 == g2 for g2 in groups] for g1 in groups])
        return np.vstack([np.full(len(groups), -50.0), np.where(same, -45.0, -55.0)])

    return make


@pytest.fixture
def write_shards():
    """Factory writing stacked scores as the one-rank shard pair ``CustomWriter`` leaves."""
    def write(shard_dir, scores) -> None:
        import torch  # only torch-gated modules ask for this; conftest itself stays torch-free

        flat = torch.tensor(np.asarray(scores).ravel(), dtype=torch.float32)
        shard_dir.mkdir(parents=True, exist_ok=True)
        torch.save([[torch.arange(len(flat))]], shard_dir / "batch_indices_0.pt")
        torch.save([flat], shard_dir / "predictions_0.pt")

    return write


@pytest.fixture
def make_result_dir(tmp_path):
    """Factory for a result dir as the concept step leaves it (D10 layout).

    The bank goes to ``tmp_path/questions.jsonl``; ``tmp_path/run`` gets the
    ``args-concept`` breadcrumb that records it and the Concept KC where the
    concept step puts it (D15).
    """
    def make(questions: list[Question], concepts: list[str]):
        rd = tmp_path / "run"
        data_path = tmp_path / "questions.jsonl"
        dump_questions(questions, str(data_path))
        (rd / "kc" / "concept").mkdir(parents=True)
        (rd / "args-concept-questions.json").write_text(json.dumps({"data_path": str(data_path)}))
        build_res_df(questions, concepts).to_csv(rd / "kc" / "concept" / "questions_concept-kc.csv", index=False)
        return rd

    return make


@pytest.fixture
def write_tsv():
    """Factory writing a frame as a tab-delimited DataShop file; returns its path as ``str``."""
    def write(df, path) -> str:
        df.to_csv(path, sep="\t", index=False)
        return str(path)

    return write


# --- an in-memory stand-in for google.cloud.storage ---

class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    @contextlib.contextmanager
    def open(self, mode="r"):
        if "w" in mode:
            buf = io.BytesIO() if "b" in mode else io.StringIO()
            yield buf
            self.bucket.store[self.name] = buf.getvalue()
        else:
            data = self.bucket.store[self.name]
            yield io.BytesIO(data) if "b" in mode else io.StringIO(data)

    def exists(self):
        return self.name in self.bucket.store


class FakeBucket:
    def __init__(self, name):
        self.name, self.store = name, {}

    def blob(self, name):
        return FakeBlob(self, name)


class FakeStorageClient:
    _buckets = {}

    def bucket(self, name):
        return self._buckets.setdefault(name, FakeBucket(name))

    def list_blobs(self, name, prefix=""):
        bucket = self.bucket(name)
        return [bucket.blob(key) for key in sorted(bucket.store) if key.startswith(prefix)]


@pytest.fixture
def gcs(monkeypatch):
    """An empty in-memory GCS, patched in as the Vertex engine's storage client."""
    from kcluster.engine import vertex  # the vertex extra; the modules asking for this skip without it

    FakeStorageClient._buckets = {}
    monkeypatch.setattr(vertex.storage, "Client", FakeStorageClient)
    return FakeStorageClient()


# --- a stand-in for the local engine, for the server and its client ---

class StubTokenizer:
    def __call__(self, texts, **kwargs):
        texts = [texts] if isinstance(texts, str) else texts
        return {"input_ids": [[len(w) for w in t.split()] for t in texts]}


class StubLLM:
    """Records calls; returns shapes matching LargeLangModel's contract (as plain lists)."""

    device = "cpu"
    tokenizer = StubTokenizer()

    def __init__(self):
        self.calls = []

    def complete_prompts(self, prompts, stop_tokens=None, max_new_tokens=200, pad_to_multiple_of=None, **kwargs):
        self.calls.append(("complete_prompts", kwargs))
        return [p + " done" for p in prompts]

    def log_prob(self, texts, contexts=None, ignore_idx=-100, pad_to_multiple_of=None, return_ppl=False,
                 figures=None, context_figures=None):
        self.calls.append(("log_prob", {"contexts": contexts, "return_ppl": return_ppl,
                                        **({"figures": figures} if figures is not None else {}),
                                        **({"context_figures": context_figures} if context_figures is not None
                                           else {})}))
        return [-float(len(t)) for t in texts]

    def next_tokens(self, prompts, choices=None, top_k=1, pad_to_multiple_of=None):
        self.calls.append(("next_tokens", {"choices": choices, "top_k": top_k}))
        return [tuple((choices or ["x"])[:top_k]) for _ in prompts]

    def next_logits(self, prompts, normalize=False, pad_to_multiple_of=None):
        return [[0.1, 0.2, 0.7] for _ in prompts]

    def encode(self, texts, contexts=None, pad_to_multiple_of=None):
        if contexts is not None and len(contexts) != len(texts):
            raise ValueError("Contexts and texts must have the equal length")
        return [[1.0, 2.0] for _ in texts]


@pytest.fixture(scope="session")
def make_stub_llm():
    """Factory for the engine stub ``kcluster serve`` is tested against.

    Session-scoped so the HTTP client's module-scoped live server can use it.
    """
    return StubLLM
