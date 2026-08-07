"""Tests for the serving container's predictor (deploy/vertex/serving/).

The predict() dispatch is the RPC contract between the Vertex engine's
instances and LargeLangModel's method surface: instances are grouped by
``purpose``, each group is scored in one call, and results return in the
original instance order.
"""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("google.cloud.aiplatform")

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def predictor_cls():
    path = REPO_ROOT / "deploy" / "vertex" / "serving" / "predictor.py"
    spec = importlib.util.spec_from_file_location("vertex_serving_predictor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Phi2Predictor


class StubLLM:
    """Echoes calls; log_prob returns a tensor to exercise .tolist()."""

    def __init__(self):
        self.calls = []

    def complete_prompts(self, texts, **kwargs):
        self.calls.append(("complete_prompts", texts, kwargs))
        return [f"completed: {t}" for t in texts]

    def log_prob(self, texts, contexts, **kwargs):
        self.calls.append(("log_prob", texts, kwargs))
        assert len(texts) == len(contexts)
        return torch.tensor([-float(len(t)) for t in texts])


def test_predict_dispatches_by_purpose_and_preserves_order(predictor_cls):
    predictor = predictor_cls()
    predictor.llm = StubLLM()

    instances = predictor.preprocess(
        {
            "instances": [
                {"id": "concept-0", "text": "prompt A", "purpose": "complete_prompts",
                 "config": {"num_beams": 5}},
                {"id": "pmi-0", "text": "q0", "context": "ctx", "purpose": "log_prob",
                 "config": {"pad_to_multiple_of": 8}},
                {"id": "pmi-1", "text": "long q1", "context": "ctx", "purpose": "log_prob",
                 "config": {"pad_to_multiple_of": 8}},
                {"id": "concept-1", "text": "prompt B", "purpose": "complete_prompts",
                 "config": {"num_beams": 5}},
            ],
            "parameters": {"complete_prompts": {"max_new_tokens": 20}},
        }
    )
    result = predictor.predict(instances)

    # Interleaved purposes come back in the original instance order
    assert result["predictions"] == ["completed: prompt A", -2.0, -7.0, "completed: prompt B"]

    # One batched call per purpose, with parameters merged with per-instance config
    [(name1, texts1, kwargs1), (name2, texts2, kwargs2)] = predictor.llm.calls
    assert (name1, texts1) == ("complete_prompts", ["prompt A", "prompt B"])
    assert kwargs1 == {"max_new_tokens": 20, "num_beams": 5}
    assert (name2, texts2) == ("log_prob", ["q0", "long q1"])
    assert kwargs2 == {"pad_to_multiple_of": 8}


def test_predict_rejects_partial_contexts(predictor_cls):
    predictor = predictor_cls()
    predictor.llm = StubLLM()

    instances = [
        {"id": "pmi-0", "text": "q0", "context": "ctx", "purpose": "log_prob"},
        {"id": "pmi-1", "text": "q1", "purpose": "log_prob"},  # missing context
    ]
    with pytest.raises(AssertionError, match="equal length"):
        predictor.predict(instances)
