"""Offline end-to-end test of the build-kc command.

Synthetic score shards and a concept CSV go in; real affinity propagation
runs; correctly labeled concept/cosine/pmi KC models come out. No model
weights are involved — this pins the whole local pipeline downstream of the
GPU steps.
"""

import argparse
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from kcluster.commands import build_kc  # noqa: E402
from kcluster.core.question import Question  # noqa: E402
from kcluster.io.jsonl import dump_questions  # noqa: E402
from kcluster.tasks.concept import build_res_df  # noqa: E402

GROUPS = ["alpha"] * 3 + ["beta"] * 3


def _questions() -> list[Question]:
    return [
        Question(
            {
                "id": f"q-{i}",
                "type": "Multiple Choice",
                "question": {"stem": f"Stem {i}?", "choices": [{"label": "a", "text": "x"}]},
                "answerKey": "a",
            }
        )
        for i in range(6)
    ]


@pytest.fixture
def pipeline_dirs(tmp_path):
    questions = _questions()

    # The questions file plus the args-*.json breadcrumb the concept step writes
    data_path = tmp_path / "questions.jsonl"
    dump_questions(questions, str(data_path))
    concept_dir = tmp_path / "concept"
    concept_dir.mkdir()
    (concept_dir / "args-concept-questions.json").write_text(
        json.dumps({"data_path": str(data_path)})
    )

    # The concept CSV (concepts follow the two groups)
    build_res_df(questions, GROUPS).to_csv(concept_dir / "questions-concept.csv", index=False)

    # Question embeddings: two directions, cosine-separable
    embeds = np.array(
        [[1.0, 0.01], [1.0, 0.02], [1.0, 0.03],
         [0.01, 1.0], [0.02, 1.0], [0.03, 1.0]]
    )
    np.save(concept_dir / "questions-question-embeds.npy", embeds)

    # Score shards: within-group conditionals 5 nats above the marginal,
    # across-group 5 below — the layout CustomWriter would produce.
    marginals = np.full(6, -50.0)
    same = np.array([[g1 == g2 for g2 in GROUPS] for g1 in GROUPS])
    conds = np.where(same, -45.0, -55.0)
    flat = np.concatenate([marginals, conds.ravel()])
    pmi_dir = tmp_path / "pmi"
    pmi_dir.mkdir()
    indices = torch.arange(len(flat))
    torch.save([[indices]], pmi_dir / "batch_indices_0.pt")
    torch.save([torch.tensor(flat, dtype=torch.float32)], pmi_dir / "predictions_0.pt")

    out_dir = tmp_path / "out"
    return concept_dir, pmi_dir, out_dir


def test_build_kc_end_to_end(pipeline_dirs):
    concept_dir, pmi_dir, out_dir = pipeline_dirs
    args = argparse.Namespace(concept_dir=str(concept_dir), pmi_dir=str(pmi_dir),
                              output_dir=str(out_dir))
    build_kc.main(args)

    # The concept model is copied through unchanged
    concept_kc = pd.read_csv(out_dir / "concept-kc.csv")
    assert concept_kc["KC"].tolist() == GROUPS

    # Both clustered models recover the two groups and carry exemplar concepts
    for fname in ("question-cosine-kc.csv", "pmi-kc.csv"):
        kc = pd.read_csv(out_dir / fname)
        assert kc["KC"].tolist() == GROUPS, fname
        assert kc["KC-raw"].str.fullmatch(r"KC-\d+").all(), fname

    # The args breadcrumb records the recovered data path for downstream steps
    breadcrumb = json.loads((out_dir / "args-kc-questions.json").read_text())
    assert breadcrumb["data_path"].endswith("questions.jsonl")
