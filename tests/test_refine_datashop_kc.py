"""Offline end-to-end test of the refine-datashop-kc command.

A four-step "big-kc" with no learning (slope 0, mid-range intercept) is split
by all three refinement strategies; the two-step healthy KC is left alone.
"""

import argparse
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from kcluster.commands.refine_datashop_kc import main  # noqa: E402
from kcluster.core.question import Question  # noqa: E402
from kcluster.io.jsonl import dump_questions  # noqa: E402
from kcluster.tasks.cluster import build_res_df  # noqa: E402

CONCEPTS = ["alpha", "alpha", "beta", "beta", "gamma", "gamma"]
GROUPS = [0, 0, 1, 1, 2, 2]


def _questions() -> list[Question]:
    return [
        Question(
            {
                "id": f"q-{i}",
                "type": "Multiple Choice",
                "question": {"stem": f"Stem {i}?", "choices": [{"label": "a", "text": "x"}]},
                "answerKey": "a",
                "ds-step-name": [f"s{i}"],
            }
        )
        for i in range(6)
    ]


def test_refine_datashop_kc_end_to_end(tmp_path):
    questions = _questions()

    # A result dir as the concept, embed, and pmi steps leave it (D10 layout)
    data_path = tmp_path / "questions.jsonl"
    dump_questions(questions, str(data_path))
    result_dir = tmp_path / "run"
    (result_dir / "kc").mkdir(parents=True)
    (result_dir / "args-concept-questions.json").write_text(json.dumps({"data_path": str(data_path)}))
    build_res_df(questions, CONCEPTS).to_csv(result_dir / "kc" / "questions_concept-kc.csv", index=False)
    embed_dir = result_dir / "mat" / "embed"
    embed_dir.mkdir(parents=True)
    embeds = np.array(
        [[1.0, 0.01], [1.0, 0.02], [0.01, 1.0], [0.02, 1.0], [5.0, 5.0], [5.0, 6.0]]
    )
    np.save(embed_dir / "questions_llm-embed.npy", embeds)

    # The pmi step's shards: within-group above the marginal, across below
    raw_dir = result_dir / "mat" / "pmi" / "raw"
    raw_dir.mkdir(parents=True)
    marginals = np.full(6, -50.0)
    same = np.array([[g1 == g2 for g2 in GROUPS] for g1 in GROUPS])
    conds = np.where(same, -45.0, -55.0)
    flat = np.concatenate([marginals, conds.ravel()])
    torch.save([[torch.arange(len(flat))]], raw_dir / "batch_indices_0.pt")
    torch.save([torch.tensor(flat, dtype=torch.float32)], raw_dir / "predictions_0.pt")

    # The expert model: one problematic four-step KC, one healthy KC
    kc_path = tmp_path / "expert-kc.txt"
    pd.DataFrame(
        {
            "Step Name": [f"s{i}" for i in range(6)],
            "KC (LOs)": ["big-kc"] * 4 + ["other"] * 2,
        }
    ).to_csv(kc_path, sep="\t", index=False)

    # The AFM KC values: big-kc shows no learning; other is fine
    kc_val_path = tmp_path / "LOs_kc-values.csv"
    pd.DataFrame(
        {
            "KC Name": ["big-kc", "other"],
            "Slope": [0.0, 0.5],
            "Intercept (probability) at Opportunity 1": [0.5, 0.5],
            "Number of Unique Steps": [4, 2],
        }
    ).to_csv(kc_val_path, index=False)

    main(argparse.Namespace(kc_path=str(kc_path), kc_val_path=str(kc_val_path),
                            result_dir=str(result_dir)))

    refined = pd.read_csv(result_dir / "kc-refine" / "refined-kc.txt", sep="\t")
    for prefix in ("cpt", "qcos", "pmi"):
        col = f"KC ({prefix}-big-kc)"
        # The problematic KC splits into the two planted sub-KCs; the healthy
        # KC's steps keep their original label.
        assert refined[col].tolist() == ["alpha", "alpha", "beta", "beta", "other", "other"], col
