"""Offline end-to-end test of the refine-datashop-kc command.

A four-step "big-kc" with no learning (slope 0, mid-range intercept) is split
by all three refinement strategies; the two-step healthy KC is left alone.
"""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from kcluster.cli import main as cli_main
from kcluster.io.jsonl import dump_questions
from kcluster.tasks.cluster import build_res_df

CONCEPTS = ["alpha", "alpha", "beta", "beta", "gamma", "gamma"]
GROUPS = [0, 0, 1, 1, 2, 2]


def test_refine_datashop_kc_end_to_end(tmp_path, make_questions, planted_scores, write_shards):
    questions = make_questions(extra=lambda i: {"ds-step-name": [f"s{i}"]})

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
    write_shards(result_dir / "mat" / "pmi" / "raw", planted_scores(GROUPS))

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

    cli_main(["refine-datashop-kc", "--kc_path", str(kc_path), "--kc_val_path", str(kc_val_path),
              "--result_dir", str(result_dir)])

    refined = pd.read_csv(result_dir / "kc-refine" / "refined-kc.txt", sep="\t")
    for prefix in ("cpt", "qcos", "pmi"):
        col = f"KC ({prefix}-big-kc)"
        # The problematic KC splits into the two planted sub-KCs; the healthy
        # KC's steps keep their original label.
        assert refined[col].tolist() == ["alpha", "alpha", "beta", "beta", "other", "other"], col
