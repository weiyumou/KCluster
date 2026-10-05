"""Offline end-to-end tests of the refine-datashop-kc command.

A four-step "big-kc" with no learning (slope 0, mid-range intercept) is split
by all three refinement strategies; the two-step healthy KC is left alone.
"""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from kcluster.cli import main as cli_main
from kcluster.datashop.kc_model import load_kc_template
from kcluster.io.jsonl import dump_questions
from kcluster.tasks.cluster import build_res_df

CONCEPTS = ["alpha", "alpha", "beta", "beta", "gamma", "gamma"]
GROUPS = [0, 0, 1, 1, 2, 2]


@pytest.fixture
def refine(tmp_path, make_questions, planted_scores, write_shards):
    """Run the command on a six-question result dir; returns the refined template.

    ``steps`` gives each question's (problem, step) and ``big`` names the
    problematic KC, which the first four steps carry.
    """
    def run(steps=tuple(("P", f"s{i}") for i in range(6)), big="big-kc") -> pd.DataFrame:
        questions = make_questions(extra=lambda i: {"ds-problem-name": steps[i][0], "ds-step-name": [steps[i][1]]})

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

        # The expert model, as DataShop exports it (unquoted, blank cells a
        # space): one problematic four-step KC, one healthy KC
        kc_path = tmp_path / "expert-kc.txt"
        labels = [big] * 4 + ["other"] * 2
        kc_path.write_text("Problem Name\tStep Name\tKC (LOs)\tKC (new KC model name)\n"
                           + "".join(f"{problem}\t{step}\t{label}\t \n"
                                     for (problem, step), label in zip(steps, labels, strict=True)))

        # The AFM KC values: the big KC shows no learning; other is fine
        kc_val_path = tmp_path / "LOs_kc-values.csv"
        pd.DataFrame(
            {
                "KC Name": [big, "other"],
                "Slope": [0.0, 0.5],
                "Intercept (probability) at Opportunity 1": [0.5, 0.5],
                "Number of Unique Steps": [4, 2],
            }
        ).to_csv(kc_val_path, index=False)

        cli_main(["refine-datashop-kc", "--kc_path", str(kc_path), "--kc_val_path", str(kc_val_path),
                  "--result_dir", str(result_dir)])
        return load_kc_template(str(result_dir / "kc-refine" / "refined-kc.txt"))

    return run


def test_refine_datashop_kc_end_to_end(refine):
    refined = refine()
    for prefix in ("cpt", "qcos", "pmi"):
        col = f"KC ({prefix}-big-kc)"
        # The problematic KC splits into the two planted sub-KCs; the healthy
        # KC's steps keep their original label.
        assert refined[col].tolist() == ["alpha", "alpha", "beta", "beta", "other", "other"], col
    # Only the model being refined is carried over
    assert "KC (new KC model name)" not in refined


def test_a_label_with_quotes_in_it_is_the_kcs_name_as_it_stands(refine):
    # DataShop's files are unquoted text. Read CSV-style, '"big" kc' lost its
    # quotes and matched no KC value; written CSV-style, DataShop would store
    # the quoting as part of the name.
    label = '"big" kc'
    refined = refine(big=label)
    assert refined["KC (LOs)"].tolist() == [label] * 4 + ["other"] * 2
    assert refined[f"KC (cpt-{label})"].tolist() == ["alpha", "alpha", "beta", "beta", "other", "other"]


def test_a_step_name_that_recurs_under_another_problem_stays_out_of_the_refinement(refine):
    # The healthy KC's two steps are named like two of the big KC's, under
    # another problem. Matched by step name alone, their questions joined the
    # refinement and lent the big KC's steps their labels.
    steps = (("P1", "s0"), ("P1", "s1"), ("P1", "s2"), ("P1", "s3"), ("P2", "s0"), ("P2", "s1"))
    refined = refine(steps=steps)
    for prefix in ("cpt", "qcos", "pmi"):
        assert refined[f"KC ({prefix}-big-kc)"].tolist() == ["alpha", "alpha", "beta", "beta", "other", "other"]
