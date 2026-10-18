"""Offline end-to-end tests of the refine-datashop-kc command.

A four-step KC with no learning (slope 0, mid-range intercept) is split by all
three refinement strategies; the two-step healthy KC is left alone.
"""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from kcluster.cli import main as cli_main
from kcluster.commands.refine_datashop_kc import kc_model_name
from kcluster.datashop.kc_model import load_kc_template
from kcluster.io.jsonl import dump_questions
from kcluster.tasks.cluster import build_res_df

CONCEPTS = ["alpha", "alpha", "beta", "beta", "gamma", "gamma"]
GROUPS = [0, 0, 1, 1, 2, 2]


@pytest.fixture
def refine(tmp_path, make_questions, planted_scores, write_shards):
    """Run the command on a six-question result dir; returns the refined template.

    ``steps`` gives each question's (problem, step), ``labels`` each step's KC
    in the model to refine (the first four carry the problematic one) and
    ``concepts`` each question's concept.
    """
    def run(steps=tuple(("P", f"s{i}") for i in range(6)), labels=("big-kc",) * 4 + ("other",) * 2,
            concepts=CONCEPTS, kc_val_name="LOs_kc-values.csv") -> pd.DataFrame:
        questions = make_questions(extra=lambda i: {"ds-problem-name": steps[i][0], "ds-step-name": [steps[i][1]]})

        # A result dir as the concept, embed, and pmi steps leave it (D10 layout)
        data_path = tmp_path / "questions.jsonl"
        dump_questions(questions, str(data_path))
        result_dir = tmp_path / "run"
        (result_dir / "kc").mkdir(parents=True)
        (result_dir / "args-concept-questions.json").write_text(json.dumps({"data_path": str(data_path)}))
        build_res_df(questions, concepts).to_csv(result_dir / "kc" / "questions_concept-kc.csv", index=False)
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
        kc_path.write_text("Problem Name\tStep Name\tKC (LOs)\tKC (new KC model name)\n"
                           + "".join(f"{problem}\t{step}\t{label}\t \n"
                                     for (problem, step), label in zip(steps, labels, strict=True)))

        # The AFM KC values: the first KC shows no learning; the other is fine
        kc_val_path = tmp_path / kc_val_name
        pd.DataFrame(
            {
                "KC Name": [labels[0], labels[-1]],
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
    refined = refine(labels=(label,) * 4 + ("other",) * 2)
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


def test_a_model_labelled_by_concept_can_be_refined(refine):
    # A KCluster model is labelled by concept, as the refinements are. Here KC
    # 'alpha' splits into alpha (its own label, which stays) and beta, which
    # already names another KC and so takes its parent's label in brackets.
    # Requiring every step to change label, refine used to stop here.
    refined = refine(labels=("alpha",) * 4 + ("beta",) * 2)
    for prefix in ("cpt", "qcos", "pmi"):
        assert refined[f"KC ({prefix}-alpha)"].tolist() == \
            ["alpha", "alpha", "beta [alpha]", "beta [alpha]", "beta", "beta"]


def test_clusters_that_share_a_concept_are_kept_apart(refine):
    # Both clusters of big-kc have an 'alpha' exemplar: one label would merge them
    refined = refine(concepts=["alpha"] * 4 + ["gamma"] * 2)
    assert refined["KC (cpt-big-kc)"].tolist() == ["alpha"] * 4 + ["other"] * 2
    for prefix in ("qcos", "pmi"):
        labels = refined[f"KC ({prefix}-big-kc)"].tolist()
        assert labels[0] == labels[1] != labels[2] == labels[3], prefix
        assert all(label.startswith("alpha [KC-") for label in labels[:4]), prefix


def test_kc_values_named_as_fit_names_them_are_read(refine):
    refined = refine(kc_val_name="LOs.csv")
    assert refined["KC (cpt-big-kc)"].tolist()[:4] == ["alpha", "alpha", "beta", "beta"]


def test_kc_model_name_comes_from_the_file_name_or_the_flag():
    template = pd.DataFrame({"Step ID": ["id1"], "Step Name": ["s1"], "KC (LOs)": ["a"], "KC (kcluster-unnorm)": ["b"]})
    assert kc_model_name(template, "fit/afm/kc-values/kcluster-unnorm.csv") == "kcluster-unnorm"
    assert kc_model_name(template, "LOs_kc-values.csv") == "LOs"
    assert kc_model_name(template, "values.csv", "LOs") == "LOs"
    with pytest.raises(SystemExit, match=r"it has \['LOs', 'kcluster-unnorm'\].*--kc_model"):
        kc_model_name(template, "concept.csv")
