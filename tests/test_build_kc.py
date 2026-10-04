"""Offline end-to-end test of the build-kc command.

Synthetic score shards and a Concept KC go in; real affinity propagation
runs; a correctly labeled KCluster model comes out. No model weights are
involved — this pins the whole local pipeline downstream of the GPU steps.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from kcluster.commands import build_kc
from kcluster.io.jsonl import dump_questions
from kcluster.tasks.cluster import build_res_df

GROUPS = ["alpha"] * 3 + ["beta"] * 3


@pytest.fixture
def result_dir(make_questions, make_result_dir, planted_scores, write_shards):
    """A result dir as the concept and pmi steps leave it (D10 layout)."""
    # The Concept KC's concepts follow the two groups, and so do the score shards
    rd = make_result_dir(make_questions(), GROUPS)
    write_shards(rd / "mat" / "pmi" / "raw", planted_scores(GROUPS))
    return rd


def test_build_kc_end_to_end(result_dir):
    build_kc.main(argparse.Namespace(result_dir=str(result_dir)))

    # The clustered model recovers the two groups and carries exemplar concepts
    kc = pd.read_csv(result_dir / "kc" / "kcluster" / "questions_kcluster-unnorm-kc.csv")
    assert kc["KC"].tolist() == GROUPS
    assert kc["KC-raw"].str.fullmatch(r"KC-\d+").all()
    # Distinct concepts, so there is no collision and no split sibling (D14)
    assert not (result_dir / "kc" / "kcluster" / "questions_kcluster-unnorm-split-kc.csv").exists()

    # The assembled congruity matrix is saved for pairwise analyses
    mat = np.load(result_dir / "mat" / "pmi" / "questions_pmi-unnorm.npy")
    assert mat.shape == (6, 6)
    assert np.allclose(mat, mat.T)

    # The args breadcrumb records the recovered data path for downstream steps
    breadcrumb = json.loads((result_dir / "args-kc-questions.json").read_text())
    assert breadcrumb["data_path"].endswith("questions.jsonl")


def test_build_kc_writes_a_split_model_when_exemplar_concepts_collide(result_dir, make_questions):
    """Two clusters whose exemplars share a concept merge under one label (the
    EDM 2025 behaviour); a -split sibling keeping them apart appears beside the
    merged model so both flow through tag/fit as rival models (D14)."""
    build_res_df(make_questions(), ["gamma"] * 6).to_csv(
        result_dir / "kc" / "concept" / "questions_concept-kc.csv", index=False)

    build_kc.main(argparse.Namespace(result_dir=str(result_dir)))

    merged = pd.read_csv(result_dir / "kc" / "kcluster" / "questions_kcluster-unnorm-kc.csv")
    split = pd.read_csv(result_dir / "kc" / "kcluster" / "questions_kcluster-unnorm-split-kc.csv")
    assert merged["KC"].nunique() == 1
    assert split["KC"].nunique() == 2
    expected = merged["KC"] + " [" + merged["KC-raw"] + "]"
    assert split["KC"].tolist() == expected.tolist()


def test_build_kc_finds_its_inputs_inside_a_run_dir(result_dir, monkeypatch):
    """The pairing win: steps that ran at different times share one result
    folder, so build-kc needs no directory arguments at all."""
    monkeypatch.setenv("KCLUSTER_RUN_DIR", str(result_dir))
    build_kc.main(argparse.Namespace())
    assert (result_dir / "kc" / "kcluster" / "questions_kcluster-unnorm-kc.csv").exists()


def test_build_kc_reads_a_pre_d15_flat_kc_dir(result_dir):
    """A result dir from before the kc/ subfolders (D15) keeps its Concept KC
    at the kc/ root; it still resolves, and new models land in kc/kcluster/."""
    (result_dir / "kc" / "concept" / "questions_concept-kc.csv").rename(
        result_dir / "kc" / "questions_concept-kc.csv")
    (result_dir / "kc" / "concept").rmdir()

    build_kc.main(argparse.Namespace(result_dir=str(result_dir)))
    assert (result_dir / "kc" / "kcluster" / "questions_kcluster-unnorm-kc.csv").exists()


def test_build_kc_residualize_full_adds_both_corrected_models(result_dir, make_questions, planted_scores,
                                                             write_shards):
    """--residualize_full implies --residualize: both format-corrected models
    (D9/D11) and matrices appear *beside* the plain ones, so downstream
    consumers can compare all three."""
    # The fixture's bank is single-format, where residualizing is a constant
    # shift and therefore a no-op; plant two formats so the strata differ.
    data_path = json.loads((result_dir / "args-concept-questions.json").read_text())["data_path"]
    questions = make_questions()
    for i, q in enumerate(questions):
        q["type"] = "Fill-in-the-blank(s)" if i < 3 else "Multiple Choice (select 1)"
    dump_questions(questions, data_path)

    # Re-write the shards with per-item effects on top of the block structure:
    # a bank this small puts every stratum below min_pairs, where both variants
    # subtract the same pooled mean — the item term is what tells them apart.
    scores = planted_scores(GROUPS)
    item = np.linspace(0.0, 2.0, 6)
    scores[1:] += item[:, None] + item[None, :]
    write_shards(result_dir / "mat" / "pmi" / "raw", scores)

    build_kc.main(argparse.Namespace(result_dir=str(result_dir), residualize_full=True))

    kc_dir, mat_dir = result_dir / "kc" / "kcluster", result_dir / "mat" / "pmi"
    assert (kc_dir / "questions_kcluster-unnorm-kc.csv").exists()
    assert (kc_dir / "questions_kcluster-unnorm-resid-kc.csv").exists()
    assert (kc_dir / "questions_kcluster-unnorm-residfull-kc.csv").exists()
    plain = np.load(mat_dir / "questions_pmi-unnorm.npy")
    resid = np.load(mat_dir / "questions_pmi-unnorm-resid.npy")
    full = np.load(mat_dir / "questions_pmi-unnorm-residfull.npy")
    assert resid.shape == full.shape == plain.shape
    assert not np.allclose(plain, resid)   # each correction actually did something
    assert not np.allclose(resid, full)    # ... and the two variants differ


def test_build_kc_skips_the_redundant_mean_only_model_on_one_format(result_dir, capsys):
    """The fixture bank is single-format, where the mean-only correction is a
    constant shift: only the joint model is worth writing (D11 follow-up)."""
    build_kc.main(argparse.Namespace(result_dir=str(result_dir), residualize_full=True))

    kc_dir, mat_dir = result_dir / "kc" / "kcluster", result_dir / "mat" / "pmi"
    assert (kc_dir / "questions_kcluster-unnorm-residfull-kc.csv").exists()
    assert not (kc_dir / "questions_kcluster-unnorm-resid-kc.csv").exists()
    assert not (mat_dir / "questions_pmi-unnorm-resid.npy").exists()
    assert "Single-format bank" in capsys.readouterr().out


def test_build_kc_without_shards_builds_no_kcluster_model(result_dir):
    """A result dir the pmi step has not reached yet: the Concept KC is
    validated and the breadcrumb written, but no KCluster model appears."""
    for f in (result_dir / "mat" / "pmi" / "raw").iterdir():
        f.unlink()
    build_kc.main(argparse.Namespace(result_dir=str(result_dir)))
    assert not (result_dir / "kc" / "kcluster" / "questions_kcluster-unnorm-kc.csv").exists()
    assert (result_dir / "args-kc-questions.json").exists()


def test_build_kc_data_path_overrides_an_unreachable_recorded_path(result_dir, tmp_path):
    """A result dir scored on a cluster and rebuilt on a laptop records a path
    that does not exist here; --data_path supplies a reachable copy."""
    recorded = json.loads((result_dir / "args-concept-questions.json").read_text())["data_path"]
    moved = tmp_path / "elsewhere.jsonl"
    os.rename(recorded, moved)

    with pytest.raises(SystemExit, match="Question file not found"):
        build_kc.main(argparse.Namespace(result_dir=str(result_dir)))

    build_kc.main(argparse.Namespace(result_dir=str(result_dir), data_path=str(moved)))
    assert (result_dir / "kc" / "kcluster" / "questions_kcluster-unnorm-kc.csv").exists()


def test_build_kc_requires_a_result_dir_without_a_run_dir(monkeypatch):
    monkeypatch.delenv("KCLUSTER_RUN_DIR", raising=False)
    with pytest.raises(SystemExit, match="--result_dir is required"):
        build_kc.main(argparse.Namespace())


def test_build_kc_normalize_adds_the_joint_normalized_models_beside_the_raw_ones(result_dir):
    """--normalize is additive: the `norm` estimator's model and matrix (and its
    format corrections) land beside the `unnorm` set, never in place of it."""
    build_kc.main(argparse.Namespace(result_dir=str(result_dir), normalize=True, residualize_full=True))

    kc_dir, mat_dir = result_dir / "kc" / "kcluster", result_dir / "mat" / "pmi"
    for tag in ("unnorm", "norm"):
        assert (kc_dir / f"questions_kcluster-{tag}-kc.csv").exists()
        assert (kc_dir / f"questions_kcluster-{tag}-residfull-kc.csv").exists()
        assert (mat_dir / f"questions_pmi-{tag}.npy").exists()
        assert (mat_dir / f"questions_pmi-{tag}-residfull.npy").exists()
    unnorm, norm = np.load(mat_dir / "questions_pmi-unnorm.npy"), np.load(mat_dir / "questions_pmi-norm.npy")
    assert unnorm.shape == norm.shape
    assert np.allclose(norm, norm.T)
    assert not np.allclose(unnorm, norm)   # a different estimator, not a relabelling
    # ... that still separates the planted groups
    kc = pd.read_csv(kc_dir / "questions_kcluster-norm-kc.csv")
    assert kc.groupby("KC-raw")["id"].apply(set).map(len).tolist() == [3, 3]


def test_build_kc_without_normalize_writes_no_norm_models(result_dir):
    build_kc.main(argparse.Namespace(result_dir=str(result_dir)))
    assert not list((result_dir / "kc" / "kcluster").glob("*-norm-*"))
    assert not list((result_dir / "mat" / "pmi").glob("*_pmi-norm*"))
