"""Affinity-propagation clustering of questions into KC models (EDM 2025).

``run_ap`` groups questions from a precomputed similarity matrix (question
congruity via PointwiseMutualInfo.pmi_mat, or an embedding baseline via
``sim_from_embeddings``) and labels each cluster ``KC-<exemplar index>``.
``create_kc`` searches for the smallest damping factor that converges and
relabels every cluster with its exemplar's concept, turning nominal cluster
ids into descriptive KC labels.
"""

import itertools
import os
import warnings
from collections.abc import Callable
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.cluster import affinity_propagation
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances

from kcluster.core.pmi import correction_variants
from kcluster.core.pmi import residualize as residualize_congruity
from kcluster.core.question import Question


def sim_from_embeddings(embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Negated pairwise distances, so that larger means more similar."""
    return -pairwise_distances(embeddings, metric=metric)


def run_ap(questions: list[Question], sim_mtx: np.ndarray,
           predicate: Callable[[Question], bool] | None = None,
           use_p: str = "median", **ap_kwargs) -> pd.DataFrame:
    """
    Create a new KC model from the similarity matrix
    :param questions: The questions the rows/columns of the matrix describe
    :param sim_mtx: A square similarity matrix over the questions
    :param predicate: A function indicating if a question should be considered
    :param use_p: Determine how to compute preference
    :param ap_kwargs: Additional arguments for the Affinity Propagation algorithm, e.g., damping=0.7
    :return: The new KC model in a DataFrame
    """
    assert use_p in ("median", "mean", "min", "max"), f"Invalid value for 'use_p': {use_p}"

    # Determine the questions and similarity matrix
    if predicate is not None:
        is_valid = [predicate(q) for q in questions]
        questions = list(itertools.compress(questions, is_valid))
        sim_mtx = sim_mtx[np.ix_(is_valid, is_valid)]
    assert sim_mtx.shape == (len(questions), len(questions)), "The shape of the similarity matrix is incorrect"

    # Determine p
    func: dict[str, Callable] = {"median": np.median, "mean": np.mean, "min": np.amin, "max": np.amax}
    p = func[use_p](sim_mtx[~np.eye(len(questions), dtype=bool)])

    # Run AP
    centers, labels, num_iters = affinity_propagation(sim_mtx, preference=p,
                                                      return_n_iter=True, random_state=42, **ap_kwargs)
    print(f"Affinity Propagation completed in {num_iters} iterations and created {len(centers)} clusters")

    # Collect clustering results
    res_dicts = []
    for q, label in zip(questions, labels, strict=True):
        q_dict = q.flat_dict
        q_dict.pop("images", None)
        q_dict["KC"] = f"KC-{centers[label]}"
        res_dicts.append(q_dict)
    return pd.DataFrame.from_records(res_dicts)


def build_res_df(questions: list[Question], concepts: list[str]) -> pd.DataFrame:
    """Pair questions with their concept labels as a KC-model DataFrame.

    Lives here (not in tasks.concept) so torch-free consumers — the Vertex
    build-kc command in particular — can import it without the local engine.
    """
    q_dicts = []
    for q, c in zip(questions, concepts):
        q_dict = q.flat_dict
        q_dict.pop("images", None)
        q_dict["KC"] = c
        q_dicts.append(q_dict)

    return pd.DataFrame.from_records(q_dicts)


def create_kc(concept_df: pd.DataFrame, questions: list[Question], sim_mtx: np.ndarray,
              **kwargs) -> pd.DataFrame | None:
    """Cluster questions and label each cluster with its exemplar's concept.

    Searches for the minimal damping factor leading to convergence; returns
    None if affinity propagation never converges.
    """
    # Flag convergence issues as errors
    warnings.filterwarnings("error", category=ConvergenceWarning)

    damping = 0.5
    while damping < 1.0:
        try:
            kc = run_ap(questions, sim_mtx, damping=damping, **kwargs)
            assert kc.shape[0] == concept_df.shape[0], "Inconsistent number of questions"
        except ConvergenceWarning:
            print(f"Did not converge when damping = {damping}")
            damping += 0.05
        else:
            # populate the concepts of exemplars to its subordinates
            kc = kc.rename(columns={"KC": "KC-raw"})
            exemplars = kc["KC-raw"].str.split("-").apply(itemgetter(1)).apply(int)
            kc["KC"] = concept_df["KC"].iloc[exemplars].reset_index(drop=True)
            return kc
    print("*** Failed to create KCs ***")
    return None


def split_collisions(kc: pd.DataFrame) -> pd.DataFrame | None:
    """The label-collision-free variant of a clustered KC model, or None.

    Keying a model by concept label merges two clusters whose exemplars
    produced the same concept — behaviour EDM 2025 acknowledges and leaves to
    the practitioner ("whichever leads to better performance"). This is the
    other side of that choice: a label shared by several clusters gets each
    cluster's id appended (``percentages [KC-152]``), keeping one KC per
    cluster; unshared labels stay as they are. Returns None when no label is
    shared, where the split model would duplicate the merged one.
    """
    counts = kc.groupby("KC")["KC-raw"].nunique()
    collided = kc["KC"].isin(counts[counts > 1].index)
    if not collided.any():
        return None
    split = kc.copy()
    split.loc[collided, "KC"] = split.loc[collided, "KC"] + " [" + split.loc[collided, "KC-raw"] + "]"
    assert split.groupby("KC")["KC-raw"].nunique().eq(1).all(), "splitting left a shared label"
    return split


def save_kc_models(kc: pd.DataFrame, kc_path: str) -> None:
    """Write a clustered KC model and, when labels collide, its split sibling.

    The sibling swaps ``-kc.csv`` for ``-split-kc.csv`` (D14), so the tagger
    picks the two up as rival models under one run: merged is the EDM 2025
    model, split keeps every affinity-propagation cluster its own KC.
    """
    kc.to_csv(kc_path, index=False)
    if (split := split_collisions(kc)) is not None:
        split_path = kc_path.removesuffix("-kc.csv") + "-split-kc.csv"
        split.to_csv(split_path, index=False)
        print(f"*** {split['KC'].nunique() - kc['KC'].nunique()} cluster(s) share another cluster's label: "
              f"also wrote {os.path.basename(split_path)} with all {split['KC'].nunique()} clusters apart ***")


#: The congruity estimators a bank can be clustered under, by artifact tag.
#: ``unnorm`` is the raw ``log P(j|i) - log P(j)`` with the model's own
#: marginals (the published KCluster model); ``norm`` renormalizes the
#: symmetrized joint over the bank and takes its marginals from that
#: (``PointwiseMutualInfo(normalize=True)``).
CONGRUITY_ESTIMATORS = (("unnorm", False), ("norm", True))


def build_kcluster_models(concept_df: pd.DataFrame, questions: list[Question],
                          congruity: Callable[[bool], np.ndarray], *, ds: str,
                          kc_out_dir: str, mat_out_dir: str, normalize: bool = False,
                          residualize: bool = False, residualize_full: bool = False) -> dict[str, pd.DataFrame]:
    """Write one bank's KCluster KC models: every estimator x every correction.

    The one builder behind both engines (local ``build-kc`` from score shards,
    ``vertex-build-kc`` from a downloaded array), so the flags mean the same
    thing everywhere:

    - ``normalize`` selects *estimators*: ``unnorm`` always, plus ``norm`` when
      set. Additive — the ``norm`` set is written beside the ``unnorm`` set,
      never in place of it.
    - ``residualize`` / ``residualize_full`` select *corrections* of the
      question-format nuisance (D9/D11: mean-only ``resid``, joint item +
      format ``residfull``; the latter implies the former). Also additive, and
      applied to **each** estimator's matrix as the last operation, so whatever
      format offset an estimator preserves is removed from the final matrix.
      A single-format bank gets no ``resid`` model: with one stratum the
      correction is a constant shift affinity propagation ignores.

    ``congruity(normalize)`` returns the square similarity matrix of the bank
    under that estimator; it is called once per estimator built. Artifacts are
    ``<ds>_pmi-<estimator>[-<correction>].npy`` in ``mat_out_dir`` and
    ``<ds>_kcluster-<estimator>[-<correction>]-kc.csv`` (plus any ``-split``
    sibling, D14) in ``kc_out_dir``.

    :return: the models written, ``{"kcluster-<tag>": frame}``
    """
    groups = [q.q_type for q in questions]
    want_mean = residualize_full or residualize
    variants = correction_variants(groups, mean_only=want_mean, joint=residualize_full)
    if want_mean and not any(tag == "resid" for tag, _ in variants):
        print("*** Single-format bank: skipping the mean-only model, which would duplicate "
              "the uncorrected one (with one stratum it is a constant shift) ***")

    models = {}
    for estimator, normalized in CONGRUITY_ESTIMATORS:
        if normalized and not normalize:
            continue
        sim_mtx = congruity(normalized)
        assert sim_mtx.shape == (len(questions), len(questions)), "Inconsistent similarity matrix shape"

        corrections = [(estimator, None)] + [(f"{estimator}-{tag}", kwargs) for tag, kwargs in variants]
        for tag, kwargs in corrections:
            print(f"*** Building KCs for KCluster-PMI ({tag}) ***")
            matrix = sim_mtx if kwargs is None else residualize_congruity(sim_mtx, groups, **kwargs)
            # The matrix is what a pairwise analysis of these questions should
            # read, so it is saved beside the model rather than left recoverable
            # only by redoing the estimator and strata by hand.
            np.save(os.path.join(mat_out_dir, f"{ds}_pmi-{tag}.npy"), matrix)
            kc = create_kc(concept_df, questions, matrix)
            if isinstance(kc, pd.DataFrame):
                save_kc_models(kc, os.path.join(kc_out_dir, f"{ds}_kcluster-{tag}-kc.csv"))
                print(f"*** Finished with {kc['KC'].nunique()} KCs ***")
                models[f"kcluster-{tag}"] = kc
    return models
