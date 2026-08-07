"""Affinity-propagation clustering of questions into KC models (EDM 2025).

``run_ap`` groups questions from a precomputed similarity matrix (question
congruity via PointwiseMutualInfo.pmi_mat, or an embedding baseline via
``sim_from_embeddings``) and labels each cluster ``KC-<exemplar index>``.
``create_kc`` searches for the smallest damping factor that converges and
relabels every cluster with its exemplar's concept, turning nominal cluster
ids into descriptive KC labels.
"""

import itertools
import warnings
from collections.abc import Callable
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.cluster import affinity_propagation
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances

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
