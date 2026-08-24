"""Evaluation of KC models: clustering-alignment metrics against a reference
model, and a shuffled-label null baseline.

Reading another tool's fit results is a separate concern and lives with the
other foreign-format readers: :mod:`kcluster.io.learnsphere`."""

import random
import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

from kcluster.io.datashop import KC_PAT, load_datashop_temp


def compute_clustering_metrics(true_kcs: Sequence[str], pred_kcs: Sequence[str]) -> dict[str, float]:
    """https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation"""
    true_kcs = LabelEncoder().fit_transform(true_kcs)
    pred_kcs = LabelEncoder().fit_transform(pred_kcs)

    return {
        "Rand Index [0, 1]": metrics.rand_score(true_kcs, pred_kcs),
        "Adj Rand Index [-1, 1]": metrics.adjusted_rand_score(true_kcs, pred_kcs),
        "Norm MI [0, 1]": metrics.normalized_mutual_info_score(true_kcs, pred_kcs),
        "Adj MI (-∞, 1]": metrics.adjusted_mutual_info_score(true_kcs, pred_kcs),
        "Fowlkes-Mallows Index [0, 1]": metrics.fowlkes_mallows_score(true_kcs, pred_kcs),
        "Homogeneity [0, 1]": metrics.homogeneity_score(true_kcs, pred_kcs),
        "Completeness [0, 1]": metrics.completeness_score(true_kcs, pred_kcs),
        "V-measure [0, 1]": metrics.v_measure_score(true_kcs, pred_kcs),
    }


def eval_datashop_kc(kc_temp: str | pd.DataFrame, true_kcm: str, random_kc: bool = False, **kwargs) -> pd.DataFrame:
    """
    Evaluate a Datashop KC model against the "ground truth".
    :param kc_temp:
        Either a path to a DataShop KC template file that contain multiple KC models,
        e.g., "data/datashop/ds5426-elearning/ds5426_kcm.txt", or a pd.DataFrame of such
    :param true_kcm: Name of the 'ground-truth' KC model
    :param random_kc: Whether to compare with a random KC model
    :return: A dictionary containing the results of evaluation
    """
    if isinstance(kc_temp, str):
        kc_temp = load_datashop_temp(kc_temp)
    assert isinstance(kc_temp, pd.DataFrame), "Incorrect type for 'kc_temp'"

    results = dict()
    kc_names = [re.match(KC_PAT, col).group("name") for col in kc_temp.filter(regex=KC_PAT)]
    for pred_kcm in kc_names:
        mask = kc_temp[f"KC ({pred_kcm})"].notna()
        true_kcs = kc_temp.loc[mask, f"KC ({true_kcm})"].to_list()
        pred_kcs = kc_temp.loc[mask, f"KC ({pred_kcm})"].to_list()
        num_kcs = kc_temp[f"KC ({pred_kcm})"].nunique(dropna=True)
        results[f"{pred_kcm} ({num_kcs} KCs)"] = compute_clustering_metrics(true_kcs, pred_kcs)

        if random_kc:
            results[f"{pred_kcm}-rand ({num_kcs} KCs)"] = evaluate_random_kc(true_kcs, pred_kcs, **kwargs)

    return pd.DataFrame.from_dict(results, orient="index")


def evaluate_random_kc(true_kcs: Sequence[str], pred_kcs: Sequence[str],
                       num_runs: int = 30, seed: int = 42) -> dict[str, float]:
    """
    This function evaluates the alignment between a ground-truth KC model and a random KC model
    :param true_kcs: A list of KC labels in the ground-truth KC model
    :param pred_kcs: A list of KC labels that will be random shuffled
    :param num_runs: Number of simulations to run
    :param seed: Random seed for reproducibility
    :return: A dictionary containing the results of evaluation
    """
    rng = random.Random(seed)
    results = dict()
    for _ in range(num_runs):
        random_kcs = rng.sample(pred_kcs, k=len(pred_kcs))
        for m, v in compute_clustering_metrics(true_kcs, random_kcs).items():
            results.setdefault(m, []).append(v)

    for m in results:
        results[m] = np.mean(results[m])
    return results
