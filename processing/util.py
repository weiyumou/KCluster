import random
import re
from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

KC_PAT = r"KC \((?P<name>.+?)(-\d+)?\)"


def read_cv_results(res_path: str, num_cv_runs: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(res_path, "r") as f:
        soup = BeautifulSoup(f, features="html.parser")

    # Extract results
    results = defaultdict(lambda: defaultdict(list))
    for model in soup.find_all("model"):
        name = model.find("name").string
        if match := re.match(KC_PAT, name):
            name = match.group("name")
        else:
            raise ValueError(f"Unrecognized name: {name}")

        for tag in model.find("name").next_siblings:
            if tag.name:
                val = float(tag.string)
                results[tag.name][name].append(val)

    # Verify there is a correct number of results
    for metric in results:
        for model in results[metric]:
            num_results = len(results[metric][model])
            assert num_results == num_cv_runs, f"Expected {num_cv_runs} results, got {num_results} for '{model}'"

    # Build a machine-readable result table for further processing
    res_table = dict()
    for metric in results:
        res_table[metric] = pd.DataFrame(results[metric])
        if metric in {"aic", "bic", "log_likelihood"}:
            res_table[metric] = res_table[metric].mean(axis=0).to_frame().T
    res_table = pd.concat(res_table)

    # Build a human-readable result table for use in a paper
    pub_table = defaultdict(lambda: dict())
    for metric in results:
        for model in results[metric]:
            mean, std = np.mean(results[metric][model]), np.std(results[metric][model])
            pub_table[metric][model] = f"{mean:.4f} ({std:.4f})"
    pub_table = pd.DataFrame.from_dict(pub_table, orient="columns")

    return res_table, pub_table


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
        kc_temp = pd.read_csv(kc_temp, sep="\t", na_values=" ").dropna(axis="columns", how="all")
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
