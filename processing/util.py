import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

KC_PAT = r"KC \((?P<name>.+?)(-\d+)?\)"


def adjust_existing_kc(kc: pd.DataFrame, prob_mask: pd.Series, step_kc_path: str,
                       old_to_new_kc: dict = None, new_kc_suffix: str = "new") -> pd.DataFrame:
    """
    Adjust existing KC models based on available problems
    :param kc: a pd.DataFrame containing KC models
    :param prob_mask: A binary mask with 1 indicating unavailable problems
    :param step_kc_path: A path to a (system-generated) unique-step KC model,
            e.g., "data/datashop/ds6160-spacing/unique-step.txt"
    :param old_to_new_kc: A mapping between old and new KC names
    :param new_kc_suffix: If `old_to_new_kc` is not provided, extend old KC names by `new_kc_suffix`
    :return: A modified KC model as a DataFrame
    """
    # Extract existing KC models
    kc_names = [re.match(KC_PAT, col).group("name") for col in kc.filter(regex=KC_PAT).columns]

    # Adjust old-to-new KC mappings
    old_to_new_kc = old_to_new_kc or {}
    old_to_new_kc = {f"KC ({key})": f"KC ({val})" for key, val in old_to_new_kc.items()}
    default_mapping = {f"KC ({kcm})": f"KC ({kcm.replace(' ', '-')}-{new_kc_suffix})" for kcm in kc_names}
    old_to_new_kc = default_mapping | old_to_new_kc

    # Load the unique-step KC model
    step_kc = pd.read_csv(step_kc_path, sep="\t").dropna(axis="columns", how="all")
    step_mask = ~step_kc["KC (Unique-step)"].str.strip().apply(bool)

    # Empty any cells where the problem name is not found in available questions
    mask = prob_mask | step_mask
    kc.loc[mask, [f"KC ({kcm})" for kcm in kc_names]] = None

    # Strip out the extraneous period for LO
    if (lo_kc := "KC (LO)") in kc:
        kc[lo_kc] = kc[lo_kc].str.rstrip(".")

    # Rename the new KC model
    kc = kc.rename(columns=old_to_new_kc)
    return kc


def merge_student_step_with_kc(ss_path: str, kc_path: str,
                               minimal: bool = False, multiplier: int = 1, save_to_file: bool = False) -> pd.DataFrame:
    """
    This function inserts (multiple) KC models contained in a DataShop KC template into a student-step file.
    In particular, it can prepare KC models for multi-run cross-validation in LearnSphere by duplicating requisite columns.
    If `minimal=False` and `multiplier=1`, it inserts KC models into a DataShop student-step file similar to what DataShop does.
    KC template -> Student Step -> Student Step with duplicate columns
    :param ss_path: Path to a student-step file
    :param kc_path: Path to a filled KC template
    :param minimal: Whether to retain the essential columns only
    :param multiplier: Duplicate the KC columns by `multiplier` times
    :param save_to_file: Whether to save the final result to a file
    :return: A student-step file with KC models inserted, ready for evaluation
    """
    minimal_cols = ["Anon Student Id", "Problem Hierarchy",
                    "Problem Name", "Step Name", "First Transaction Time", "First Attempt"]

    # Identify all KC models
    kc = pd.read_csv(kc_path, sep="\t").dropna(axis="columns", how="all")
    kc_cols = kc.set_index(["Problem Name", "Step Name"]).filter(regex=KC_PAT)
    kc_names = [re.match(KC_PAT, col).group("name") for col in kc_cols.columns]
    if minimal:  # Transcribe KC labels to minimize file size
        for col in kc_cols.columns:
            kc_cols[col] = [f"KC-{lbl}" for lbl in LabelEncoder().fit_transform(kc_cols[col])]

    # Load student-step data
    ss = pd.read_csv(ss_path, sep="\t", dtype={"Anon Student Id": str}, usecols=minimal_cols)

    # Merge KCs into student-step
    ss = pd.merge(ss, kc_cols, how="left", on=["Problem Name", "Step Name"])

    # Adjust columns to minimize file size
    if minimal:
        # Empty nonessential columns
        ss["Problem Hierarchy"] = ""
        # Transcribe columns
        ss["Anon Student Id"] = [f"ST-{lbl}" for lbl in LabelEncoder().fit_transform(ss["Anon Student Id"])]
        ss["Problem Name"] = [f"PN-{lbl}" for lbl in LabelEncoder().fit_transform(ss["Problem Name"])]
        ss["Step Name"] = [f"SN-{lbl}" for lbl in LabelEncoder().fit_transform(ss["Step Name"])]

    # Initialize opportunity columns
    for idx, kcm in enumerate(reversed(kc_names)):
        ss.insert(ss.shape[1] - 2 * idx, f"Opportunity ({kcm})", 0)

    # Calculate opportunity
    opps = defaultdict(lambda: defaultdict(int))
    for idx, row in ss.iterrows():
        for kcm in kc_names:
            kc_col, opp_col = f"KC ({kcm})", f"Opportunity ({kcm})"
            kc_label = f"{kcm}/{row[kc_col]}"
            opps[row["Anon Student Id"]][kc_label] += 1
            ss.loc[idx, opp_col] = opps[row["Anon Student Id"]][kc_label]

    # LearnSphere does not support multi-run CV natively,
    # so we duplicate every KC model to circumvent this limitation.
    replica = dict()
    for idx in range(1, multiplier):
        for kcm in kc_names:
            replica[f"KC ({kcm}-{idx})"] = ss[f"KC ({kcm})"]
            replica[f"Opportunity ({kcm}-{idx})"] = ss[f"Opportunity ({kcm})"]
    replica = pd.DataFrame(replica)

    ss = pd.concat([ss, replica], axis=1)
    if save_to_file:
        ss.to_csv(f"{os.path.splitext(kc_path)[0]}-merged.txt", sep="\t", index=False)
    return ss


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


def compute_clustering_metrics(true_kcs: list[str], pred_kcs: list[str]) -> dict[str, float]:
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


def eval_datashop_kc(kc_temp: str | pd.DataFrame, true_kcm: str) -> pd.DataFrame:
    """
    Evaluate a Datashop KC model against the "ground truth".
    :param kc_temp:
        Either a path to a DataShop KC template file that contain multiple KC models,
        e.g., "data/datashop/ds5426-elearning/ds5426_kcm.txt", or a pd.DataFrame of such
    :param true_kcm: Name of the 'ground-truth' KC model
    :return: A dictionary containing the results of evaluation
    """
    if isinstance(kc_temp, str):
        kc_temp = pd.read_csv(kc_temp, sep="\t").dropna(axis="columns", how="all")
    assert isinstance(kc_temp, pd.DataFrame), "Incorrect type for 'kc_temp'"

    results = dict()
    kc_names = [re.match(KC_PAT, col).group("name") for col in kc_temp.filter(regex=KC_PAT).columns]
    for pred_kcm in kc_names:
        mask = kc_temp[f"KC ({pred_kcm})"].str.strip().apply(bool)
        true_kcs = kc_temp.loc[mask, f"KC ({true_kcm})"]
        pred_kcs = kc_temp.loc[mask, f"KC ({pred_kcm})"]
        results[pred_kcm] = compute_clustering_metrics(true_kcs, pred_kcs)

    return pd.DataFrame.from_dict(results, orient="index")
