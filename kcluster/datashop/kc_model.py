# -*- coding: utf-8 -*-
"""
This module contains functions for working with DataShop Knowledge Component (KC) models.
"""
import itertools
import os
import re
from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from kcluster.io.jsonl import load_questions

KC_PAT = r"KC \((?P<name>.+?)(-\d+)?\)"


def load_datashop_temp(path: str) -> pd.DataFrame:
    """Load KC models from a DataShop template file"""
    kc_temp = pd.read_csv(path, sep="\t", na_values=" ").dropna(axis="columns", how="all")
    kc_temp["Problem Hierarchy"] = kc_temp["Problem Hierarchy"].fillna("")
    kc_temp["Problem Name"] = kc_temp["Problem Name"].fillna("")
    kc_temp["Step Name"] = kc_temp["Step Name"].fillna("")
    return kc_temp


def save_datashop_temp(df: pd.DataFrame, path: str) -> None:
    """Save KC models to a DataShop tab-delimited template file"""
    df.to_csv(path, sep="\t", index=False)


def create_default_kc(kc_temp_path: str,
                      single_kc_name: str = "Single-KC-full", unique_step_name: str = "Unique-step-full",
                      kc_level: str = None, save_to_file: bool = False) -> pd.DataFrame:
    """
    Create a full unique-step KC model that does not exclude any steps as DataShop does
    :param kc_temp_path: Path to an empty DataShop KC template file
    :param single_kc_name: The name of the Single-KC model
    :param unique_step_name: The name of the unique-step KC model
    :param kc_level: If provided, prepend the values in this column to each KC label, separated by "/"
    :param save_to_file: Whether to save the new KC models to a file
    :return: The two default KC models as a DataFrame
    """
    kc_temp = load_datashop_temp(kc_temp_path)

    # Create the Single-KC model
    kc_temp[f"KC ({single_kc_name})"] = "Single-KC"
    if kc_level is not None:
        kc_temp[f"KC ({single_kc_name})"] = kc_level + "/" + kc_temp[f"KC ({single_kc_name})"]

    # Create the Unique-step model
    kc_temp[f"KC ({unique_step_name})"] = kc_temp["Problem Hierarchy"] + kc_temp["Problem Name"] + kc_temp["Step Name"]
    unique_keys = pd.unique(kc_temp[f"KC ({unique_step_name})"])
    key_map = dict(zip(unique_keys, (f"KC-{i}" for i in range(1, len(unique_keys) + 1)), strict=True))
    kc_temp[f"KC ({unique_step_name})"] = kc_temp[f"KC ({unique_step_name})"].map(key_map)
    if kc_level is not None:
        kc_temp[f"KC ({unique_step_name})"] = kc_level + "/" + kc_temp[f"KC ({unique_step_name})"]

    if save_to_file:
        fname = os.path.join(os.path.dirname(kc_temp_path), "default-kc.txt")
        save_datashop_temp(kc_temp, fname)

    return kc_temp


def create_kc_from_questions(data_path: str, kc_temp_path: str,
                             kc_fields: list[str], kc_names: list[str] = None, kc_level: str = None,
                             save_to_file: bool = False, **kwargs) -> pd.DataFrame:
    """
    Create KCs based on properties of questions
    :param data_path: Path to a jsonl file containing questions
    :param kc_temp_path: Path to an empty DataShop KC template file
    :param kc_fields: Fields in the question data to be used as KCs
    :param kc_names: New names for the KCs; if None, use `fields` as names
    :param kc_level: If provided, prepend the values in this column to each KC label, separated by "/"
    :param save_to_file: Whether to save the new KC model to a file
    :return: The new KC model as a DataFrame
    """
    questions = load_questions(data_path)
    data_df = pd.DataFrame.from_records([q.flat_dict for q in questions])
    if kc_level is not None:
        for col in kc_fields:
            data_df[col] = data_df[kc_level] + "/" + data_df[col]

    kc_temp = create_datashop_kc(data_df, kc_temp_path,
                                 kc_cols=kc_fields, new_kc_names=kc_names, drop_other_kc=True, **kwargs)

    match_other_kc = kwargs.get("match_other_kc", False)
    assert match_other_kc or kc_temp.filter(regex=KC_PAT).notna().all(axis=None), "Some KC labels are missing"

    if save_to_file:
        fname = "_".join(kc_names).lower()
        fname = os.path.join(os.path.dirname(kc_temp_path), f"{fname}.txt")
        save_datashop_temp(kc_temp, fname)

    return kc_temp


def adjust_datashop_kc(data_path: str, kc_path: str, step_kc_path: str, save_to_file: bool = False,
                       old_to_new_kc: dict = None, new_kc_suffix: str = "new") -> pd.DataFrame:
    """
    Adjust existing DataShop KC models according to available questions
    :param data_path: A path to a jsonl file containing questions, e.g., "data/elearning/elearning22-mcq.jsonl"
    :param kc_path: A path to a (filled) DataShop KC template file, e.g., "data/datashop/ds5426-elearning/ds5426_kcm.txt"
    :param step_kc_path: A path to a (system-generated) unique-step KC model,
        e.g., "data/datashop/ds5426-elearning/unique-step.txt"
    :param save_to_file: Whether to save the new KC models to a file
    :param old_to_new_kc: A mapping between old and new KC names
    :param new_kc_suffix: If `old_to_new_kc` is not provided, extend old KC names by `new_kc_suffix`
    :return: A modified KC model as a DataFrame
    """
    # Load the existing KC model and identify the KC mask
    kc = load_datashop_temp(kc_path)
    kc_mask = kc.filter(regex=KC_PAT).isna().any(axis=1)

    # Extract existing KC models
    kc_names = [re.match(KC_PAT, col).group("name") for col in kc.filter(regex=KC_PAT).columns]

    # Load questions and identify the problem mask
    questions = load_questions(data_path)

    ds_step_names = set(itertools.chain.from_iterable(q["ds-step-name"] for q in questions))
    prob_mask = ~kc["Step Name"].isin(ds_step_names)

    # Load the unique-step KC model
    step_kc = load_datashop_temp(step_kc_path)
    step_mask = step_kc["KC (Unique-step)"].isna()

    # Empty any cells where the problem name is not found in available questions
    mask = kc_mask | prob_mask | step_mask
    kc.loc[mask, [f"KC ({kcm})" for kcm in kc_names]] = None

    # Adjust old-to-new KC mappings
    old_to_new_kc = old_to_new_kc or {}
    old_to_new_kc = {f"KC ({key})": f"KC ({val})" for key, val in old_to_new_kc.items()}
    default_mapping = {f"KC ({kcm})": f"KC ({kcm.replace(' ', '-')}-{new_kc_suffix})" for kcm in kc_names}
    old_to_new_kc = default_mapping | old_to_new_kc

    # Rename and save KC models
    kc.rename(columns=old_to_new_kc, inplace=True)
    if save_to_file:
        kc.to_csv(f"{os.path.splitext(kc_path)[0]}-new.txt", sep="\t", index=False)
    return kc


def get_step_to_kc(kc: pd.DataFrame) -> dict[str, str]:
    """Create a dictionary mapping step names to KC labels"""
    steps, labels = [], []
    for step, label in kc[["ds-step-name", "KC"]].itertuples(index=False):
        step = step.split("~")
        steps.extend(step)
        labels.extend([label] * len(step))
    step_to_kc = dict(zip(steps, labels, strict=True))
    return step_to_kc


def create_datashop_kc(kc: str | pd.DataFrame, kc_temp: str | pd.DataFrame,
                       kc_cols: list[str], new_kc_names: list[str] = None,
                       match_other_kc: bool = False, drop_other_kc: bool = False,
                       ignore_ph: bool = False) -> pd.DataFrame:
    """
    Populate a custom Datashop KC model
    :param kc: Either a path to a human-readable KC model or a pd.DataFrame of such
    :param kc_temp: Either a path to DataShop KC template file (empty or filled),
            e.g., "data/datashop/ds5426-elearning/kc_temp.txt", or a DataFrame of a loaded template
    :param kc_cols: A list of column names in `kc` to be used as KCs
    :param new_kc_names: New names for the KCs; if None, use `kc_cols` as names
    :param match_other_kc: Whether to match other KC models in the template;
            if True, the new KC model will not map to steps that are not mapped by other KC models
    :param drop_other_kc: Whether to drop other KC models in the template
    :param ignore_ph: Whether to ignore the "Problem Hierarchy" column when matching KCs
    :return: The new DataShop KC model as a DataFrame
    """
    # Load KC model
    if isinstance(kc, str):
        kc = pd.read_csv(kc)
    assert isinstance(kc, pd.DataFrame), "Incorrect type for 'kc'"
    if ignore_ph:
        kc = kc.drop(columns="ds-problem-hierarchy")

    # Load KC template
    if isinstance(kc_temp, str):
        kc_temp = load_datashop_temp(kc_temp)
    assert isinstance(kc_temp, pd.DataFrame), "Incorrect type for 'kc_temp'"

    kc_mask = None
    if match_other_kc:
        kc_mask = kc_temp.filter(regex=KC_PAT).isna().any(axis=1)  # match other KC models if any

    if drop_other_kc:
        other_kc_cols = kc_temp.filter(regex=KC_PAT).columns
        kc_temp = kc_temp.drop(columns=other_kc_cols)  # drop other KC models if any

    # These columns are used as keys when merging the KCs into the template
    key_cols = {
        "ds-problem-hierarchy": "Problem Hierarchy",
        "ds-problem-name": "Problem Name",
        "ds-step-name": "Step Name"
    }

    # Expand each key column if it contains multiple values separated by "~"
    for col in key_cols:
        if col in kc:
            kc[col] = kc[col].str.split("~")
            kc = kc.explode(col, ignore_index=True)

    # Grab all available key columns
    df = kc.filter(items=list(key_cols), axis="columns").rename(columns=key_cols)
    keys = df.columns.tolist()

    # Grab all KC columns
    new_kc_names = new_kc_names or kc_cols
    kc_cols = dict(zip(kc_cols, (f"KC ({name})" for name in new_kc_names), strict=True))
    kc_df = kc.filter(items=list(kc_cols), axis="columns").rename(columns=kc_cols)
    df = pd.concat([df, kc_df], axis="columns").set_index(keys)

    # Merge KC columns into the template
    kc_temp = kc_temp.join(df, on=keys, how="left")

    # Align with other KCs
    if kc_mask is not None:
        kc_temp.loc[kc_mask, list(kc_cols.values())] = None

    return kc_temp


def merge_student_step_with_kc(ss_path: str, kc: str | pd.DataFrame,
                               minimal: bool = False, multiplier: int = 1) -> pd.DataFrame:
    """
    This function inserts (multiple) KC models contained in a DataShop KC template into a student-step file.
    In particular, it can prepare KC models for multi-run cross-validation in LearnSphere by duplicating requisite columns.
    If `minimal=False` and `multiplier=1`, it inserts KC models into a DataShop student-step file similar to what DataShop does.
    KC template -> Student Step -> Student Step with duplicate columns
    :param ss_path: Path to a student-step file
    :param kc: Either a path to a filled KC template or a DataFrame containing KC models
    :param minimal: Whether to retain the essential columns only
    :param multiplier: Duplicate the KC columns by `multiplier` times
    :return: A student-step file with KC models inserted, ready for evaluation
    """
    minimal_cols = ["Anon Student Id", "Problem Hierarchy",
                    "Problem Name", "Step Name", "First Transaction Time", "First Attempt"]  # required columns
    key_cols = ["Problem Hierarchy", "Problem Name", "Step Name"]  # primary-key columns

    # Identify all KC models
    if isinstance(kc, str):
        kc = load_datashop_temp(kc)
    assert isinstance(kc, pd.DataFrame), "Incorrect type for 'kc'"

    kc["Problem Hierarchy"] = kc["Problem Hierarchy"].str.replace("(", "").str.replace(")", "")
    kc_cols = kc.set_index(key_cols).filter(regex=KC_PAT)
    kc_names = [re.match(KC_PAT, col).group("name") for col in kc_cols.columns]
    if minimal:  # Transcribe KC labels to minimize file size
        for col in kc_cols:
            mask = kc_cols[col].isna()
            kc_cols[col] = [f"KC-{lbl}" for lbl in LabelEncoder().fit_transform(kc_cols[col])]
            kc_cols.loc[mask, col] = None  # Keep the NaN values as they were

    # Load student-step data
    ss = pd.read_csv(ss_path, sep="\t", dtype={"Anon Student Id": str}, usecols=minimal_cols)
    ss["Problem Hierarchy"] = ss["Problem Hierarchy"].str.replace("(", "").str.replace(")", "")

    # Merge KCs into student-step
    ss = pd.merge(ss, kc_cols, how="left", on=key_cols, validate="many_to_one")

    if minimal:  # Transcribe columns to minimize file size
        ss["Anon Student Id"] = [f"ST-{lbl}" for lbl in LabelEncoder().fit_transform(ss["Anon Student Id"])]
        ss["Problem Hierarchy"] = [f"PH-{lbl}" for lbl in LabelEncoder().fit_transform(ss["Problem Hierarchy"])]
        ss["Problem Name"] = [f"PN-{lbl}" for lbl in LabelEncoder().fit_transform(ss["Problem Name"])]
        ss["Step Name"] = [f"SN-{lbl}" for lbl in LabelEncoder().fit_transform(ss["Step Name"])]

    # Initialize opportunity columns (object dtype: the column mixes "" for
    # steps without the KC and int counts, which pandas >= 3 forbids under
    # its default str dtype)
    for idx, kcm in enumerate(reversed(kc_names)):
        ss.insert(ss.shape[1] - 2 * idx, f"Opportunity ({kcm})", pd.Series("", index=ss.index, dtype=object))

    # Calculate opportunity
    opps = defaultdict(lambda: defaultdict(int))
    for idx, row in ss.iterrows():
        for kcm in kc_names:
            kc_col, opp_col = f"KC ({kcm})", f"Opportunity ({kcm})"
            if isinstance(row[kc_col], str):
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
    return ss
