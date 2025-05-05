import argparse
import glob
import os
import re
import time
from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from processing.elearning import create_datashop_kc, load_datashop_temp
from processing.util import KC_PAT


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

    # Initialize opportunity columns
    for idx, kcm in enumerate(reversed(kc_names)):
        ss.insert(ss.shape[1] - 2 * idx, f"Opportunity ({kcm})", "")

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


def main(args):
    output_dir = os.path.join(args.kc_dir, time.asctime()).replace(' ', '_').replace(':', '-')
    os.makedirs(output_dir, exist_ok=False)

    # Add KCs to the template
    kc_temp = args.kc_temp
    for fname in glob.iglob("*-kc.csv", root_dir=args.kc_dir):
        new_kc_name = re.match(r".+?(?=-kc)", os.path.splitext(fname)[0]).group(0)
        print(f"*** Adding KC '{new_kc_name}' to the template ***")
        kc = os.path.join(args.kc_dir, fname)
        kc_temp = create_datashop_kc(kc, kc_temp, args.step_kc, new_kc_name,
                                     match_other_kc=True, drop_other_kc=False)

    # Save the template
    kc_path = os.path.join(output_dir, "all-kc.txt")
    kc_temp.to_csv(kc_path, sep="\t", index=False)

    # Merge KCs into student step (for cross-validation) if a path is present
    if ss_path := getattr(args, "ss_path", None):
        multiplier = getattr(args, "multiplier", 1)
        minimal = getattr(args, "minimal", False)
        print("*** Merging KCs with student steps ***")
        ss = merge_student_step_with_kc(ss_path, kc_temp, minimal=minimal, multiplier=multiplier)
        fname = f"ss-merged-minimal={minimal}-multiplier={multiplier}.txt"
        ss.to_csv(os.path.join(output_dir, fname), sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kc_dir", required=True, type=str, help="Path to a directory containing non-DataShop KCs")
    parser.add_argument("--kc_temp", required=True, type=str, help="Path to a DataShop KC template file")
    parser.add_argument("--step_kc", required=True, type=str, help="Path to a DataShop step-KC file")
    parser.add_argument("--ss_path", default=argparse.SUPPRESS, type=str, help="Path to a DataShop student-step file")
    parser.add_argument("--minimal", default=argparse.SUPPRESS, action="store_true",
                        help="Whether to minimize the merged student-step file")
    parser.add_argument("--multiplier", default=argparse.SUPPRESS, type=int,
                        help="Number of times to duplicate each KC model for cross-validation")

    cl_args = parser.parse_args()
    main(cl_args)
