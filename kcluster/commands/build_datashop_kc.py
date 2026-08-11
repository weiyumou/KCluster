import argparse
import glob
import os
import re

import pandas as pd

from kcluster.io.datashop import (
    KC_PAT,
    get_step_to_kc,
    load_datashop_temp,
    merge_student_step_with_kc,
)
from kcluster.paths import prepare_output_dir, timestamp


def insert_kc_model(kc: str | pd.DataFrame, kc_temp: str | pd.DataFrame,
                    step_kc_path: str, new_kc_name: str,
                    match_other_kc: bool = True, drop_other_kc: bool = True) -> pd.DataFrame:
    """
    Insert one KCluster result into a DataShop template, keyed by step name.

    Unlike io.datashop.create_datashop_kc (which key-joins on the available
    ds-* columns), this maps by Step Name alone and additionally masks steps
    that the system-generated unique-step model excludes.
    :param kc: Either a path to a human-readable KC model or a pd.DataFrame of such
    :param kc_temp: Either a path to DataShop KC template file or a pd.DataFrame of a loaded template
    :param step_kc_path: A path to a (system-generated) unique-step KC model
    :param new_kc_name: The name given to the new KC model, e.g., "KCluster"
    :param match_other_kc: Whether to match other KC models in the template;
            if True, the new KC model will not map to steps that are not mapped by other KC models
    :param drop_other_kc: Whether to drop other KC models in the template
    :return: The new DataShop KC model as a DataFrame
    """
    # Load KC model
    if isinstance(kc, str):
        kc = pd.read_csv(kc)
    assert isinstance(kc, pd.DataFrame), "Incorrect type for 'kc'"

    # Load KC template
    if isinstance(kc_temp, str):
        kc_temp = load_datashop_temp(kc_temp)
    assert isinstance(kc_temp, pd.DataFrame), "Incorrect type for 'kc_temp'"

    kc_mask = False
    if match_other_kc:
        kc_mask = kc_temp.filter(regex=KC_PAT).isna().any(axis=1)  # match other KC models if any

    if drop_other_kc:
        kc_cols = kc_temp.filter(regex=KC_PAT).columns
        kc_temp.drop(columns=kc_cols, inplace=True)  # drop other KC models if any

    # Load the unique-step KC model
    step_kc = load_datashop_temp(step_kc_path)
    step_mask = step_kc["KC (Unique-step)"].isna()

    # Fill in KC
    step_to_kc = get_step_to_kc(kc)
    kc_temp[f"KC ({new_kc_name})"] = kc_temp["Step Name"].map(step_to_kc)
    kc_temp.loc[kc_mask | step_mask, f"KC ({new_kc_name})"] = None

    return kc_temp


def main(args):
    output_dir = prepare_output_dir(os.path.join(args.kc_dir, timestamp()), exist_ok=False)
    print(f"*** Writing results to {output_dir} ***")

    # Add KCs to the template
    kc_temp = args.kc_temp
    for fname in glob.iglob("*-kc.csv", root_dir=args.kc_dir):
        # Drop the dataset prefix (D10 filenames are <ds>_<model>-kc.csv) so
        # the DataShop model keeps its short name, e.g. "kcluster-unnorm"
        stem = os.path.splitext(fname)[0].split("_", 1)[-1]
        new_kc_name = re.match(r".+?(?=-kc)", stem).group(0)
        print(f"*** Adding KC '{new_kc_name}' to the template ***")
        kc = os.path.join(args.kc_dir, fname)
        kc_temp = insert_kc_model(kc, kc_temp, args.step_kc, new_kc_name,
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


def add_arguments(parser):
    parser.add_argument("--kc_dir", required=True, type=str, help="Path to a directory containing non-DataShop KCs")
    parser.add_argument("--kc_temp", required=True, type=str, help="Path to a DataShop KC template file")
    parser.add_argument("--step_kc", required=True, type=str, help="Path to a DataShop step-KC file")
    parser.add_argument("--ss_path", default=argparse.SUPPRESS, type=str, help="Path to a DataShop student-step file")
    parser.add_argument("--minimal", default=argparse.SUPPRESS, action="store_true",
                        help="Whether to minimize the merged student-step file")
    parser.add_argument("--multiplier", default=argparse.SUPPRESS, type=int,
                        help="Number of times to duplicate each KC model for cross-validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
