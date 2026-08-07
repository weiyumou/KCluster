import glob
import os

from kcluster.io.datashop import KC_PAT, create_datashop_kc, save_datashop_temp
from kcluster.io.datashop import create_default_kc, create_kc_from_questions

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create DataShop JSON files from transaction data.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to question data")
    parser.add_argument("--kc_temp_path", type=str, required=True, help="Path to Datashop KC template file")
    args = parser.parse_args()

    # Create default KC models
    print("Creating Single-KC and Unique-step", end=" ... ")
    default_kc_df = create_default_kc(args.kc_temp_path, save_to_file=True)
    num_kcs = default_kc_df.filter(regex=KC_PAT).nunique().tolist()
    print(f"{num_kcs} KCs created")

    # Create KCs from objectives
    print("Create KCs from objectives", end=" ... ")
    q_kc_df = create_kc_from_questions(args.data_path, args.kc_temp_path,
                                       kc_fields=["objectives"], kc_names=["Objectives"],
                                       save_to_file=True, ignore_ph=True)
    num_kcs = q_kc_df.filter(regex=KC_PAT).nunique().tolist()
    print(f"{num_kcs} KCs created")

    kc_temp = args.kc_temp_path
    kc_dir = os.path.dirname(kc_temp)
    new_kc_names = ["Concept", "KCluster-norm", "KCluster-unnorm"]
    for kc_file, kc_name in zip(sorted(glob.iglob(os.path.join(kc_dir, "kc", "*.csv"))), new_kc_names, strict=True):
        print(f"Creating KCs from {kc_file} as {kc_name}")
        kc_temp = create_datashop_kc(kc_file, kc_temp, kc_cols=["KC"], new_kc_names=[kc_name],
                                     match_other_kc=True, drop_other_kc=False, ignore_ph=True)
    save_datashop_temp(kc_temp, os.path.join(kc_dir, "all-custom-kc.txt"))
    num_kcs = kc_temp.filter(regex=KC_PAT).nunique().tolist()
    print(f"{num_kcs} KCs created")
