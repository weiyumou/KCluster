import argparse
import glob
import os
import re

from processing.util import merge_student_step_with_kc


def main(args):
    match args.dataset:
        case "elearning":
            from processing.elearning import create_datashop_kc
        case "spacing":
            from processing.spacing import create_datashop_kc
        case _:
            raise ValueError("Unknown dataset")

    output_dir = {
        "elearning": "data/datashop/ds5426-elearning",
        "spacing": "data/datashop/ds6160-spacing",
    }

    # Add KCs to the template
    kc_temp = args.kc_temp
    kc_dir = os.path.join("results/concept/", args.dataset, "kc")
    for fname in glob.iglob("*.csv", root_dir=kc_dir):
        new_kc_name = re.match(r".+?(?=-kc)", os.path.splitext(fname)[0]).group(0)
        print(f"*** Adding KC '{new_kc_name}' to the template ***")
        kc = os.path.join(kc_dir, fname)
        kc_temp = create_datashop_kc(kc_temp, kc, args.step_kc, new_kc_name)

    # Save the template
    kc_path = os.path.join(output_dir[args.dataset], "all-kc.txt")
    kc_temp.to_csv(kc_path, sep="\t", index=False)

    # Merge KCs into student step (for cross-validation) if a path is present
    if ss_path := getattr(args, "ss_path", None):
        multiplier = getattr(args, "multiplier", 1)
        print("*** Merging KCs with student steps ***")
        merge_student_step_with_kc(ss_path, kc_path, minimal=True, multiplier=multiplier, save_to_file=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True, type=str, choices=("elearning", "spacing"))
    parser.add_argument("--kc_temp", required=True, type=str, help="Path to a DataShop KC template file")
    parser.add_argument("--step_kc", required=True, type=str, help="Path to a DataShop step-KC file")
    parser.add_argument("--ss_path", default=argparse.SUPPRESS, type=str, help="Path to a DataShop student-step file")
    parser.add_argument("--multiplier", default=argparse.SUPPRESS, type=int,
                        help="Number of times to duplicate each KC model for cross-validation")

    cl_args = parser.parse_args()
    main(cl_args)
