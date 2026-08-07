import os

import pandas as pd

from kcluster.io.datashop import load_datashop_temp


def extract_data(stu_data_path: str,
                 problem_dir_prefix: str = "webwork/raw_data",
                 expected_records: int = 27071) -> tuple[pd.DataFrame, pd.DataFrame]:
    """This function extracts and cleans Rogawski Calculus problems from raw student data."""
    df = pd.read_csv(stu_data_path, dtype=str)

    # Apply corrections to the "Problem Path" column
    df["Problem Path"] = df["Problem Path"].str.replace("_Offering", "")
    df["Problem Path"] = df["Problem Path"].str.replace("2.6_Trigonometric Limits", "2.6_Trigonometric_Limits")
    df["Problem Path"] = df["Problem Path"].str.replace("4.5_L'Hopital's_Rule", "4.5_LHopitals_Rule")
    df["Problem Path"] = df["Problem Path"].str.replace("4.8_Newton's_Method", "4.8_Newtons_Method")
    df["Problem Path"] = df["Problem Path"].str.replace("17.1_Green's_Theorem", "17.1_Greens_Theorem")
    df["Problem Path"] = df["Problem Path"].str.replace("17.2_Stokes'_Theorem", "17.2_Stokes_Theorem")

    # Select student activities related to Rogawski Calculus problems
    mask = (
            (df["Permission Level"] == "student") &
            (df["Problem Path"].str.startswith("Rogawski_Calculus")) &
            (df["OPL Subject"].str.startswith("Calculus")) &
            (df["Problem Path"].apply(lambda t: os.path.isfile(os.path.join(problem_dir_prefix, t))))
    )

    students_df = df[mask].reset_index(drop=True)
    students_df["Answer Timestamp"] = students_df["Answer Timestamp"].astype(int)

    # Sort the transactions for each student by timestamp (selecting the
    # columns explicitly keeps the grouping column, which pandas >= 3 would
    # otherwise exclude from the applied frames)
    students_df = students_df.groupby("Student ID hash", sort=False, group_keys=False)[
        students_df.columns.tolist()].apply(lambda grp: grp.sort_values("Answer Timestamp"))

    # Fix empty values in "OPL Chapter" and "OPL Section"
    mask = (students_df["OPL Chapter"].isna()) & (students_df["Problem Path"].str.contains("12_Vector_Geometry"))
    students_df.loc[mask, "OPL Chapter"] = "Vector Geometry"
    students_df.loc[mask, "OPL Section"] = "Vectors"

    mask = (students_df["OPL Chapter"].isna()) & (students_df["Problem Path"].str.contains("5_The_Integral"))
    students_df.loc[mask, "OPL Chapter"] = "Integrals"
    students_df.loc[mask, "OPL Section"] = "Net Change as the Integral of a Rate"

    assert len(students_df) == expected_records, \
        f"Expected {expected_records} records, but found a different number: {len(students_df)}"

    # Extract problem content from the raw data directory
    problems = []
    for p in students_df["Problem Path"].unique():
        with open(os.path.join(problem_dir_prefix, p), "r") as f:
            problems.append({"Problem Path": p, "Problem Content": f.read()})
    problems_df = pd.DataFrame(problems)

    return students_df, problems_df


def create_datashop_transactions(clean_stu_data_path: str) -> pd.DataFrame:
    """This function creates a DataShop transaction DataFrame from cleaned student data."""
    students_df = pd.read_csv(clean_stu_data_path, dtype=str)
    num_blanks = students_df["Number of Answer Blanks"].astype(int) - 1

    # Gather answers
    answers = []
    for idx, row in students_df.filter(regex=r"Answer \d+ Value").iterrows():
        answers.append(row.iloc[:num_blanks[idx]].to_list())

    # Gather answer statuses
    statuses = []
    for idx, row in students_df.filter(regex=r"Answer \d+ Status").iterrows():
        statuses.append(row.iloc[:num_blanks[idx]].to_list())  # Ensure we only take the relevant number of statuses

    # Create step names
    blanks = []
    for status in statuses:
        blanks.append([f"Blank-{i}" for i in range(1, len(status) + 1)])

    # Add and drop columns
    cols_to_drop = students_df.filter(regex=r"Answer \d+").columns
    students_df = students_df.drop(columns=cols_to_drop)
    students_df["Answer Value"] = answers
    students_df["Answer Status"] = statuses
    students_df["Step Name"] = blanks

    # Expand the DataFrame to have one row per step
    students_df = students_df.explode(["Answer Value", "Answer Status", "Step Name"], ignore_index=True)

    # Create the transaction DataFrame
    trans_df = pd.DataFrame()
    trans_df["Anon Student Id"] = students_df["Student ID hash"].apply(lambda x: x[:32])
    trans_df["Session Id"] = trans_df["Anon Student Id"] + students_df["Answer Date"]
    trans_df["Time"] = students_df["Answer Timestamp"].astype(int) * 1000  # Convert to milliseconds
    trans_df["Student Response Type"] = "ATTEMPT"
    trans_df["Tutor Response Type"] = "RESULT"
    trans_df["Level (Subject)"] = "Calculus"  # students_df["OPL Subject"].str.replace(":", " -")
    trans_df["Level (Chapter)"] = students_df["OPL Chapter"]
    trans_df["Level (Section)"] = students_df["OPL Section"]
    trans_df["Problem Name"] = students_df["Problem Path"]
    trans_df["Problem View"] = students_df["Attempt Number"]
    # trans_df["Problem Start Time"] = students_df["Answer Timestamp"].astype(int)
    trans_df["Step Name"] = students_df["Step Name"]
    trans_df["Outcome"] = students_df["Answer Status"].map({"1": "CORRECT", "0": "INCORRECT"})
    trans_df["Input"] = students_df["Answer Value"]
    trans_df["CF (Problem Seed)"] = students_df["Problem Seed"]

    def extract_keywords(s: str) -> str:
        return "~~".join(sorted(t.strip().lower() for t in s.replace("'", "").split(","))).replace(" ", "_")

    trans_df["CF (Problem Keywords)"] = students_df["OPL Keywords"].apply(extract_keywords)

    return trans_df


def create_chapter_n_section_kc(kc_temp: str | pd.DataFrame) -> pd.DataFrame:
    """This function extracts chapter and section KCs from a DataShop KC template."""
    if isinstance(kc_temp, str):
        kc_temp = load_datashop_temp(kc_temp)

    def extract_hierarchy(s: str):
        import re
        pattern = r"\(Subject\) (.*), \(Chapter\) (.*), \(Section\) (.*)"
        if m := re.match(pattern, s):
            _, chapter, section = m.groups()
            return chapter, section
        return None, None

    kc_df = pd.DataFrame(kc_temp["Problem Hierarchy"].apply(extract_hierarchy).to_list(),
                         columns=["KC (Chapter)", "KC (Section)"])
    kc_temp = pd.concat([kc_temp, kc_df], axis=1)

    if "KC (Unique-step)" in kc_temp:
        mask = kc_temp["KC (Unique-step)"].isna()
        kc_temp.loc[mask, ["KC (Chapter)", "KC (Section)"]] = None
        kc_temp.drop(columns=["KC (Unique-step)"], inplace=True)

    return kc_temp


def create_keywords_kc(kc_temp: str | pd.DataFrame, trans_df: str | pd.DataFrame) -> pd.DataFrame:
    """This function creates a KC model based on the keywords of each question"""
    if isinstance(kc_temp, str):
        kc_temp = load_datashop_temp(kc_temp)
    if isinstance(trans_df, str):
        trans_df = pd.read_csv(trans_df, sep="\t")

    new_kc = trans_df[["Problem Name", "CF (Problem Keywords)"]].copy()
    new_kc = new_kc.drop_duplicates(subset=["Problem Name"], ignore_index=True)
    new_kc = new_kc.rename(columns={"CF (Problem Keywords)": "KC (Keywords)"})

    kc_temp = kc_temp.join(new_kc.set_index("Problem Name"), on="Problem Name", how="left", validate="many_to_one")
    if "KC (Unique-step)" in kc_temp:
        mask = kc_temp["KC (Unique-step)"].isna()
        kc_temp.loc[mask, "KC (Keywords)"] = None
        kc_temp.drop(columns=["KC (Unique-step)"], inplace=True)

    return kc_temp


def main(args):
    import logging
    import warnings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Extract student and problem data
    logging.info("Extracting student and problem data...")
    stu_df, prob_df = extract_data(args.raw_stu_data_path, args.problem_dir_prefix)

    # Save cleaned student data and problems
    clean_stu_path = os.path.join(args.data_dir, "Rogawski_student_data.csv")
    stu_df.to_csv(clean_stu_path, index=False)
    prob_df.to_csv(os.path.join(args.data_dir, "Rogawski_problems.csv"), index=False)

    logging.info("Creating DataShop transactions...")
    tx_df = create_datashop_transactions(clean_stu_path)
    tx_df.to_csv(os.path.join(args.data_dir, "ds-transactions.txt"), sep="\t", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_stu_data_path", type=str, required=True, help="Path to raw student data file")
    parser.add_argument("--problem_dir_prefix", type=str, default="webwork/raw_data",
                        help="The prefix for the problem directory")
    parser.add_argument("--data_dir", type=str, default="webwork/data/", help="Path to the cleaned data directory")

    cli_args = parser.parse_args()
    main(cli_args)
