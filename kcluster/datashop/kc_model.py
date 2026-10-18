# -*- coding: utf-8 -*-
"""
This module contains functions for working with DataShop Knowledge Component (KC) models.

A KC model reaches DataShop's rows through one join, :func:`join_kc_model`:
onto a student-step file (``kcluster tag``) or onto a KC-model export, to
import back into DataShop (:func:`fill_kc_template`).
"""
import csv
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from kcluster.datashop.student_step import DS_KEY_FIELDS, MULTI_KC_SEP, check_coverage, covered_rows
from kcluster.io.jsonl import load_questions

KC_PAT = r"KC \((?P<name>.+?)(-\d+)?\)"

#: A KC model's column in a KC-model export. A step with several KCs repeats
#: its model's column, one KC in each, which pandas reads back as
#: ``KC (<model>).1``, ``KC (<model>).2`` and so on.
TEMPLATE_KC_COLUMN = re.compile(r"^KC \((?P<name>.+)\)(\.\d+)?$")


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


def load_kc_template(path: str) -> pd.DataFrame:
    """Read a DataShop KC-model export cell for cell, to be filled and imported back.

    DataShop's import wants every column but the KC models back as exported, so
    unlike :func:`load_datashop_temp` nothing is parsed, filled, unquoted or
    dropped: every cell is a string, an empty cell ``""``. The exception is the
    nameless, empty column the trailing tab on DataShop's header line makes.
    """
    template = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    nameless = [col for col in template.columns
                if col.startswith("Unnamed: ") and template[col].fillna("").eq("").all()]
    return template.drop(columns=nameless)


def save_kc_template(df: pd.DataFrame, path: str) -> None:
    """Write a filled KC-model export for DataShop's import, every cell as it is.

    Nothing is quoted: DataShop splits each line on tabs and takes the cells as
    they come, so a KC label with quotes in it (``the "carrots and sticks"
    approach``) must go out as it is, or the quoting becomes part of its name.
    A tab or line break in a cell raises instead.
    """
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE)


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
            kc = kc.assign(**{col: kc[col].str.split("~")}).explode(col, ignore_index=True)

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


def join_kc_model(steps: pd.DataFrame, kc: pd.DataFrame, name: str, universe: pd.Series) -> pd.DataFrame:
    """Join one KC model onto DataShop's rows as ``KC (name)``, blank outside ``universe``.

    The rows are a student-step file's or a KC-model export's; either way each
    one in the universe must resolve to exactly one question of the model, on
    the ``ds-*`` keys the questions carry, and take a label from it. A row
    outside the universe may match nothing, and is left blank.

    :param steps: the rows to label, with DataShop's step columns
    :param kc: a KC CSV frame: one question per row, its ``ds-*`` keys and ``KC``
    :param name: the KC model's name
    :param universe: the rows the model must label, as a mask over ``steps``
    :return: a new frame, ``steps`` plus the ``KC (name)`` column
    """
    # Unmatched, ambiguous and unkeyed rows raise. The flattened CSV rows
    # duck-type Question here.
    check_coverage(kc.to_dict("records"), steps.loc[universe])

    fields = [field for field in DS_KEY_FIELDS if field in kc.columns and DS_KEY_FIELDS[field] in steps.columns]
    join_kc = kc[fields + ["KC"]].copy()
    for field in fields:
        # Collapse self-duplicates ("a~a", one question listing a step twice):
        # the explode-join would otherwise duplicate its rows.
        join_kc[field] = join_kc[field].str.split("~").map(lambda parts: "~".join(dict.fromkeys(parts)))

    n_rows = len(steps)
    steps = create_datashop_kc(join_kc, steps, kc_cols=["KC"], new_kc_names=[name])
    assert len(steps) == n_rows, "the step-to-question join duplicated rows"

    col = f"KC ({name})"
    unlabeled = universe & (steps[col].isna() | steps[col].astype(str).str.strip().eq(""))
    if unlabeled.any():
        raise ValueError(f"KC model {name!r} leaves {int(unlabeled.sum())} expert-tagged row(s) without a "
                         "label. Every KC model must cover the same rows or fits under them are incomparable; "
                         "rebuild the model from this dataset's questions.")
    steps[col] = steps[col].where(universe, "")
    return steps


def template_models(template: pd.DataFrame) -> dict[str, pd.Series]:
    """The KC models in a DataShop KC-model export, each with the steps it tags.

    A model that tags no step is left out: that is the placeholder column
    (``KC (new KC model name)``) the export adds for a new model.
    """
    tagged: dict[str, pd.Series] = {}
    for col in template.columns:
        if match := TEMPLATE_KC_COLUMN.match(col):
            steps = template[col].fillna("").astype(str).str.strip().ne("")
            name = match["name"]
            tagged[name] = tagged[name] | steps if name in tagged else steps
    return {name: steps for name, steps in tagged.items() if steps.any()}


def template_universe(template: pd.DataFrame) -> pd.Series:
    """The steps every KC model in a DataShop KC-model export tags (all of them, if it has none).

    Where the models disagree, that is the steps they all tag: the export of an
    expert model alongside DataShop's own ``Unique-step``, say, which leaves out
    the steps too few students reached.
    """
    universe = pd.Series(True, index=template.index)
    for steps in template_models(template).values():
        universe &= steps
    return universe


def fill_kc_template(template: pd.DataFrame, kc_models: dict[str, pd.DataFrame],
                     restrict_suffix: str | None = None) -> pd.DataFrame:
    """Fill a DataShop KC-model export with KC models, to import back into DataShop.

    DataShop imports KC models from a file shaped like its KC-model export: one
    row per step, matched on ``Step ID``, a ``KC (<model>)`` column per model,
    and every other column as exported. The steps the export's own models all
    tag are the comparable universe (:func:`template_universe`): every model
    joined here must label each of them and is blanked on the rest, so DataShop
    fits all of them over the same steps. The export's own models are then left
    out, as DataShop advises, since importing them again would overwrite them;
    a new model of the same name raises for that reason.

    A question bank can reach fewer steps than the export's models tag: the
    multiple-choice questions of a course whose expert model covers every
    question type, say. That raises, unless ``restrict_suffix`` is given. The
    universe is then cut down to the steps the new models' questions reach, and
    each of the export's own models goes out again cut down the same, as
    ``<model>-<restrict_suffix>``, so that DataShop still has the models to
    compare over one set of steps.

    :param template: a KC-model export (:func:`load_kc_template`)
    :param kc_models: model name -> KC CSV frame, joined in order
    :param restrict_suffix: label only the steps the new models' questions
        reach, and name the cut-down copies of the export's models by it
    :return: a new frame: the export's other columns unchanged, then the
        cut-down copies if any, then one ``KC (<model>)`` column per new model
    """
    if missing := [col for col in ("Step ID", "Step Name") if col not in template.columns]:
        raise ValueError(f"missing column(s) {missing}: not a DataShop KC-model export")
    if restrict_suffix and not kc_models:
        raise ValueError("no KC models to say which steps their questions reach")
    existing = template_models(template)
    copies = {name: f"{name}-{restrict_suffix}" for name in existing} if restrict_suffix else {}
    new_names = [*copies.values(), *kc_models]
    if taken := [name for name in new_names if name in existing]:
        raise ValueError(f"KC model(s) {taken} are already in the template, and importing them again would "
                         "overwrite them in DataShop. Rename them, or export the template without them if "
                         "overwriting is what you want.")
    if twice := [name for name in kc_models if name in copies.values()]:
        raise ValueError(f"KC model(s) {twice} are named like the cut-down copies of the template's own models")
    for name, kc in kc_models.items():
        if multi := int(kc["KC"].astype(str).str.contains(MULTI_KC_SEP, regex=False).sum()):
            raise ValueError(f"KC model {name!r} gives {multi} question(s) several KCs ({MULTI_KC_SEP!r}-joined). "
                             "DataShop's import takes one KC per column, repeating the model's column for "
                             "each extra KC, which a filled template does not do.")

    template = template.reset_index(drop=True)
    universe = template_universe(template)
    for name, kc in kc_models.items():
        reached = covered_rows(kc.to_dict("records"), template)
        if restrict_suffix:
            universe &= reached
        elif (unreached := universe & ~reached).any():
            raise ValueError(f"KC model {name!r} has no question for {int(unreached.sum())} of the "
                             f"{int(universe.sum())} step(s) to label, e.g. Step ID "
                             f"{template.loc[unreached, 'Step ID'].iloc[0]!r}. Models fitted over different "
                             "steps cannot be compared. If the question bank is meant to reach fewer steps "
                             "than the template's KC models tag, set restrict_suffix (--restrict_suffix): the "
                             "template's models are then imported again under it, cut down to the bank's steps.")
    if not universe.any():
        raise ValueError("no step is left to label: the template's KC models and the questions share none")

    own = [col for col in template.columns if TEMPLATE_KC_COLUMN.match(col)]
    filled = template.drop(columns=own)
    n_other = filled.shape[1]
    for name, kc in kc_models.items():
        filled = join_kc_model(filled, kc, name, universe)

    # A model's repeated columns keep one name, as DataShop wrote them
    copied = [col for col in own if TEMPLATE_KC_COLUMN.match(col)["name"] in copies]
    cut_down = template[copied].where(universe, "")
    cut_down.columns = [f"KC ({copies[TEMPLATE_KC_COLUMN.match(col)['name']]})" for col in copied]
    return pd.concat([filled.iloc[:, :n_other], cut_down, filled.iloc[:, n_other:]], axis=1)


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
        kc = load_kc_template(kc)
    assert isinstance(kc, pd.DataFrame), "Incorrect type for 'kc'"

    kc["Problem Hierarchy"] = kc["Problem Hierarchy"].str.replace("(", "").str.replace(")", "")
    # A blank cell is a step without a KC (DataShop writes it as one space), and
    # a model without any is the empty placeholder of a KC-model export
    kc_cols = kc.set_index(key_cols).filter(regex=KC_PAT).replace(r"^\s*$", np.nan, regex=True)
    kc_cols = kc_cols.dropna(axis="columns", how="all")
    kc_names = [re.match(KC_PAT, col).group("name") for col in kc_cols.columns]
    if minimal:  # Transcribe KC labels to minimize file size
        for col in kc_cols:
            mask = kc_cols[col].isna()
            kc_cols[col] = [f"KC-{lbl}" for lbl in LabelEncoder().fit_transform(kc_cols[col])]
            kc_cols.loc[mask, col] = None  # Keep the NaN values as they were

    # Load student-step data as text, the way the KC models are read: DataShop's
    # files are unquoted, so a step name may begin with a quote
    ss = pd.read_csv(ss_path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                     usecols=minimal_cols)
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
