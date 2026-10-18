import csv
import glob
import os
import re

import numpy as np
import pandas as pd

from kcluster.core.pmi import PointwiseMutualInfo
from kcluster.datashop.kc_model import (
    KC_PAT,
    load_kc_template,
    merge_student_step_with_kc,
    save_kc_template,
    template_models,
)
from kcluster.datashop.student_step import step_questions
from kcluster.io.jsonl import load_questions
from kcluster.paths import (
    concept_kc_path,
    embed_dir,
    pmi_raw_dir,
    prepare_output_dir,
    question_path,
    required_result_dir,
    result_dataset,
)
from kcluster.tasks.cluster import create_kc, sim_from_embeddings, split_collisions
from kcluster.tasks.tag import load_kc_csv


def main(args):
    # The result dir (D10 layout) holds the concept KC, embeddings, and shards
    result_dir = required_result_dir(args.result_dir, args.run_dir)
    output_dir = prepare_output_dir(os.path.join(result_dir, "kc-refine"), exist_ok=False)
    print(f"*** Writing results to {output_dir} ***")

    # Load the KC model to refine, cell for cell: DataShop's files are unquoted
    # text, so a label with quotes in it is the KC's name as it stands
    kc = load_kc_template(args.kc_path)
    kcm = kc_model_name(kc, args.kc_val_path, args.kc_model)
    kc = kc.drop(columns=[col for col in kc.columns if re.match(KC_PAT, col) and col != f"KC ({kcm})"])

    # Its KC values: the KCs that show no learning are the ones to refine
    kc_val = pd.read_csv(args.kc_val_path)
    val_mask = (kc_val["Slope"].le(0.001)) & (kc_val["Intercept (probability) at Opportunity 1"].between(0.2, 0.8))

    # Load concepts (the concept step writes them under kc/)
    concept_df = load_kc_csv(concept_kc_path(result_dir))
    assert concept_df["KC"].str.strip().all(), "Some concepts are invalid"
    concept_records = concept_df.to_dict("records")  # rows of the flattened questions

    # Recover the questions behind the concepts (--data_path overrides the recorded path)
    questions = load_questions(question_path(result_dir, result_dataset(result_dir), args.data_path))

    # Build the two similarity matrices once
    [fname] = glob.glob("*_llm-embed.npy", root_dir=embed_dir(result_dir))
    embeds = np.load(os.path.join(embed_dir(result_dir), fname))
    assert embeds.shape[0] == len(questions), \
        f"Expected {len(questions)} questions, got {embeds.shape[0]} embeddings"
    q_cos_sim = sim_from_embeddings(embeds, metric="cosine")

    raw_dir = args.pmi_dir or pmi_raw_dir(result_dir)
    pmi = PointwiseMutualInfo.from_shards(raw_dir, len(questions), len(questions),
                                          normalize=True, symmetric=True)
    pmi_sim = pmi.pmi_mat

    count = 0
    for kc_label, num_steps in kc_val.loc[val_mask, ["KC Name", "Number of Unique Steps"]].itertuples(index=False):
        print(f"*** Refining KC '{kc_label}' ***")

        label_mask = kc[f"KC ({kcm})"].eq(kc_label)
        assert label_mask.sum() == num_steps, "Inconsistent number of steps"

        # The question behind each of its steps, keyed on every ds-* field the
        # questions carry, as kcluster tag keys them: a step name alone can
        # recur under several problems
        step_ids = step_questions(concept_records, kc.loc[label_mask])
        in_kc = concept_df["id"].isin(set(step_ids))
        if in_kc.sum() <= 2:  # skip refinement if there are two or fewer questions
            print("Too few questions tagged with this KC, skipping refinement. ")
            continue

        # Refinement using 'concept'
        concept_kc = concept_df[in_kc]

        def predicate(q, ids=frozenset(step_ids)):  # bind the loop variable
            return str(q["id"]) in ids

        # Refinement using 'question-cosine'
        print("*** Refining KCs based on question, metric='cosine' ***")
        q_cos_kc = create_kc(concept_kc, questions, q_cos_sim, predicate=predicate)
        assert q_cos_kc is not None, "Failed to create KCs with question-cosine"

        # Refinement using PMI
        print("*** Refining KCs with KCluster-PMI ***")
        pmi_kc = create_kc(concept_kc, questions, pmi_sim, predicate=predicate)
        assert pmi_kc is not None, "Failed to create KCs with KCluster-PMI"

        for item, df in (("cpt", concept_kc), ("qcos", q_cos_kc), ("pmi", pmi_kc)):
            col = f"KC ({item}-{kc_label})"
            kc[col] = kc[f"KC ({kcm})"]
            kc.loc[label_mask, col] = refined_labels(kc[f"KC ({kcm})"], label_mask, step_ids, df)
            print(f"*** {item}: {kc.loc[label_mask, col].nunique()} KC(s) ***")
        count += 1

    print(f"*** Refined {count} KCs in {kcm} ***")
    kc_path = os.path.join(output_dir, "refined-kc.txt")
    save_kc_template(kc, kc_path)

    # Merge KCs into student step (for cross-validation) if a path is present
    if args.ss_path:
        print("*** Merging KCs with student steps ***")
        ss = merge_student_step_with_kc(args.ss_path, kc, minimal=args.minimal, multiplier=args.multiplier)
        fname = os.path.splitext(os.path.basename(kc_path))[0]
        fname = f"{fname}-merged-minimal={args.minimal}-multiplier={args.multiplier}.txt"
        ss.to_csv(os.path.join(output_dir, fname), sep="\t", index=False, quoting=csv.QUOTE_NONE)


def kc_model_name(kc: pd.DataFrame, kc_val_path: str, explicit: str | None = None) -> str:
    """The model whose KC values ``kc_val_path`` holds, which the template must carry.

    ``kcluster fit`` names the file after the model (``kc-values/<model>.csv``);
    a hand-named file puts the model before the first underscore
    (``LOs_kc-values.csv``). ``--kc_model`` names it outright.
    """
    stem = os.path.splitext(os.path.basename(kc_val_path))[0]
    for name in (explicit,) if explicit else (stem, stem.split("_")[0]):
        if f"KC ({name})" in kc:
            return name
    raise SystemExit(f"No KC model {explicit or stem!r} in the template (it has {list(template_models(kc))}); "
                     "name the one these KC values are for with --kc_model")


def refined_labels(model: pd.Series, in_kc: pd.Series, step_ids: pd.Series, refinement: pd.DataFrame) -> pd.Series:
    """The steps of one KC of ``model`` (``in_kc``), relabelled by a refinement of its questions.

    Each step takes the label of its own question (``step_ids``, from
    ``step_questions``).

    A refinement labels its KCs by concept, and so does a KCluster model, so
    the labels can coincide where the KCs do not. Two clusters whose exemplars
    share a concept are told apart by cluster id, as build-kc's split models
    are (D14); a part labelled like another KC of the model takes its parent's
    label in brackets, so it does not merge with that KC. A part may keep the
    parent's own label: no other step carries it.
    """
    if "KC-raw" in refinement and (split := split_collisions(refinement)) is not None:
        refinement = split
    labels = step_ids.map(dict(zip(refinement["id"].astype(str), refinement["KC"], strict=True)))
    assert labels.notna().all(), "A refined step has no label"
    elsewhere = labels.isin(set(model[~in_kc]))
    return labels.where(~elsewhere, labels + " [" + model[in_kc] + "]")


def add_arguments(parser):
    parser.add_argument("--kc_path", required=True, type=str, help="Path to a DataShop KC template file")
    parser.add_argument("--kc_val_path", required=True, type=str,
                        help="KC values of the model to refine, as kcluster fit writes them (kc-values/<model>.csv)")
    parser.add_argument("--kc_model", default=None, type=str,
                        help="The model the KC values are for (default: from their file name)")
    parser.add_argument("--result_dir", default=None, type=str,
                        help="The result directory holding the concept KC, embeddings, and score shards "
                             "(default: --run_dir)")
    parser.add_argument("--pmi_dir", default=None, type=str,
                        help="Directory of raw score shards (default: <result_dir>/mat/pmi/raw)")
    parser.add_argument("--data_path", default=None, type=str,
                        help="Question file (default: the path recorded in args-*.json)")
    parser.add_argument("--run_dir", default=None, type=str,
                        help="Result folder shared by every step of this run (env: KCLUSTER_RUN_DIR)")
    parser.add_argument("--ss_path", default=None, type=str, help="Path to a DataShop student-step file")
    parser.add_argument("--minimal", action="store_true", help="Whether to minimize the merged student-step file")
    parser.add_argument("--multiplier", default=1, type=int,
                        help="Number of times to duplicate each KC model for cross-validation")
