"""Compute question/concept embeddings and their cosine KC models.

Runs on a conforming result dir from either engine (D10): the Concept KC in
``kc/`` supplies the concepts and cluster labels, the root ``args-*.json``
supplies the questions. Embeddings land in ``mat/embed/`` and their KC models
in ``kc/``, so the command is additive and idempotent — point it at an
existing result dir to fill in whichever models its flags enable. This is
how Vertex runs gain embedding models: the serving container has no
embedding endpoint, but ``--sent_path`` needs no GPU.

Three models (D10):

- ``sbert``: SentenceTransformer over ``str(question)``;
- ``llm``: the LLM's own encoding of ``str(question)``, conditioned on the
  congruity marginal context (``Question-emb`` in EDM 2025) — needs a GPU;
- ``concept``: SentenceTransformer over the extracted concept phrases
  (``Concept-emb`` in EDM 2025).

Both question encoders embed the same rendering, so their KC models differ
only in the encoder.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from kcluster.engine.local import LargeLangModel
from kcluster.io.jsonl import load_questions
from kcluster.paths import embed_dir, kc_dir, prepare_output_dir, run_dir
from kcluster.tasks.cluster import create_kc, sim_from_embeddings
from kcluster.tasks.concept import extract_question_embeds


def _as_numpy(embeds) -> np.ndarray:
    return embeds.cpu().numpy() if isinstance(embeds, torch.Tensor) else np.asarray(embeds)


def _save_and_cluster(name: str, embeds: np.ndarray, ds: str, concept_df, questions, out_kc: str, out_embed: str):
    np.save(os.path.join(out_embed, f"{ds}_{name}-embed.npy"), embeds)
    print(f"*** Building KCs based on {name}, metric='cosine' ***")
    kc = create_kc(concept_df, questions, sim_from_embeddings(embeds, metric="cosine"))
    if isinstance(kc, pd.DataFrame):
        kc.to_csv(os.path.join(out_kc, f"{ds}_{name}-cosine-kc.csv"), index=False)
        print(f"*** Finished with {kc['KC'].nunique()} KCs ***")


def main(args):
    result_dir = getattr(args, "result_dir", None) or run_dir(getattr(args, "run_dir", None))
    if not result_dir:
        raise SystemExit("--result_dir is required unless --run_dir (or KCLUSTER_RUN_DIR) is set")
    args.result_dir = result_dir = os.path.abspath(result_dir)
    if not getattr(args, "sent_path", None) and not getattr(args, "llm_path", None):
        raise SystemExit("Nothing to do: pass --sent_path and/or --llm_path")
    print(f"*** Writing results to {result_dir} ***")

    # The Concept KC anchors everything: its rows are the questions, its
    # phrases are what the concept model embeds
    out_kc = kc_dir(result_dir)
    match = glob.glob("*_concept-kc.csv", root_dir=out_kc)
    if len(match) != 1:
        raise SystemExit(f"Expected exactly one *_concept-kc.csv in {out_kc}, found {len(match)} — "
                         "run the concept step (or vertex-build-kc) into this result dir first")
    [fname] = match
    ds = fname.removesuffix("_concept-kc.csv")
    concept_df = pd.read_csv(os.path.join(out_kc, fname))

    # Recover the questions; any step's breadcrumb records the data path.
    # --data_path overrides it, because a result dir is often embedded on a
    # different machine than it was built on (a Vertex run launched from a
    # laptop records a path that does not exist on a cluster).
    if not getattr(args, "data_path", None):
        args_files = sorted(glob.glob(f"args-*-{ds}.json", root_dir=result_dir))
        recorded = next((p for f in args_files
                         if (p := json.load(open(os.path.join(result_dir, f))).get("data_path"))), None)
        if not recorded:
            raise SystemExit(f"No data_path recorded in args-*-{ds}.json under {result_dir} — "
                             "pass --data_path with the question file")
        args.data_path = recorded
    if not os.path.isfile(args.data_path):
        raise SystemExit(f"Question file not found: {args.data_path} — pass --data_path with a "
                         "reachable copy of this dataset's questions")
    questions = load_questions(args.data_path)
    # Row i of the Concept KC is question i, and that is what makes the
    # embeddings joinable to it; a different bank would silently mislabel.
    if len(questions) != len(concept_df):
        raise SystemExit(f"{args.data_path} has {len(questions)} questions, but the Concept KC has "
                         f"{len(concept_df)} rows — wrong question file for this result dir")

    out_embed = prepare_output_dir(embed_dir(result_dir))

    if sent_path := getattr(args, "sent_path", None):
        model = SentenceTransformer(sent_path, local_files_only=True)
        with torch.inference_mode():
            q_embeds = _as_numpy(model.encode([str(q) for q in questions]))
            c_embeds = _as_numpy(model.encode(concept_df["KC"].tolist()))
        _save_and_cluster("sbert", q_embeds, ds, concept_df, questions, out_kc, out_embed)
        _save_and_cluster("concept", c_embeds, ds, concept_df, questions, out_kc, out_embed)

    if llm_path := getattr(args, "llm_path", None):
        llm = LargeLangModel(llm_path, trust_remote_code=True, torch_dtype=torch.float16)
        q_embeds = _as_numpy(extract_question_embeds(llm, questions, args.batch_size))
        _save_and_cluster("llm", q_embeds, ds, concept_df, questions, out_kc, out_embed)

    # Save arguments
    with open(os.path.join(result_dir, f"args-embed-{ds}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def add_arguments(parser):
    parser.add_argument("--result_dir", default=argparse.SUPPRESS, type=str,
                        help="The result directory holding the Concept KC (default: --run_dir)")
    parser.add_argument("--run_dir", default=argparse.SUPPRESS, type=str,
                        help="Result folder shared by every step of this run (env: KCLUSTER_RUN_DIR)")
    parser.add_argument("--sent_path", default=argparse.SUPPRESS, type=str,
                        help="Path to a SentenceTransformer; builds the sbert-cosine and "
                             "concept-cosine models (CPU is fine)")
    parser.add_argument("--llm_path", default=argparse.SUPPRESS, type=str,
                        help="Path to a downloaded LLM; builds the llm-cosine model (needs a GPU)")
    parser.add_argument("--data_path", default=argparse.SUPPRESS, type=str,
                        help="Question file for this result dir (default: the path recorded in "
                             "args-*.json, which may not exist on this machine)")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
