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

import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from kcluster.engine.local import LargeLangModel
from kcluster.io.jsonl import load_questions
from kcluster.paths import (
    concept_kc_path,
    embed_dir,
    kc_dir,
    prepare_output_dir,
    question_path,
    required_result_dir,
    result_dataset,
    save_args,
)
from kcluster.tasks.cluster import create_kc, save_kc_models, sim_from_embeddings
from kcluster.tasks.concept import extract_question_embeds


def _as_numpy(embeds) -> np.ndarray:
    return embeds.cpu().numpy() if isinstance(embeds, torch.Tensor) else np.asarray(embeds)


def _save_and_cluster(name: str, embeds: np.ndarray, ds: str, concept_df, questions, out_kc: str, out_embed: str):
    np.save(os.path.join(out_embed, f"{ds}_{name}-embed.npy"), embeds)
    print(f"*** Building KCs based on {name}, metric='cosine' ***")
    kc = create_kc(concept_df, questions, sim_from_embeddings(embeds, metric="cosine"))
    if kc is not None:
        save_kc_models(kc, os.path.join(out_kc, f"{ds}_{name}-cosine-kc.csv"))
        print(f"*** Finished with {kc['KC'].nunique()} KCs ***")


def main(args):
    args.result_dir = result_dir = required_result_dir(args.result_dir, args.run_dir)
    if not args.sent_path and not args.llm_path:
        raise SystemExit("Nothing to do: pass --sent_path and/or --llm_path")
    print(f"*** Writing results to {result_dir} ***")

    # The Concept KC anchors everything: its rows are the questions, its
    # phrases are what the concept model embeds
    ds = result_dataset(result_dir)
    concept_df = pd.read_csv(concept_kc_path(result_dir))
    # The two question encoders are embed-family; concept-cosine clusters the
    # concept phrases, so it files with the concept family (D15).
    out_embed_kc = prepare_output_dir(kc_dir(result_dir, "embed"))
    out_concept_kc = prepare_output_dir(kc_dir(result_dir, "concept"))

    # Recover the questions (--data_path overrides the recorded path)
    args.data_path = question_path(result_dir, ds, args.data_path)
    questions = load_questions(args.data_path)
    # Row i of the Concept KC is question i, and that is what makes the
    # embeddings joinable to it; a different bank would silently mislabel.
    if len(questions) != len(concept_df):
        raise SystemExit(f"{args.data_path} has {len(questions)} questions, but the Concept KC has "
                         f"{len(concept_df)} rows — wrong question file for this result dir")

    out_embed = prepare_output_dir(embed_dir(result_dir))

    if args.sent_path:
        model = SentenceTransformer(args.sent_path, local_files_only=True)
        with torch.inference_mode():
            q_embeds = _as_numpy(model.encode([str(q) for q in questions]))
            c_embeds = _as_numpy(model.encode(concept_df["KC"].tolist()))
        _save_and_cluster("sbert", q_embeds, ds, concept_df, questions, out_embed_kc, out_embed)
        _save_and_cluster("concept", c_embeds, ds, concept_df, questions, out_concept_kc, out_embed)

    if args.llm_path:
        llm = LargeLangModel(args.llm_path, trust_remote_code=True, dtype=torch.float16)
        q_embeds = _as_numpy(extract_question_embeds(llm, questions, args.batch_size))
        _save_and_cluster("llm", q_embeds, ds, concept_df, questions, out_embed_kc, out_embed)

    save_args(result_dir, "embed", ds, vars(args))


def add_arguments(parser):
    parser.add_argument("--result_dir", default=None, type=str,
                        help="The result directory holding the Concept KC (default: --run_dir)")
    parser.add_argument("--run_dir", default=None, type=str,
                        help="Result folder shared by every step of this run (env: KCLUSTER_RUN_DIR)")
    parser.add_argument("--sent_path", default=None, type=str,
                        help="Path to a SentenceTransformer; builds the sbert-cosine and "
                             "concept-cosine models (CPU is fine)")
    parser.add_argument("--llm_path", default=None, type=str,
                        help="Path to a downloaded LLM; builds the llm-cosine model (needs a GPU)")
    parser.add_argument("--data_path", default=None, type=str,
                        help="Question file for this result dir (default: the path recorded in "
                             "args-*.json, which may not exist on this machine)")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
