import argparse
import os

import torch
from sklearn.exceptions import ConvergenceWarning
from transformers.utils import logging

from core.kcluster import KCluster
from core.model import LargeLangModel


def main(args):
    import warnings
    warnings.filterwarnings("error", category=ConvergenceWarning)

    # Load an LLM
    llm = LargeLangModel(args.llm_path, trust_remote_code=True,
                         torch_dtype=torch.float16, attn_implementation="flash_attention_2")

    for embed_type in ("concept", "question"):
        for metric in ("cosine", "euclidean"):
            kcluster = KCluster(args.sim_dir, metric=metric, embed_type=embed_type)
            # Search for the minimal damping factor leading to convergence
            damping = 0.5
            while damping < 1.0:
                try:
                    kc = kcluster.create_new_kc(damping=damping)
                except ConvergenceWarning:
                    print(f"Did not converge when damping = {damping}")
                    damping += 0.05
                else:
                    kc = kcluster.populate_concepts(llm, kc, args.batch_size, args.num_beams, args.length_penalty)
                    kc.to_csv(os.path.join(args.sim_dir, f"{embed_type}-{metric}-kc.csv"), sep="\t", index=False)
                    break


if __name__ == "__main__":
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--sim_dir", required=True, type=str, help="Path to a directory containing similarity")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams employed in beam search")
    parser.add_argument("--length_penalty", type=float, default=-0.1, help="Length penalty for beam search")

    cl_args = parser.parse_args()
    main(cl_args)
