import glob
import itertools
import json
import os
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import affinity_propagation
from sklearn.metrics import pairwise_distances

from core.pmi import PointwiseMutualInfo
from core.question import Question


class Embedding:
    def __init__(self, embed_path: str, num_questions: int):
        self.embeds = np.load(embed_path)
        n_embeds = self.embeds.shape[0]
        assert n_embeds == num_questions, f"Expected {num_questions} questions, got {n_embeds} embeddings"

    def sim_mtx(self, metric: str):
        return -pairwise_distances(self.embeds, metric=metric)


class KCluster:
    def __init__(self, sim_dir: str, metric: str = "pmi", embed_type: str = "question", **pmi_kwargs):
        assert metric in ("cosine", "euclidean", "pmi"), f"Unknown similarity type: {metric}"
        assert embed_type in ("question", "concept"), f"Unknown embedding type: {embed_type}"

        # Load questions
        self.questions = self.load_questions(sim_dir)
        num_questions = len(self.questions)

        # Determine the similarity matrix
        if metric == "pmi":
            [fname] = glob.glob(f"args*.json", root_dir=sim_dir)
            with open(os.path.join(sim_dir, fname), "r") as f:
                cond_on_dim = json.load(f).get("cond_on_dim", 0)
            pmi = PointwiseMutualInfo(sim_dir, num_questions, num_questions, cond_on_dim, **pmi_kwargs)
            self.sim_mtx = pmi.pmi_mat.cpu().numpy()
        else:
            [fname] = glob.glob(f"*-{embed_type}-embeds.npy", root_dir=sim_dir)
            embed = Embedding(os.path.join(sim_dir, fname), num_questions)
            self.sim_mtx = embed.sim_mtx(metric)

    @staticmethod
    def load_questions(sim_dir: str) -> list[Question]:
        [fname] = glob.glob("args*.json", root_dir=sim_dir)
        with open(os.path.join(sim_dir, fname), "r") as f:
            data_path = json.load(f)["data_path"]
        with open(data_path, "r") as f:
            questions = [Question(eval(line)) for line in f]
        return questions

    def create_new_kc(self, predicate: Callable[[Question], bool] = None,
                      use_p: str = "median", **ap_kwargs) -> pd.DataFrame:
        """
        Create a new KC model from the similarity matrix
        :param predicate: A function indicating if a question should be considered
        :param use_p: Determine how to compute preference
        :param ap_kwargs: Additional arguments for the Affinity Propagation algorithm, e.g., damping=0.7
        :return: The new KC model in a DataFrame
        """
        assert use_p in ("median", "mean", "min", "max"), f"Invalid value for 'use_p': {use_p}"

        # Determine the questions and similarity matrix
        questions, sim_mtx = self.questions, self.sim_mtx
        if predicate is not None:
            is_valid = [predicate(q) for q in questions]
            questions = list(itertools.compress(questions, is_valid))
            sim_mtx = sim_mtx[np.ix_(is_valid, is_valid)]
        assert sim_mtx.shape == (len(questions), len(questions)), "The shape of the similarity matrix is incorrect"

        # Determine p
        func: dict[str, Callable] = {"median": np.median, "mean": np.mean, "min": np.amin, "max": np.amax}
        p = func[use_p](sim_mtx[~np.eye(len(questions), dtype=bool)])

        # Run AP
        centers, labels, num_iters = affinity_propagation(sim_mtx, preference=p,
                                                          return_n_iter=True, random_state=42, **ap_kwargs)
        print(f"Affinity Propagation completed in {num_iters} iterations and created {len(centers)} clusters")

        # Collect clustering results
        res_dicts = []
        for q, label in zip(questions, labels, strict=True):
            q_dict = q.flat_dict
            q_dict.pop("images", None)
            q_dict["KC"] = f"KC-{centers[label]}"
            res_dicts.append(q_dict)
        return pd.DataFrame.from_records(res_dicts)
