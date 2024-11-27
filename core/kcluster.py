import glob
import json
import os
from functools import cached_property
from operator import itemgetter

import pandas as pd
import torch
from sklearn.cluster import affinity_propagation

from core.model import LargeLangModel
from core.question import Question
from experiments.run_concept import extract_concepts


class PointwiseMutualInfo:
    def __init__(self, pmi_dir: str, num_questions: int, normalize: bool = True):
        self._vec, self._mat = self.load_probs(pmi_dir, num_questions)
        self._normalize = normalize

    @cached_property
    def marginals(self) -> torch.Tensor:
        # normalize the marginals
        return torch.log_softmax(self._vec, dim=-1) if self._normalize else self._vec

    @cached_property
    def conditionals(self) -> torch.Tensor:
        # use PMI to re-calculate conditionals
        return self.pmi_mat + self.marginals.unsqueeze(-1)

    @cached_property
    def pmi_mat(self) -> torch.Tensor:
        mat = torch.log_softmax(self._mat, dim=0) if self._normalize else self._mat
        mat = mat - self.marginals.unsqueeze(-1)
        mat = (mat + mat.T) / 2  # make symmetric
        return mat

    @staticmethod
    def load_probs(pmi_dir: str, num_questions: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load conditional and marginal log-probabilities from a result folder
        :param pmi_dir: A path to saved log-probability data produced by 'run_pmi.py'
        :param num_questions: The number of questions
        :return: A Tuple consisting of
         - a `num_qs` vector of marginal log-probabilities
         - a `num_qs x num_qs` conditional-log-probability matrix
        """
        # Prepare an empty matrix of the appropriate size to fill in
        mtx = torch.full((num_questions ** 2 + num_questions,), torch.inf)

        # Load data from all available files
        rank = 0
        while os.path.exists(fname := os.path.join(pmi_dir, f"batch_indices_{rank}.pt")):
            batch_inds = torch.load(fname)[0]
            predictions = torch.load(os.path.join(pmi_dir, f"predictions_{rank}.pt"))
            for inds, preds in zip(batch_inds, predictions, strict=True):
                mtx[inds] = preds
            rank += 1
        assert torch.isinf(mtx).sum() == 0, "Number of questions doesn't match the save value"

        mtx = mtx.reshape(-1, num_questions)
        marginals = mtx[0]  # first row is the marginals
        conds = mtx[1:]  # second row and below is the conditionals
        return marginals, conds


class KCluster:
    def __init__(self, pmi_dir: str, normalize_pmi: bool = True):
        self.questions = self.load_questions(pmi_dir)
        self.pmi = PointwiseMutualInfo(pmi_dir, len(self.questions), normalize_pmi)

    def create_new_kc(self, use_p: str = "median", **ap_kwargs) -> pd.DataFrame:
        """
        Create a new KC model from a similarity matrix
        :param use_p: Determine how to compute preference
        :param ap_kwargs: Additional arguments for the Affinity Propagation algorithm, e.g., damping=0.7
        :return: The new KC model in a DataFrame
        """
        assert use_p in ("median", "mean", "min", "max"), f"Invalid value for 'use_p': {use_p}"

        S = self.pmi.pmi_mat
        # Determine p
        func = {"median": torch.median, "mean": torch.mean, "min": torch.amin, "max": torch.amax}
        p = func[use_p](S[~torch.eye(len(self.questions), dtype=torch.bool)])

        # Run AP
        centers, labels, num_iters = affinity_propagation(S, preference=p,
                                                          random_state=42, return_n_iter=True, **ap_kwargs)
        print(f"Affinity Propagation completed in {num_iters} iterations")

        # Collect clustering results
        res_dicts = []
        for q, label in zip(self.questions, labels):
            q_dict = q.flat_dict
            q_dict.pop("images", None)
            q_dict["KC"] = f"KC-{centers[label]}"
            res_dicts.append(q_dict)

        return pd.DataFrame.from_records(res_dicts)

    def populate_concepts(self, llm: LargeLangModel, kc: pd.DataFrame,
                          batch_size: int = 16, num_beams: int = 5, length_penalty: float = -0.1,
                          pad_to_multiple_of: int = 8, **kwargs) -> pd.DataFrame:
        """
        Use LLM to generate concept labels for an existing KC model
        :param llm: An instance of LargeLangModel
        :param kc: The existing KC model in a pd.DataFrame
        :param batch_size: Number of questions to process in a batch
        :param num_beams: Number of beams to use
        :param length_penalty: Penalty applied to encourage shorter concept labels
        :param pad_to_multiple_of: Padding to utilize tensor cores efficiently
        :param kwargs: Any additional keyword args
        :return: An updated KC model with concept labels
        """
        kc["exemplar"] = kc["KC"].str.split("-").apply(itemgetter(1)).apply(int)
        exemplars = kc["exemplar"].unique().tolist()
        questions = [self.questions[idx] for idx in exemplars]

        concepts = extract_concepts(llm, questions, batch_size=batch_size,
                                    num_beams=num_beams, length_penalty=length_penalty,
                                    pad_to_multiple_of=pad_to_multiple_of, **kwargs)

        ids_to_concepts = dict(zip(exemplars, concepts))
        kc["KC-raw"] = kc["KC"]  # retain the raw KC labels
        kc["KC"] = kc["exemplar"].apply(ids_to_concepts.get)  # populate "KC" column with concepts
        kc.drop(columns=["exemplar"], inplace=True)
        return kc

    @staticmethod
    def load_questions(pmi_dir: str) -> list[Question]:
        [fname] = glob.glob("args*.json", root_dir=pmi_dir)
        with open(os.path.join(pmi_dir, fname), "r") as f:
            data_path = json.load(f)["data_path"]
        with open(data_path, "r") as f:
            questions = [Question(eval(line)) for line in f]
        return questions
