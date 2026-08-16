"""Reading [LearnSphere](https://github.com/LearnSphere/WorkflowComponents) results.

A DataShop workflow that fits AFM writes its per-model statistics as
``Analysis-*_model_values.xml``: one ``<model>`` block per KC model per
cross-validation run, holding the fit statistics and the blocked-CV scores. The
workflow duplicates each KC model once per run, so a ten-run comparison carries
ten blocks per model, and the numbers a paper reports are their mean.

This is a reader for another tool's artifacts, like :mod:`kcluster.io.datashop`
— which is also where the ``KC (<name>)`` naming convention it shares comes
from. It stays because those archived runs are the published baselines; fitting
locally (`kcluster fit`) does not make them unreadable, and the two want to be
compared. To that end both should speak one column vocabulary — ``kc_model``,
``n_params``, ``aic``, ``bic``, ``cv_rmse_<scheme>`` — so analysis code never
branches on which tool produced a table.
"""

import re
from collections import defaultdict

import numpy as np
import pandas as pd

from kcluster.io.datashop import KC_PAT


def read_cv_results(res_path: str, num_cv_runs: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a workflow's ``model_values.xml`` into a machine- and a paper-table.

    Returns ``(res_table, pub_table)``: the first keyed by metric with one row
    per run — fit statistics, which do not vary across the duplicated runs,
    collapsed to their mean — and the second holding ``mean (sd)`` strings.

    Raises if any model does not carry exactly ``num_cv_runs`` blocks, since a
    short count means the workflow's runs and the table's rows disagree about
    what is being averaged.
    """
    # bs4 ships with the optional [datashop] extra; import lazily so importing
    # this module does not require it
    from bs4 import BeautifulSoup

    with open(res_path, "r") as f:
        soup = BeautifulSoup(f, features="html.parser")

    # Extract results
    results = defaultdict(lambda: defaultdict(list))
    for model in soup.find_all("model"):
        name = model.find("name").string
        if match := re.match(KC_PAT, name):
            name = match.group("name")
        else:
            raise ValueError(f"Unrecognized name: {name}")

        for tag in model.find("name").next_siblings:
            if tag.name:
                val = float(tag.string)
                results[tag.name][name].append(val)

    # Verify there is a correct number of results
    for metric in results:
        for model in results[metric]:
            num_results = len(results[metric][model])
            assert num_results == num_cv_runs, f"Expected {num_cv_runs} results, got {num_results} for '{model}'"

    # Build a machine-readable result table for further processing
    res_table = dict()
    for metric in results:
        res_table[metric] = pd.DataFrame(results[metric])
        if metric in {"aic", "bic", "log_likelihood"}:
            res_table[metric] = res_table[metric].mean(axis=0).to_frame().T
    res_table = pd.concat(res_table)

    # Build a human-readable result table for use in a paper
    pub_table = defaultdict(lambda: dict())
    for metric in results:
        for model in results[metric]:
            mean, std = np.mean(results[metric][model]), np.std(results[metric][model])
            pub_table[metric][model] = f"{mean:.4f} ({std:.4f})"
    pub_table = pd.DataFrame.from_dict(pub_table, orient="columns")

    return res_table, pub_table
