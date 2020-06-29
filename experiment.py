import os

import numpy as np
from scipy.stats import ttest_ind
from sklearn.base import clone
from sklearn.metrics import (adjusted_rand_score,
                             homogeneity_completeness_v_measure)
from sklearn.model_selection import RepeatedStratifiedKFold
from tabulate import tabulate

from experiment_config import clrs, datasets, datasets_path, measures, fit_predict

if not os.path.exists(f"./results/"):
    os.makedirs(f"./results/")

n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=5588)

scores = {}
for measure in measures:
    scores.setdefault(measure, np.zeros((len(clrs), len(datasets), n_splits * n_repeats)))


for data_id, dataset_name in enumerate(datasets):
    dataset = np.genfromtxt(f"{datasets_path}{dataset_name}.csv", delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    print(f"{data_id}.{dataset_name}")

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clr_id, clr_name in enumerate(clrs):
            clr = clone(clrs[clr_name])
            pred = fit_predict[clr_name](clr, X[train], X[test])

            # labels_unique = np.unique(clr.labels_)
            # n_clusters_ = len(labels_unique)
            # print(f"{clr_name}. Estimated clusters: %d" % n_clusters_)

            for measure_name in scores:
                scores[measure_name][clr_id, data_id, fold_id] = measures[measure_name](y[test], pred)

np.savez(f"./results/scores", **scores)
