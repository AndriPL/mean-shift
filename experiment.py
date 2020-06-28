import os
from datetime import datetime

import numpy as np
from scipy.stats import ttest_ind
from sklearn.base import clone
from sklearn.metrics import (adjusted_rand_score,
                             homogeneity_completeness_v_measure)
from tabulate import tabulate

from experiment_config import clrs, datasets, datasets_path, measures

time = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
if not os.path.exists(f"./results/{time}/"):
    os.makedirs(f"./results/{time}/")

scores = {}
for measure in measures:
    scores.setdefault(measure, np.zeros((len(clrs), len(datasets))))

for data_id, dataset_name in enumerate(datasets):
    dataset = np.genfromtxt(f"{datasets_path}{dataset_name}.csv", delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    # Experiments
    for clr_id, clr_name in enumerate(clrs):
        clr = clone(clrs[clr_name])
        clr.fit_predict(X)

        # labels_unique = np.unique(clr.labels_)
        # n_clusters_ = len(labels_unique)
        # print(f"{clr_name}. Estimated clusters: %d" % n_clusters_)

        for measure_name in scores:
            scores[measure_name][clr_id, data_id] = measures[measure_name](y, clr.labels_)

np.savez(f"./results/scores", **scores)
