import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.cluster import DBSCAN, OPTICS, KMeans, MeanShift
from sklearn.datasets import make_blobs, make_circles, make_moons, make_s_curve
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_completeness_v_measure,
    homogeneity_score,
)
from tabulate import tabulate

logger = logging.getLogger("MSI")

# Parameters
n_samples = 500
n_clusters = 5
data_random_state = 123

km_random_state = 957
dbs_eps = 0.6
min_samples = 3
# bandwidth
# min_samples
# eps =

# Generating or reading input data
datasets_path = "./datasets/"
datasets = [
    "iris",
    "wine",
    "sonar",
    "popfailures",
    "german",
    "diabetes",
    "banknote",
    "soybean",
    "wisconsin",
    "spambase",
]
# dataset = np.genfromtxt("./dataset/australian.csv", delimiter=",")
# X = dataset[:, :-1]

# X, y = make_blobs(
#     n_samples=n_samples,
#     n_features=2,
#     centers=n_clusters,
#     random_state=data_random_state,
# )


# Algorithms
# clrs = {
#     "k-Means": KMeans(n_clusters=n_clusters, random_state=km_random_state),
#     "MeanShift": MeanShift(),  # if not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth
#     "DBScan": DBSCAN(eps=dbs_eps, min_samples=min_samples),
#     "OPTICS": OPTICS(
#         min_samples=min_samples, min_cluster_size=1 / (4 * n_clusters), xi=0.12
#     ),
# }

clrs = {
    "k-Means": KMeans(random_state=km_random_state),
    "MeanShift": MeanShift(),  # if not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth
    "DBScan": DBSCAN(),
    "OPTICS": OPTICS()
}

time = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
if not os.path.exists(f"./results/{time}/"):
    os.makedirs(f"./results/{time}/")

for dataset_file in datasets:
    dataset = np.genfromtxt(f"{datasets_path}{dataset_file}.csv", delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    scores = {}

    # Experiments
    for idx, clr_name in enumerate(clrs):
        clr = clone(clrs[clr_name])
        clr.fit_predict(X)

        labels_unique = np.unique(clr.labels_)
        n_clusters_ = len(labels_unique)
        print(f"{clr_name}. Estimated clusters: %d" % n_clusters_)

        clr_scores = {}

        clr_scores.setdefault(
            "adjusted_rand_score", adjusted_rand_score(y, clr.labels_)
        )
        clr_scores.setdefault(
            "completeness_score", completeness_score(y, clr.labels_)
        )
        clr_scores.setdefault(
            "homogeneity_score", homogeneity_score(y, clr.labels_)
        )
        clr_scores.setdefault(
            "homogeneity_completeness_v_measure",
            homogeneity_completeness_v_measure(y, clr.labels_),
        )

        scores.setdefault(clr_name, clr_scores)

    np.save(f"./results/{time}/{dataset_file}", scores)

    #     # 2D figure
    #     plt.subplot(2, 2, idx + 1)
    #     plt.scatter(X[:, 0], X[:, 1], c=clr.labels_)
    #     plt.xlabel("$x^1$")
    #     plt.ylabel("$x^2$")
    # plt.tight_layout()
    # plt.savefig(f"./plots/{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}.png")

# # 3D figure
# fig = plt.figure(figsize=(12, 7), dpi=80, facecolor="w", edgecolor="k")
# ax = plt.axes(projection="3d")
# ax.scatter3D(pca.T[0], pca.T[1], pca.T[2], c=minikm_labels, cmap="Spectral")
# xLabel = ax.set_xlabel("X")
# yLabel = ax.set_ylabel("Y")
# zLabel = ax.set_zlabel("Z")

# # Results formating
# headers = ["GNB", "kNN", "CART"]
# names_column = np.array([["GNB"], ["kNN"], ["CART"]])
# t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
# t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
# p_value_table = np.concatenate((names_column, p_value), axis=1)
# p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
