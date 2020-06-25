import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS, KMeans, MeanShift
from sklearn.datasets import make_blobs
from tabulate import tabulate

logger = logging.getLogger("MSI")

# Generating or reading input data
n_clusters = 5

# dataset = np.genfromtxt("./dataset/australian.csv", delimiter=",")
# X = dataset[:, :-1]

X, y = make_blobs(n_samples=500, n_features=2, centers=n_clusters, random_state=123)


# Algorithms
clrs = {
    "k-Means": KMeans(n_clusters=n_clusters, random_state=957),
    "MeanShift": MeanShift(),  # if not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth
    "DBScan": DBSCAN(),
    "OPTICS": OPTICS()
}


# MeanShift
for idx, clr_name in enumerate(clrs):
    clr = clrs[clr_name].fit(X)
    labels_unique = np.unique(clr.labels_)
    n_clusters_ = len(labels_unique)
    print(f"{clr_name}. Estimated clusters: %d" % n_clusters_)

    # 2D figure
    plt.subplot(2, 2, idx + 1)
    plt.scatter(X[:, 0], X[:, 1], c=clr.labels_)
    plt.xlabel("$x^1$")
    plt.ylabel("$x^2$")
    plt.tight_layout()
    

plt.savefig(f"./plots/{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}.png")
logger.log(logging.DEBUG, msg="Plot saved")

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
