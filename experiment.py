import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS, KMeans, MeanShift
from sklearn.datasets import make_blobs
from tabulate import tabulate

logger = logging.getLogger("MSI")

# dataset = np.genfromtxt("./dataset/australian.csv", delimiter=",")
# X = dataset[:, :-1]
X, y = make_blobs(n_samples=100, n_features=2, random_state=123)
logger.log(logging.DEBUG, msg="Data generated")

clrs = {
    "k-Means": KMeans(n_clusters=3, random_state=957),
    "MeanShift": MeanShift(),  # if not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth
    "DBScan": DBSCAN(),
    "OPTICS": OPTICS()
}
print("Clustering methods created")

labels = clrs["MeanShift"].fit_predict(X)
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("Number of estimated clusters : %d" % n_clusters_)
logger.log(logging.DEBUG, msg="MeanShift finished")

plt.figure(figsize=(5, 2.5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="bwr")
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

# headers = ["GNB", "kNN", "CART"]
# names_column = np.array([["GNB"], ["kNN"], ["CART"]])
# t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
# t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
# p_value_table = np.concatenate((names_column, p_value), axis=1)
# p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
