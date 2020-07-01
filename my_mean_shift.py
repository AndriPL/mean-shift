import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import DistanceMetric
from sklearn.utils.validation import check_array, check_is_fitted


def find_neighbours(points, center, bandwidth, metric):
    distance_metric = DistanceMetric.get_metric(metric)
    neigbours = []
    distances = distance_metric.pairwise(points, np.array(center, ndmin=2))

    for i, distance in enumerate(distances):
        if distance <= bandwidth:
            neigbours.append(points[i])

    return neigbours


def flat_kernel(self, X):
    pass


class MyMeanShift(BaseEstimator, ClusterMixin):
    """
    Mean Shift clustering.
    Algorytm segmentacji Mean Shift.
    """

    def __init__(
        self, bandwidth=None, kernel=flat_kernel, metric="euclidean"
    ):  # metric?
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.metric = metric
        self.dm_ = DistanceMetric.get_metric(self.metric)
        self.centroids_ = None

    def fit(self, X, y=None):
        # data validation
        X = check_array(X)
        self.X_ = X

        # if bandwidth is None:
        #     bandwidth = estimate_bandwidth(X, n_jobs=self.n_jobs)

        # perform data clustering
        self.centroids = {}  # or []

        return self

    def predict(self, X, y=None):
        # check if fitting has been performed
        check_is_fitted(self)
        # data validation
        X = check_array(X)
        # for every point calculate distances to centroids
        distances = self.dm_.pairwise(self.centroids_, X)
        # for every point find nearest centroid
        y_pred = np.argmin(distances, axis=0)
        return y_pred

        # return pairwise_distances_argmin(X, self.cluster_centers_)

    def fit_predict(self, X, y=None):
        pass
        # self.fit(X)
        # return self.labels_
