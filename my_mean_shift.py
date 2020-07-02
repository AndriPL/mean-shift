import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import DistanceMetric
from sklearn.utils.validation import check_array, check_is_fitted
from kernels import gaussian_kernel, flat_kernel
from sklearn.cluster._mean_shift import estimate_bandwidth


def find_neighbours(bandwidth, center, dist_metric, points):
    # for each point calculate distances from center
    distances = dist_metric.pairwise(points, np.array(center, ndmin=2))
    # find points within bandwidth from center
    return [points[i] for i, distance in enumerate(distances) if distance <= bandwidth]


def shift_center(bandwidth, center, dist_metric, kernel, neighbours):
    numerator = 0
    denominator = 0
    for neighbour in neighbours:
        distance = dist_metric.pairwise([center], [neighbour])
        weight = kernel(distance, bandwidth)
        numerator += weight * neighbour
        denominator += weight
    new_center = numerator / denominator
    return new_center


class MyMeanShift(BaseEstimator, ClusterMixin):
    """
    Mean Shift clustering.
    Algorytm segmentacji Mean Shift.
    """

    def __init__(self, bandwidth=None, dist_metric="euclidean", max_n_iter = 10):
        self.bandwidth = bandwidth
        self.dist_metric = DistanceMetric.get_metric(dist_metric)
        self.max_n_iter = max_n_iter

    def fit(self, X, y=None):
        # data validation
        X = check_array(X)
        self.X_ = X
        # estimate bandwidth
        if self.bandwidth is None:
            self.bandwidth = estimate_bandwidth(X, n_jobs=1) # TODO
        # perform data clustering
        cluster_centers = np.copy(X)
        for n in range(self.max_n_iter):
            for i, center in enumerate(cluster_centers):
                neighbours = find_neighbours(
                    center=center,
                    bandwidth=self.bandwidth,
                    dist_metric=self.dist_metric,
                    points=X
                )
                cluster_centers[i] = shift_center(
                    bandwidth=self.bandwidth,
                    center=center,
                    dist_metric=self.dist_metric,
                    kernel=flat_kernel,
                    neighbours=neighbours
                )
            cluster_centers = np.unique(cluster_centers, axis=0)
        self.cluster_centers_ = cluster_centers

        self.labels_ = self.predict(X)

        return self

    def predict(self, X, y=None):
        # check if fitting has been performed
        check_is_fitted(self)
        # data validation
        X = check_array(X)
        # for every point calculate distances to centroids
        distances = self.dist_metric.pairwise(self.cluster_centers_, X)
        # for every point find nearest centroid
        y_pred = np.argmin(distances, axis=0)
        return y_pred

        # return pairwise_distances_argmin(X, self.cluster_centers_)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_
