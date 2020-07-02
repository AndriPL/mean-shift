import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster._mean_shift import estimate_bandwidth
from sklearn.neighbors import BallTree, DistanceMetric
from sklearn.utils.validation import check_array, check_is_fitted

from kernels import flat_kernel


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

        self.tree_ = BallTree(X, leaf_size=X.size)

        # perform data clustering
        cluster_centers = np.copy(X)
        for n in range(self.max_n_iter):
            for i, center in enumerate(cluster_centers):
                neighbours_idxs = self.find_neighbours(center=np.array(center, ndmin=2))
                cluster_centers[i] = self.shift_center(
                    center=center,
                    kernel=flat_kernel,
                    neighbours=np.array(X[neighbours_idxs]),
                    X=X
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


    def find_neighbours(self, center):
        # for each point calculate distances from center
        neighbours_idxs = self.tree_.query_radius(center, self.bandwidth)
        return neighbours_idxs[0]

    def shift_center(self, center, kernel, neighbours, X):
        distances = self.dist_metric.pairwise([center], neighbours)
        weights = kernel(distances, self.bandwidth)
        numerator = np.multiply(
            np.tile(weights.T, neighbours.shape[1]), 
            neighbours
        ).sum(axis=0)
        denominator = np.sum(weights)
        new_center = np.divide(numerator, denominator)
        return new_center