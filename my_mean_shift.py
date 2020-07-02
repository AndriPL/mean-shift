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

    def __init__(self, bandwidth=None, dist_metric="euclidean", max_n_iter = 20):
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
        # Create BallTree for faster detection of points within bandwidth
        self.tree_ = BallTree(X, leaf_size=X.size)

        # perform data clustering
        centroids = np.copy(X)
        prev_centroids = []
        for n in range(self.max_n_iter):
            for i, centroid in enumerate(centroids):
                # Find points within bandwidth
                neighbours_idxs = self._find_neighbours(center=np.array(centroid, ndmin=2))
                # Shift centroid
                centroids[i] = self._shift_center(
                    center=centroid,
                    kernel=flat_kernel,
                    neighbours=np.array(X[neighbours_idxs])
                )

            centroids = np.unique(centroids, axis=0) # TODO maybe move it below convergence detection
            
            # If centroids didn't changed, then finish
            optimized = False
            if np.array_equal(centroids, prev_centroids):
                optimized = True  
            if optimized:
                break
            prev_centroids = centroids

        self.centroids_ = centroids
        self.labels_ = self.predict(X)

        return self

    def predict(self, X, y=None):
        # check if fitting has been performed
        check_is_fitted(self)
        # data validation
        X = check_array(X)
        # for every point calculate distances to centroids
        distances = self.dist_metric.pairwise(self.centroids_, X)
        # for every point find nearest centroid
        y_pred = np.argmin(distances, axis=0)
        return y_pred


    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_


    def _find_neighbours(self, center):
        neighbours_idxs = self.tree_.query_radius(center, self.bandwidth)
        return neighbours_idxs[0]

    def _shift_center(self, center, kernel, neighbours):
        distances = self.dist_metric.pairwise([center], neighbours)
        weights = kernel(distances, self.bandwidth)
        numerator = np.multiply(
            np.tile(weights.T, neighbours.shape[1]), 
            neighbours
        ).sum(axis=0)
        denominator = np.sum(weights)
        new_center = np.divide(numerator, denominator)
        return new_center