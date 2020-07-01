import numpy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_


class MyMeanShift(BaseEstimator, ClusterMixin):
    """
    Mean Shift clustering.
    Algorytm segmentacji Mean Shift.
    """
    def __init__(self, bandwidth=None): # metric, kernel?
        self.bandwidth = bandwidth


    def fit(self, X, y=None):
        # # data validation
        # X = check_array(X)

        # self.X_ = X

        # # perform data clustering
        
        
        return self

    
    def predict(self, X, y=None):
        pass
        # # check if fitting has been performed 
        # check_is_fitted(self)
        # # data validation
        # X = check_array(X)
    
        # return pairwise_distances_argmin(X, self.cluster_centers_)


    def fit_predict(self, X, y=None):
        pass
        # self.fit(X)
        # return self.labels_
