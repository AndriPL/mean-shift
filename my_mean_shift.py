import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.neighbors import DistanceMetric

class MyMeanShift(BaseEstimator, ClassifierMixin):
    """
    Mean Shift clustering.
    Algorytm segmentacji Mean Shift.
    """
    def __init__(self, bandwidth=None, metric='euclidean'): # metric, kernel?
        self.bandwidth = bandwidth
        self.metric = metric
        self.dm_ = DistanceMetric.get_metric(self.metric)
 

    def fit(self, X, y=None):
        # # data validation
        # X = check_array(X)
        self.centroids_ = []
        self.dm_ = DistanceMetric.get_metric(self.metric)
        # self.X_ = X

        # # perform data clustering
        
        
        return self

    
    def predict(self, X, y=None):
        
        # check if fitting has been performed 
        check_is_fitted(self)
        # data validation
        X = check_array(X)
        # liczymy dystanse instancji testowych od centroidow
   

        distance_pred = self.dm_.pairwise(self.centroids_, X)

        y_pred = np.argmin(distance_pred, axis=0)
        return y_pred




    def fit_predict(self, X, y=None):
        pass
        # self.fit(X)
        # return self.labels_


