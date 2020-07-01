import numpy as np
from sklearn.cluster import MeanShift

from my_mean_shift import MyMeanShift

datasets_path = "./datasets/"
dataset_name = "iris"
# Wczytaj dane
dataset = np.genfromtxt(f"{datasets_path}{dataset_name}.csv", delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

clr = MeanShift()
clr.fit(X)
clusters = clr.cluster_centers_

my_ms = MyMeanShift(55)
my_ms.centroids_ = clusters
results = my_ms.predict(X)
print(y)
print(results)
print(clr.labels_)
