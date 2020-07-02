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

my_ms = MyMeanShift(bandwidth=3)
results = my_ms.fit_predict(X)
print(y)
print(results)
print(clr.labels_)

print(my_ms.cluster_centers_)
print(clr.cluster_centers_)