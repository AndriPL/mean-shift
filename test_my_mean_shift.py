import numpy as np
from sklearn.cluster import MeanShift

from my_mean_shift import MyMeanShift, find_neighbours

# datasets_path = "./datasets/"
# dataset_name = "iris"
# # Wczytaj dane
# dataset = np.genfromtxt(f"{datasets_path}{dataset_name}.csv", delimiter=",")
# X = dataset[:, :-1]
# y = dataset[:, -1].astype(int)

# clr = MeanShift()
# clr.fit(X)
X = np.array(
    [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [8, 2], [10, 2], [9, 3],]
)

my_ms = MyMeanShift()
results = my_ms.fit(X)
neighbours = my_ms.find_neighbours(center=[[2, 2]])
print(f"Neighbours: \n{neighbours}")

# print(y)
# print(results)
# print(clr.labels_)

# print(my_ms.cluster_centers_)
# print(clr.cluster_centers_)
