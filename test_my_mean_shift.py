from sklearn.cluster import MeanShift


# Wczytaj dane
dataset = np.genfromtxt(f"{datasets_path}{dataset_name}.csv", delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

clr = MeanShift()
clr.fit(X)
clusters = clf.cluster_centers_

my_ms = MyMeanShift()
my_ms.clusters = clusters

results = my_ms.predict(X)

print(results)