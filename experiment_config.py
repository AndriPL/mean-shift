from sklearn.cluster import DBSCAN, OPTICS, KMeans, MeanShift
from sklearn.metrics import (adjusted_rand_score, completeness_score,
                             homogeneity_score, v_measure_score)

# Parameters
n_samples = 500
n_clusters = 5
data_random_state = 123

km_random_state = 957
dbs_eps = 0.6
min_samples = 3
# bandwidth
# min_samples
# eps =

# Generating or reading input data
datasets_path = "./datasets/"
datasets = [
    "iris",
    "wine",
    "sonar",
    "popfailures",
    "german",
    "diabetes",
    "banknote",
    "soybean",
    "wisconsin",
    "spambase",
]
# dataset = np.genfromtxt("./dataset/australian.csv", delimiter=",")
# X = dataset[:, :-1]

# X, y = make_blobs(
#     n_samples=n_samples,
#     n_features=2,
#     centers=n_clusters,
#     random_state=data_random_state,
# )


# Algorithms
# clrs = {
#     "k-Means": KMeans(n_clusters=n_clusters, random_state=km_random_state),
#     "MeanShift": MeanShift(),  # if not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth
#     "DBScan": DBSCAN(eps=dbs_eps, min_samples=min_samples),
#     "OPTICS": OPTICS(
#         min_samples=min_samples, min_cluster_size=1 / (4 * n_clusters), xi=0.12
#     ),
# }

clrs = {
    "k-Means": KMeans(random_state=km_random_state),
    "MeanShift": MeanShift(),  # if not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth
    "DBScan": DBSCAN(),
    "OPTICS": OPTICS()
}

measures = {
    "adjusted_rand_score": adjusted_rand_score,
    "completeness_score": completeness_score,
    "homogeneity_score": homogeneity_score,
    "v_measure_score": v_measure_score
}
