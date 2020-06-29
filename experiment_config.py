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

# Datasets
datasets_path = "./datasets/"
datasets = [
    "iris",
    "banknote",
    "diabetes",
    "wisconsin",
    "wine",
    "popfailures",
    "german",
    "soybean",
    "spambase",
    "sonar",
]
# datasets = [
#     'australian',
#     'balance',
#     'cryotherapy',
#     'diabetes',
#     'digit',
#     'german',
#     'heart',
#     'liver',
#     'soybean',
#     'waveform'
# ]



# Algorithms
clrs = {
    "k-Means": KMeans(random_state=km_random_state),
    "MeanShift": MeanShift(),  # if not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth
    "DBScan": DBSCAN(),
    "OPTICS": OPTICS()
}

# fit_predict method for each algorithm - because DBScan and OPTICS doesn't have predict() method
fit_predict = {
    "k-Means": lambda clr, X_train, X_test: clr.fit(X_train).predict(X_test),
    "MeanShift": lambda clr, X_train, X_test: clr.fit(X_train).predict(X_test),
    "DBScan": lambda clr, _, X_test: clr.fit_predict(X_test),
    "OPTICS": lambda clr, _, X_test: clr.fit_predict(X_test)
}

# Measures
measures = {
    "adjusted_rand_score": adjusted_rand_score,
    "completeness_score": completeness_score,
    "homogeneity_score": homogeneity_score,
    "v_measure_score": v_measure_score
}
