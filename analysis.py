from datetime import datetime

import numpy as np
from scipy.stats import rankdata, ttest_ind
from tabulate import tabulate

from experiment_config import clrs

scores = np.load(f"./results/scores.npz")

# for measure in scores:
#     print(f"\n{measure}s:\n", scores[measure].shape)

mean_scores = {}
for measure in scores:
    # mean_scores = np.mean(scores[measure], axis=2).T
    mean_scores.setdefault(measure, scores[measure].T)
    # print(f"\n{measure} mean_scores:\n", mean_scores[measure])


ranks = {}
for measure in mean_scores:
    measure_ranks = []
    for score in mean_scores[measure]:
        measure_ranks.append(rankdata(score).tolist())
    ranks.setdefault(measure, np.array(measure_ranks))
    print(f"\n{measure} ranks:\n", ranks[measure])


# # Statistical analysis
# alfa = .05
# t_statistic = np.zeros((len(clrs.keys()), len(clrs.keys())))
# p_value = np.zeros((len(clrs.keys()), len(clrs.keys())))

# mean_scores = 


# # Results formating
# headers = clrs.keys()
# print(f"Headers: {headers}")
# names_column = np.array([[clr_name] for clr_name in clrs.keys()])
# print(f"Names column: {names_column}")
# t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
# t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
# p_value_table = np.concatenate((names_column, p_value), axis=1)
# p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

# advantage = np.zeros((len(clrs.keys()), len(clrs.keys())))
# advantage[t_statistic > 0] = 1
# advantage_table = tabulate(np.concatenate(
#     (names_column, advantage), axis=1), headers)
# print("Advantage:\n", advantage_table)

# significance = np.zeros((len(clrs.keys()), len(clrs.keys())))
# significance[p_value <= alfa] = 1
# significance_table = tabulate(np.concatenate(
#     (names_column, significance), axis=1), headers)
# print("Statistical significance (alpha = 0.05):\n", significance_table)

# stat_better = significance * advantage
# stat_better_table = tabulate(np.concatenate(
#     (names_column, stat_better), axis=1), headers)
# print("Statistically significantly better:\n", stat_better_table)
