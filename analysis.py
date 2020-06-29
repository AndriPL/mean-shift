from datetime import datetime

import numpy as np
from scipy.stats import rankdata, ranksums, ttest_ind
from tabulate import tabulate

from experiment_config import clrs, datasets

# Load scores
scores = np.load(f"./results/scores.npz")


# Calculate mean_scores
mean_scores = {}
for measure in scores:
    mean_scores.setdefault(measure, np.mean(scores[measure], axis=2).T)
    # mean_scores.setdefault(measure, scores[measure].T)
    mean = np.copy(mean_scores[measure])
    headers = list(clrs.keys())
    names_column = np.expand_dims(np.array(datasets), axis=1)
    mean = np.concatenate((names_column, mean), axis=1)
    mean = tabulate(mean, headers, floatfmt=".3f")
    print(f"\n{measure} mean_scores:\n", mean)

# Calculate ranks
ranks = {}
for measure in mean_scores:
    measure_ranks = []
    for score in mean_scores[measure]:
        measure_ranks.append(rankdata(score).tolist())
    ranks.setdefault(measure, np.array(measure_ranks))
    # print(f"\n{measure} ranks:\n", ranks[measure])

# Calculate mean_ranks

for measure in ranks:
    mean_ranks = np.mean(ranks[measure], axis=0)
    # print(f"\n{measure} mean_ranks:\n", mean_ranks)


# Statistical analysis
alfa = .05

for measure in ranks:

    # Calculate Wilcoxon statistic and p-value
    w_statistic = np.zeros((len(clrs), len(clrs)))
    p_value = np.zeros((len(clrs), len(clrs)))
    for i in range(len(clrs)):
        for j in range(len(clrs)):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks[measure].T[i], ranks[measure].T[j])
    
    headers = list(clrs.keys())
    names_column = np.expand_dims(np.array(list(clrs.keys())), axis=1)
    w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print(f"\n------------------------{measure}----------------------------")
    print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

    # Calculate advantage matrix
    advantage = np.zeros((len(clrs), len(clrs)))
    advantage[w_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("\nAdvantage:\n", advantage_table)

    # Calculate significance
    significance = np.zeros((len(clrs), len(clrs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("\nStatistical significance (alpha = 0.05):\n", significance_table)

    # Which algoritm is better
    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("\nStatistically significantly better:\n", stat_better_table)
