import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Sample data
# y = [0.0109, 0.3787, 0.7894, 12.4716, 90.7383, 194.47]
df = pd.read_csv('results/evaluation_19.csv')
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100]
categories = sorted(df['group'].unique())
x = np.arange(len(categories))

# ortools_time = df[df['algorithm'] == 'ORTools']['time'].tolist()
# ortools_objective = df[df['algorithm'] == 'ORTools']['Normalized objective'].tolist()
# topsis_time = df[df['algorithm'] == 'TOPSIS']['time'].tolist()
# topsis_objective = df[df['algorithm'] == 'TOPSIS']['Normalized objective'].tolist()
# local_time = df[df['algorithm'] == 'LocalSearch']['time'].tolist()
# local_objective = df[df['algorithm'] == 'LocalSearch']['Normalized objective'].tolist()
greedy_acc_time = df[df['algorithm'] == 'Greedy_accfirst']['time'].tolist()
greedy_acc_objective = df[df['algorithm'] == 'Greedy_accfirst']['Normalized objective'].tolist()
greedy_delay_time = df[df['algorithm'] == 'Greedy_delayfirst']['time'].tolist()
greedy_delay_objective = df[df['algorithm'] == 'Greedy_delayfirst']['Normalized objective'].tolist()
greedy_multi_time = df[df['algorithm'] == 'Greedy_multi']['time'].tolist()
greedy_multi_objective = df[df['algorithm'] == 'Greedy_multi']['Normalized objective'].tolist()
local_new_time = df[df['algorithm'] == 'LocalSearch_new']['time'].tolist()
local_new_objective = df[df['algorithm'] == 'LocalSearch_new']['Normalized objective'].tolist()
ILS_time = df[df['algorithm'] == 'ILS']['time'].tolist()
ILS_objective = df[df['algorithm'] == 'ILS']['Normalized objective'].tolist()


bar_width = 0.1

# objective
# plt.bar(x, greedy_acc_objective, width=bar_width, label='greedy_acc')
# plt.bar(x + bar_width, greedy_delay_objective, width=bar_width, label='greedy_delay')
# plt.bar(x + 2*bar_width, greedy_multi_objective, width=bar_width, label='greedy_multi')
# plt.bar(x + 3*bar_width, local_new_objective, width=bar_width, label='LocalSearch')
# plt.bar(x + 4*bar_width, ILS_objective, width=bar_width, label='ILS')
#
#
# # Add labels and title
# plt.xlabel('Number of workflows')
# plt.ylabel('Objective')
# plt.xticks(x + 2*bar_width / 2, categories)

# time
plt.bar(x, greedy_acc_time, width=bar_width, label='greedy_acc')
plt.bar(x + bar_width, greedy_delay_time, width=bar_width, label='greedy_delay')
plt.bar(x + 2*bar_width, greedy_multi_time, width=bar_width, label='greedy_multi')
plt.bar(x + 3*bar_width, local_new_time, width=bar_width, label='LocalSearch')
plt.bar(x + 4*bar_width, ILS_time, width=bar_width, label='ILS')
#
# Add labels and title
plt.xlabel('Number of workflows')
plt.ylabel('Running Time(s)')
plt.xticks(x + 2*bar_width / 2, categories)

# Add a legend
plt.legend()

# Display the chart
plt.show()