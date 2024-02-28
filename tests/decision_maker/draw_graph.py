import matplotlib.pyplot as plt
import pandas as pd
# Sample data
# y = [0.0109, 0.3787, 0.7894, 12.4716, 90.7383, 194.47]
df = pd.read_csv('results/evaluation_17.csv')
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x = sorted(df['group'].unique())

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

# objective
plt.plot(x, greedy_acc_objective, label='greedy_acc', marker='o', linestyle='-')
plt.plot(x, greedy_delay_objective, label='greedy_delay', marker='o', linestyle='-')
plt.plot(x, greedy_multi_objective, label='greedy_multi', marker='o', linestyle='-')
plt.plot(x, local_new_objective, label='LocalSearch', marker='o', linestyle='-')
plt.plot(x, ILS_objective, label='ILS', marker='o', linestyle='-')

# Add labels and title
plt.xlabel('Number of requests')
plt.ylabel('Objective')

# time
# plt.plot(x, greedy_acc_time, label='greedy_acc', marker='o', linestyle='-')
# plt.plot(x, greedy_delay_time, label='greedy_delay', marker='o', linestyle='-')
# plt.plot(x, greedy_multi_time, label='greedy_multi', marker='o', linestyle='-')
# plt.plot(x, local_new_time, label='LocalSearch_new', marker='o', linestyle='-')
# plt.plot(x, ILS_time, label='ILS', marker='o', linestyle='-')
#
# # Add labels and title
# plt.xlabel('Number of requests')
# plt.ylabel('Decision Making Time(s)')

# Add a legend
plt.legend()

# Display the chart
plt.show()