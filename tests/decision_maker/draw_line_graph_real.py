import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Sample data
# y = [0.0109, 0.3787, 0.7894, 12.4716, 90.7383, 194.47]
df = pd.read_csv('results/evaluation_22_real.csv')
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100]
categories = sorted(df['group'].unique())
x = np.arange(len(categories))

# ortools_time = df[df['algorithm'] == 'ORTools']['time'].tolist()
# ortools_objective = df[df['algorithm'] == 'ORTools']['Normalized objective'].tolist()
# topsis_time = df[df['algorithm'] == 'TOPSIS']['time'].tolist()
# topsis_objective = df[df['algorithm'] == 'TOPSIS']['Normalized objective'].tolist()
# local_time = df[df['algorithm'] == 'LocalSearch']['time'].tolist()
# local_objective = df[df['algorithm'] == 'LocalSearch']['Normalized objective'].tolist()

local_new_time = df[df['algorithm'] == 'LocalSearch_new']['time'].tolist()
local_new_objective = df[df['algorithm'] == 'LocalSearch_new']['Normalized objective'].tolist()
local_new_cpu = df[df['algorithm'] == 'LocalSearch_new']['CPU usage'].tolist()
local_new_memory = df[df['algorithm'] == 'LocalSearch_new']['Memory usage'].tolist()
local_new_accuracy = df[df['algorithm'] == 'LocalSearch_new']['Average accuracy'].tolist()
local_new_delay = df[df['algorithm'] == 'LocalSearch_new']['Average delay'].tolist()
local_new_satisfied = df[df['algorithm'] == 'LocalSearch_new']['Satisfied workflows'].tolist()

ILS_time = df[df['algorithm'] == 'ILS']['time'].tolist()
ILS_objective = df[df['algorithm'] == 'ILS']['Normalized objective'].tolist()
ILS_cpu = df[df['algorithm'] == 'ILS']['CPU usage'].tolist()
ILS_memory = df[df['algorithm'] == 'ILS']['Memory usage'].tolist()
ILS_accuracy = df[df['algorithm'] == 'ILS']['Average accuracy'].tolist()
ILS_delay = df[df['algorithm'] == 'ILS']['Average delay'].tolist()
ILS_satisfied = df[df['algorithm'] == 'ILS']['Satisfied workflows'].tolist()

ODP_LS_objective = df[df['algorithm'] == 'ODP-LS']['Normalized objective'].tolist()
ODP_LS_time = df[df['algorithm'] == 'ODP-LS']['time'].tolist()
ODP_LS_cpu = df[df['algorithm'] == 'ODP-LS']['CPU usage'].tolist()
ODP_LS_memory = df[df['algorithm'] == 'ODP-LS']['Memory usage'].tolist()
ODP_LS_accuracy = df[df['algorithm'] == 'ODP-LS']['Average accuracy'].tolist()
ODP_LS_delay = df[df['algorithm'] == 'ODP-LS']['Average delay'].tolist()
ODP_LS_satisfied = df[df['algorithm'] == 'ODP-LS']['Satisfied workflows'].tolist()

ODP_TS_objective = df[df['algorithm'] == 'ODP-TS']['Normalized objective'].tolist()
ODP_TS_time = df[df['algorithm'] == 'ODP-TS']['time'].tolist()
ODP_TS_cpu = df[df['algorithm'] == 'ODP-TS']['CPU usage'].tolist()
ODP_TS_memory = df[df['algorithm'] == 'ODP-TS']['Memory usage'].tolist()
ODP_TS_accuracy = df[df['algorithm'] == 'ODP-TS']['Average accuracy'].tolist()
ODP_TS_delay = df[df['algorithm'] == 'ODP-TS']['Average delay'].tolist()
ODP_TS_satisfied = df[df['algorithm'] == 'ODP-TS']['Satisfied workflows'].tolist()


# bar_width = 0.1

# # objective
# plt.plot(x, local_new_objective, label='MSP-LS', marker='o', linestyle='-')
# plt.plot(x, ILS_objective, label='MSP-ILS', marker='^', linestyle='-')
# plt.plot(x, ODP_LS_objective, label='ODP-LS', marker='o', linestyle='--')
# plt.plot(x, ODP_TS_objective, label='ODP-TS', marker='^', linestyle='--')
# # Add labels and title
# plt.xlabel('Number of Workflows')
# plt.ylabel('Objective')
# plt.legend()
# filename = "results/figures/real/objective.eps"
# plt.show()
# plt.clf()
# time
# plt.bar(x, greedy_acc_time, width=bar_width, label='greedy_acc')
# plt.bar(x + bar_width, greedy_delay_time, width=bar_width, label='greedy_delay')
# plt.bar(x + 2*bar_width, greedy_multi_time, width=bar_width, label='greedy_multi')
# plt.bar(x + 3*bar_width, local_new_time, width=bar_width, label='LocalSearch')
# plt.bar(x + 4*bar_width, ILS_time, width=bar_width, label='ILS')
# #
# # Add labels and title
# plt.xlabel('Number of workflows')
# plt.ylabel('Running Time(s)')
# plt.xticks(x + 2*bar_width / 2, categories)

# # cpu consumption
# plt.plot(x, local_new_cpu, label='MSP-LS', marker='o', linestyle='-')
# plt.plot(x, ILS_cpu, label='MSP-ILS', marker='^', linestyle='-')
# plt.plot(x, ODP_LS_cpu, label='ODP-LS', marker='o', linestyle='--')
# plt.plot(x, ODP_TS_cpu, label='ODP-TS', marker='^', linestyle='--')
# # Add labels and title
# plt.xlabel('Number of Workflows')
# plt.ylabel('CPU Usage(%)')
# filename = "results/figures/real/cpu_usage.eps"
# # Add a legend
# plt.legend()
# plt.show()
# plt.clf()
#
# memory consumption
# plt.plot(x, local_new_memory, marker='o', label='MSP-LS',linestyle='-')
# plt.plot(x, ILS_memory, marker='^', label='MSP-ILS', linestyle='-')
# plt.plot(x, ODP_LS_memory, marker='o', label='ODP-LS', linestyle='--')
# plt.plot(x, ODP_TS_memory, marker='^', label='ODP-TS', linestyle='--')
# # Add labels and title
# plt.xlabel('Number of Workflows')
# plt.ylabel('Memory usage(%)')
# filename = "results/figures/real/memory_usage.eps"
# # Add a legend
# plt.legend()
# plt.show()
# plt.clf()

# accuracy
# plt.plot(x, local_new_accuracy, marker='o', label='MSP-LS',linestyle='-')
# plt.plot(x, ILS_accuracy, marker='^', label='MSP-ILS', linestyle='-')
# plt.plot(x, ODP_LS_accuracy, marker='o', label='ODP-LS', linestyle='--')
# plt.plot(x, ODP_TS_accuracy, marker='^', label='ODP-TS', linestyle='--')
# # Add labels and title
# plt.xlabel('Number of Workflows')
# plt.ylabel('Accuracy(%)')
# filename = "results/figures/real/accuracy.eps"
# # Add a legend
# plt.legend()
# plt.show()
# plt.clf()

# delay
plt.plot(x, ILS_delay, marker='o', label='MSP-LS',linestyle='-')
plt.plot(x, local_new_delay, marker='^', label='MSP-ILS', linestyle='-')
plt.plot(x, ODP_LS_delay, marker='o', label='ODP-LS', linestyle='--')
plt.plot(x, ODP_TS_delay, marker='^', label='ODP-TS', linestyle='--')
# Add labels and title
plt.xlabel('Number of Workflows')
plt.ylabel('Average Delay(s)')
filename = "results/figures/real/delay.eps"
# Add a legend
plt.legend()
plt.show()
plt.clf()

# satisfied workflows
# plt.plot(x, local_new_satisfied, marker='o', label='MSP-LS',linestyle='-')
# plt.plot(x, ILS_satisfied, marker='^', label='MSP-ILS', linestyle='-')
# plt.plot(x, ODP_LS_satisfied, marker='o', label='ODP-LS', linestyle='--')
# plt.plot(x, ODP_TS_satisfied, marker='^', label='ODP-TS', linestyle='--')
# # Add labels and title
# plt.xlabel('Number of Workflows')
# plt.ylabel('Satisfied Workflows')
# filename = "results/figures/real/satisfied.eps"
# # Add a legend
# plt.legend()
# plt.show()
# plt.clf()
# # Display the chart
# foo_fig = plt.gcf()  # 'get current figure'
# foo_fig.savefig(filename, format='eps', dpi=2000)
# plt.show()