import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Sample data
# y = [0.0109, 0.3787, 0.7894, 12.4716, 90.7383, 194.47]
df = pd.read_csv('results/evaluation_21.csv')
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
greedy_acc_cpu = df[df['algorithm'] == 'Greedy_accfirst']['CPU usage'].tolist()
greedy_acc_memory = df[df['algorithm'] == 'Greedy_accfirst']['Memory usage'].tolist()

greedy_delay_time = df[df['algorithm'] == 'Greedy_delayfirst']['time'].tolist()
greedy_delay_objective = df[df['algorithm'] == 'Greedy_delayfirst']['Normalized objective'].tolist()
greedy_delay_cpu = df[df['algorithm'] == 'Greedy_delayfirst']['CPU usage'].tolist()
greedy_delay_memory = df[df['algorithm'] == 'Greedy_delayfirst']['Memory usage'].tolist()

greedy_multi_time = df[df['algorithm'] == 'Greedy_multi']['time'].tolist()
greedy_multi_objective = df[df['algorithm'] == 'Greedy_multi']['Normalized objective'].tolist()
greedy_multi_cpu = df[df['algorithm'] == 'Greedy_multi']['CPU usage'].tolist()
greedy_multi_memory = df[df['algorithm'] == 'Greedy_multi']['Memory usage'].tolist()

local_new_time = df[df['algorithm'] == 'LocalSearch_new']['time'].tolist()
local_new_objective = df[df['algorithm'] == 'LocalSearch_new']['Normalized objective'].tolist()
local_new_cpu = df[df['algorithm'] == 'LocalSearch_new']['CPU usage'].tolist()
local_new_memory = df[df['algorithm'] == 'LocalSearch_new']['Memory usage'].tolist()

ILS_time = df[df['algorithm'] == 'ILS']['time'].tolist()
ILS_objective = df[df['algorithm'] == 'ILS']['Normalized objective'].tolist()
ILS_cpu = df[df['algorithm'] == 'ILS']['CPU usage'].tolist()
ILS_memory = df[df['algorithm'] == 'ILS']['Memory usage'].tolist()

ODP_LS_objective = df[df['algorithm'] == 'ODP-LS']['Normalized objective'].tolist()
ODP_LS_time = df[df['algorithm'] == 'ODP-LS']['time'].tolist()
ODP_LS_cpu = df[df['algorithm'] == 'ODP-LS']['CPU usage'].tolist()
ODP_LS_memory = df[df['algorithm'] == 'ODP-LS']['Memory usage'].tolist()

ODP_TS_objective = df[df['algorithm'] == 'ODP-TS']['Normalized objective'].tolist()
ODP_TS_time = df[df['algorithm'] == 'ODP-TS']['time'].tolist()
ODP_TS_cpu = df[df['algorithm'] == 'ODP-TS']['CPU usage'].tolist()
ODP_TS_memory = df[df['algorithm'] == 'ODP-TS']['Memory usage'].tolist()


bar_width = 0.1

# objective
# plt.bar(x, greedy_acc_objective, width=bar_width, label='MSP-Greedy-Acc')
# plt.bar(x + bar_width, greedy_delay_objective, width=bar_width, label='MSP-Greedy-Del')
# plt.bar(x + 2*bar_width, greedy_multi_objective, width=bar_width, label='MSP-Greedy-Multi')
# plt.bar(x + 3*bar_width, local_new_objective, width=bar_width, label='MSP-LS')
# plt.bar(x + 4*bar_width, ILS_objective, width=bar_width, label='MSP-ILS')
# plt.bar(x + 5*bar_width, ODP_LS_objective, width=bar_width, label='ODP-LS')
# plt.bar(x + 6*bar_width, ODP_TS_objective, width=bar_width, label='ODP-TS')
#
#
# # Add labels and title
# plt.xlabel('Number of workflows')
# plt.ylabel('Objective')
# plt.xticks(x + 2*bar_width / 2, categories)

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
# # plt.bar(x, greedy_acc_cpu, width=bar_width, label='MSP-Greedy-Acc')
# # plt.bar(x + bar_width, greedy_delay_cpu, width=bar_width, label='MSP-Greedy-Del')
# # plt.bar(x + 2*bar_width, greedy_multi_cpu, width=bar_width, label='MSP-Greedy-Multi')
# plt.bar(x + 1*bar_width, local_new_cpu, width=bar_width, label='MSP-LS')
# plt.bar(x + 2*bar_width, ILS_cpu, width=bar_width, label='MSP-ILS')
# plt.bar(x + 3*bar_width, ODP_LS_cpu, width=bar_width, label='ODP-LS')
# plt.bar(x + 4*bar_width, ODP_TS_cpu, width=bar_width, label='ODP-TS')
# # Add labels and title
# plt.xlabel('Number of workflows')
# plt.ylabel('CPU Usage(%)')
# plt.xticks(x + 2*bar_width / 2, categories)
# filename = "results/figures/cpu_usage.eps"
# # Add a legend
# plt.legend()

# memory consumption
# plt.bar(x, greedy_acc_cpu, width=bar_width, label='MSP-Greedy-Acc')
# plt.bar(x + bar_width, greedy_delay_cpu, width=bar_width, label='MSP-Greedy-Del')
# plt.bar(x + 2*bar_width, greedy_multi_cpu, width=bar_width, label='MSP-Greedy-Multi')
plt.bar(x + 1*bar_width, local_new_memory, width=bar_width, label='MSP-LS')
plt.bar(x + 2*bar_width, ILS_memory, width=bar_width, label='MSP-ILS')
plt.bar(x + 3*bar_width, ODP_LS_memory, width=bar_width, label='ODP-LS')
plt.bar(x + 4*bar_width, ODP_TS_memory, width=bar_width, label='ODP-TS')
# Add labels and title
plt.xlabel('Number of workflows')
plt.ylabel('Memory usage(%)')
plt.xticks(x + 2*bar_width / 2, categories)
filename = "results/figures/memory_usage.eps"
# Add a legend
plt.legend()

# Display the chart
foo_fig = plt.gcf()  # 'get current figure'
foo_fig.savefig(filename, format='eps', dpi=2000)
plt.show()