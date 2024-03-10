import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Sample data
# y = [0.0109, 0.3787, 0.7894, 12.4716, 90.7383, 194.47]
x_specify = [1,3,5,7,11,13,15]
df = pd.read_csv('results/evaluation_22_real.csv')
df = df[df['group'].isin(x_specify)]
df2 = pd.read_csv('results/evaluation_24_real.csv')
df2 = df2[df2['group'].isin(x_specify)]
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100]
categories = sorted(df['group'].unique())
x = np.arange(len(categories))

# ortools_time = df[df['algorithm'] == 'ORTools']['time'].tolist()
# ortools_objective = df[df['algorithm'] == 'ORTools']['Normalized objective'].tolist()
# topsis_time = df[df['algorithm'] == 'TOPSIS']['time'].tolist()
# topsis_objective = df[df['algorithm'] == 'TOPSIS']['Normalized objective'].tolist()
# local_time = df[df['algorithm'] == 'LocalSearch']['time'].tolist()
# local_objective = df[df['algorithm'] == 'LocalSearch']['Normalized objective'].tolist()

metrics = ["time", "objective", ""]

local_new_time = df[df['algorithm'] == 'LocalSearch_new']['time'].tolist()
local_new_objective = df[df['algorithm'] == 'LocalSearch_new']['Normalized objective'].tolist()
local_new_cpu = df[df['algorithm'] == 'LocalSearch_new']['CPU usage'].tolist()
local_new_memory = df[df['algorithm'] == 'LocalSearch_new']['Memory usage'].tolist()
local_new_accuracy = df[df['algorithm'] == 'LocalSearch_new']['Average accuracy'].tolist()
local_new_delay = df[df['algorithm'] == 'LocalSearch_new']['Average delay'].tolist()
local_new_satisfied = df[df['algorithm'] == 'LocalSearch_new']['Satisfied workflows'].tolist()
local_new_time_err = df2[df2['algorithm'] == 'LocalSearch_new']['time_err'].tolist()
local_new_obj_err = df2[df2['algorithm'] == 'LocalSearch_new']['obj_err'].tolist()
local_new_cpu_err = df2[df2['algorithm'] == 'LocalSearch_new']['cpu_err'].tolist()
local_new_memory_err = df2[df2['algorithm'] == 'LocalSearch_new']['mem_err'].tolist()
local_new_acc_err = df2[df2['algorithm'] == 'LocalSearch_new']['acc_err'].tolist()
local_new_delay_err = df2[df2['algorithm'] == 'LocalSearch_new']['delay_err'].tolist()
local_new_satisfied_err = df2[df2['algorithm'] == 'LocalSearch_new']['satisfied_err'].tolist()

ILS_time = df[df['algorithm'] == 'ILS']['time'].tolist()
ILS_objective = df[df['algorithm'] == 'ILS']['Normalized objective'].tolist()
ILS_cpu = df[df['algorithm'] == 'ILS']['CPU usage'].tolist()
ILS_memory = df[df['algorithm'] == 'ILS']['Memory usage'].tolist()
ILS_accuracy = df[df['algorithm'] == 'ILS']['Average accuracy'].tolist()
ILS_delay = df[df['algorithm'] == 'ILS']['Average delay'].tolist()
ILS_satisfied = df[df['algorithm'] == 'ILS']['Satisfied workflows'].tolist()
ILS_time_err = df2[df2['algorithm'] == 'ILS']['time_err'].tolist()
ILS_obj_err = df2[df2['algorithm'] == 'ILS']['obj_err'].tolist()
ILS_cpu_err = df2[df2['algorithm'] == 'ILS']['cpu_err'].tolist()
ILS_memory_err = df2[df2['algorithm'] == 'ILS']['mem_err'].tolist()
ILS_acc_err = df2[df2['algorithm'] == 'ILS']['acc_err'].tolist()
ILS_delay_err = df2[df2['algorithm'] == 'ILS']['delay_err'].tolist()
ILS_satisfied_err = df2[df2['algorithm'] == 'ILS']['satisfied_err'].tolist()

ODP_LS_objective = df[df['algorithm'] == 'ODP-LS']['Normalized objective'].tolist()
ODP_LS_time = df[df['algorithm'] == 'ODP-LS']['time'].tolist()
ODP_LS_cpu = df[df['algorithm'] == 'ODP-LS']['CPU usage'].tolist()
ODP_LS_memory = df[df['algorithm'] == 'ODP-LS']['Memory usage'].tolist()
ODP_LS_accuracy = df[df['algorithm'] == 'ODP-LS']['Average accuracy'].tolist()
ODP_LS_delay = df[df['algorithm'] == 'ODP-LS']['Average delay'].tolist()
ODP_LS_satisfied = df[df['algorithm'] == 'ODP-LS']['Satisfied workflows'].tolist()
ODP_LS_time_err = df2[df2['algorithm'] == 'ODP-LS']['time_err'].tolist()
ODP_LS_obj_err = df2[df2['algorithm'] == 'ODP-LS']['obj_err'].tolist()
ODP_LS_cpu_err = df2[df2['algorithm'] == 'ODP-LS']['cpu_err'].tolist()
ODP_LS_memory_err = df2[df2['algorithm'] == 'ODP-LS']['mem_err'].tolist()
ODP_LS_acc_err = df2[df2['algorithm'] == 'ODP-LS']['acc_err'].tolist()
ODP_LS_delay_err = df2[df2['algorithm'] == 'ODP-LS']['delay_err'].tolist()
ODP_LS_satisfied_err = df2[df2['algorithm'] == 'ODP-LS']['satisfied_err'].tolist()

ODP_TS_objective = df[df['algorithm'] == 'ODP-TS']['Normalized objective'].tolist()
ODP_TS_time = df[df['algorithm'] == 'ODP-TS']['time'].tolist()
ODP_TS_cpu = df[df['algorithm'] == 'ODP-TS']['CPU usage'].tolist()
ODP_TS_memory = df[df['algorithm'] == 'ODP-TS']['Memory usage'].tolist()
ODP_TS_accuracy = df[df['algorithm'] == 'ODP-TS']['Average accuracy'].tolist()
ODP_TS_delay = df[df['algorithm'] == 'ODP-TS']['Average delay'].tolist()
ODP_TS_satisfied = df[df['algorithm'] == 'ODP-TS']['Satisfied workflows'].tolist()
ODP_TS_time_err = df2[df2['algorithm'] == 'ODP-TS']['time_err'].tolist()
ODP_TS_obj_err = df2[df2['algorithm'] == 'ODP-TS']['obj_err'].tolist()
ODP_TS_cpu_err = df2[df2['algorithm'] == 'ODP-TS']['cpu_err'].tolist()
ODP_TS_memory_err = df2[df2['algorithm'] == 'ODP-TS']['mem_err'].tolist()
ODP_TS_acc_err = df2[df2['algorithm'] == 'ODP-TS']['acc_err'].tolist()
ODP_TS_delay_err = df2[df2['algorithm'] == 'ODP-TS']['delay_err'].tolist()
ODP_TS_satisfied_err = df2[df2['algorithm'] == 'ODP-TS']['satisfied_err'].tolist()


bar_width = 0.2

# objective
plt.bar(x, local_new_objective, yerr=local_new_obj_err ,width=bar_width, label='MSP-LS')
plt.bar(x + 1*bar_width, ILS_objective, yerr=ILS_obj_err, width=bar_width, label='MSP-ILS')
plt.bar(x + 2*bar_width, ODP_LS_objective, yerr=ODP_LS_obj_err, width=bar_width, label='ODP-LS')
plt.bar(x + 3*bar_width, ODP_TS_objective, yerr=ODP_TS_obj_err, width=bar_width, label='ODP-TS')
# Add labels and title
plt.xlabel('Number of Workflows')
plt.ylabel('Objective')
plt.xticks(x + 2*bar_width / 2, categories)
plt.legend()
filename = "results/figures/real/objective.eps"
foo_fig = plt.gcf()  # 'get current figure'
# foo_fig.savefig(filename, format='eps', dpi=2000)
plt.show()
plt.clf()


# cpu consumption
plt.bar(x, local_new_cpu, yerr=local_new_cpu_err ,width=bar_width, label='MSP-LS')
plt.bar(x + 1*bar_width, ILS_cpu, yerr=ILS_cpu_err ,width=bar_width, label='MSP-ILS')
plt.bar(x + 2*bar_width, ODP_LS_cpu, yerr=ODP_LS_cpu_err, width=bar_width, label='ODP-LS')
plt.bar(x + 3*bar_width, ODP_TS_cpu, yerr=ODP_TS_cpu_err, width=bar_width, label='ODP-TS')
# Add labels and title
plt.xlabel('Number of Workflows')
plt.ylabel('CPU Usage(%)')
plt.legend()
plt.xticks(x + 2*bar_width / 2, categories)
filename = "results/figures/real/cpu_usage.eps"
foo_fig = plt.gcf()  # 'get current figure'
# foo_fig.savefig(filename, format='eps', dpi=2000)
# Add a legend
plt.show()
plt.clf()
#
# memory consumption
plt.bar(x, local_new_memory, yerr=local_new_memory_err, width=bar_width, label='MSP-LS')
plt.bar(x + 1*bar_width, ILS_memory, yerr=ILS_memory_err, width=bar_width, label='MSP-ILS')
plt.bar(x + 2*bar_width, ODP_LS_memory, yerr=ODP_LS_memory_err ,width=bar_width, label='ODP-LS')
plt.bar(x + 3*bar_width, ODP_TS_memory, yerr=ODP_TS_memory_err, width=bar_width, label='ODP-TS')
# Add labels and title
plt.xlabel('Number of Workflows')
plt.ylabel('Memory usage(%)')
plt.xticks(x + 2*bar_width / 2, categories)
filename = "results/figures/real/memory_usage.eps"
foo_fig = plt.gcf()  # 'get current figure'
# foo_fig.savefig(filename, format='eps', dpi=2000)
# Add a legend
plt.legend()
plt.show()

# accuracy
plt.bar(x, local_new_accuracy, yerr=local_new_acc_err, width=bar_width, label='MSP-LS')
plt.bar(x + 1*bar_width, ILS_accuracy, yerr=ILS_acc_err, width=bar_width, label='MSP-ILS')
plt.bar(x + 2*bar_width, ODP_LS_accuracy, yerr=ODP_LS_acc_err ,width=bar_width, label='ODP-LS')
plt.bar(x + 3*bar_width, ODP_TS_accuracy, yerr=ODP_TS_acc_err, width=bar_width, label='ODP-TS')
# Add labels and title
plt.xlabel('Number of Workflows')
plt.ylabel('Accuracy(%)')
plt.xticks(x + 2*bar_width / 2, categories)
plt.legend()
filename = "results/figures/real/accuracy.eps"
foo_fig = plt.gcf()  # 'get current figure'
# foo_fig.savefig(filename, format='eps', dpi=2000)
# Add a legend
plt.show()
# plt.clf()

# delay
plt.bar(x, local_new_delay, yerr=local_new_delay_err ,width=bar_width, label='MSP-LS')
plt.bar(x + 1*bar_width, ILS_delay, yerr=ILS_delay_err ,width=bar_width, label='MSP-ILS')
plt.bar(x + 2*bar_width, ODP_LS_delay, yerr=ODP_LS_delay_err ,width=bar_width, label='ODP-LS')
plt.bar(x + 3*bar_width, ODP_TS_delay, yerr=ODP_TS_delay_err , width=bar_width, label='ODP-TS')
# Add labels and title
plt.xlabel('Number of Workflows')
plt.ylabel('Average Delay(s)')
plt.xticks(x + 2*bar_width / 2, categories)
plt.legend()
filename = "results/figures/real/delay.eps"
foo_fig = plt.gcf()  # 'get current figure'
# foo_fig.savefig(filename, format='eps', dpi=2000)
plt.show()
plt.clf()

# satisfied workflows
plt.bar(x , local_new_satisfied, yerr=local_new_satisfied_err ,width=bar_width, label='MSP-LS')
plt.bar(x + 1*bar_width, ILS_satisfied, yerr=ILS_satisfied_err ,width=bar_width, label='MSP-ILS')
plt.bar(x + 2*bar_width, ODP_LS_satisfied, yerr=ODP_LS_satisfied_err, width=bar_width, label='ODP-LS')
plt.bar(x + 3*bar_width, ODP_TS_satisfied, yerr=ODP_TS_satisfied_err ,width=bar_width, label='ODP-TS')
# Add labels and title
plt.xlabel('Number of Workflows')
plt.ylabel('Satisfied Workflows')
plt.xticks(x + 2*bar_width / 2, categories)
plt.legend()
filename = "results/figures/real/satisfied.eps"
foo_fig = plt.gcf()  # 'get current figure'
# foo_fig.savefig(filename, format='eps', dpi=2000)
# Add a legend

plt.show()
plt.clf()