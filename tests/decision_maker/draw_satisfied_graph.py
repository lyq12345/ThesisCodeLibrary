import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
algorithms = ["AMPHI-LS", "AMPHI-ILS", "ODP-LS", "ODP-TS"]

# x_specify = []
# for i in range(0, 101, 10):
#     x_specify.append(i)
df = pd.read_csv('results/simulations/evaluation_satisfied_workflows.csv')
# df = df[df['group'].isin(x_specify)]
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100]
grouped_by_solver = df.groupby('algorithm')
label_fontsize=34
tick_fontsize=25
# solvers = ["AMPHI-LS", "AMPHI-ILS", "ODP-LS", "ODP-TS"]
legend_size = 25
bar_width = 0.2
categories = sorted(df['dev_num'].unique())
conf_co = 0.95
x = np.arange(len(categories))
fig_size = ((11, 8.2))
data = {"AMPHI-LS": [], "AMPHI-ILS": [], "ODP-LS": [], "ODP-TS": []}
data_err = {"AMPHI-LS": [], "AMPHI-ILS": [], "ODP-LS": [], "ODP-TS": []}

for solver, group_data in grouped_by_solver:
    print("Algorithm:", solver)
    name = None
    if solver == "LocalSearch_new":
        name = "AMPHI-LS"
    elif solver == "ILS":
        name = "AMPHI-ILS"
    elif solver == "ODP-LS":
        name = "ODP-LS"
    elif solver == "ODP-TS":
        name = "ODP-TS"
    group_by_dev_num = group_data.groupby('dev_num')
    for dev_num, satisfied_data in group_by_dev_num:
        average_satisfied = satisfied_data["satisfied"].mean()
        std = satisfied_data["satisfied"].std()
        err = conf_co * (std / math.sqrt(len(satisfied_data)))
        data[name].append(average_satisfied)
        data_err[name].append(err)

offset = [-1.5, -0.5, 0.5, 1.5]
count = 0
plt.figure(figsize=fig_size)
for solver, satisfied in data.items():
    plt.bar(x+offset[count]*bar_width, satisfied, yerr=data_err[solver] ,width=bar_width, label=solver)
    # Add labels and title

    # foo_fig = plt.gcf()  # 'get current figure'
    # foo_fig.savefig(filename, format='eps', dpi=2000)
    # Add a legend
    # plt.show()
    # plt.clf()
    count += 1

plt.xlabel('Number of Devices', fontsize=label_fontsize)
plt.ylabel('Satisfied Workflow Number', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize, ticks=x, labels=categories)
plt.yticks(fontsize=tick_fontsize)
plt.legend(fontsize=legend_size)
filename = "results/figures/simulation/bar/satisfied_devnumber.eps"
foo_fig = plt.gcf()  # 'get current figure'
foo_fig.savefig(filename, format='eps', dpi=2000)
plt.show()

# categories = sorted(df['group'].unique())
# x = np.arange(len(categories))
#
# label_fontsize=15
# tick_fontsize=13
#
# # ortools_time = df[df['algorithm'] == 'ORTools']['time'].tolist()
# # ortools_objective = df[df['algorithm'] == 'ORTools']['Normalized objective'].tolist()
# # topsis_time = df[df['algorithm'] == 'TOPSIS']['time'].tolist()
# # topsis_objective = df[df['algorithm'] == 'TOPSIS']['Normalized objective'].tolist()
# # local_time = df[df['algorithm'] == 'LocalSearch']['time'].tolist()
# # local_objective = df[df['algorithm'] == 'LocalSearch']['Normalized objective'].tolist()
#
# colors = [(1, 0.5, 0.5), (0.5, 0, 0), (1, 0.5, 0.5), (1, 0.5, 0.5)]
#
# metrics = ["time", "objective", ""]
#
# local_new_time = df[df['algorithm'] == 'LocalSearch_new']['time'].tolist()
# local_new_objective = df[df['algorithm'] == 'LocalSearch_new']['Normalized objective'].tolist()
# local_new_cpu = df[df['algorithm'] == 'LocalSearch_new']['CPU usage'].tolist()
# local_new_memory = df[df['algorithm'] == 'LocalSearch_new']['Memory usage'].tolist()
# local_new_accuracy = df[df['algorithm'] == 'LocalSearch_new']['Average accuracy'].tolist()
# local_new_delay = df[df['algorithm'] == 'LocalSearch_new']['Average delay'].tolist()
# local_new_satisfied = df[df['algorithm'] == 'LocalSearch_new']['Satisfied workflows'].tolist()
# local_new_time_err = df[df['algorithm'] == 'LocalSearch_new']['time_err'].tolist()
# local_new_obj_err = df[df['algorithm'] == 'LocalSearch_new']['obj_err'].tolist()
# local_new_cpu_err = df[df['algorithm'] == 'LocalSearch_new']['cpu_err'].tolist()
# local_new_memory_err = df[df['algorithm'] == 'LocalSearch_new']['mem_err'].tolist()
# local_new_acc_err = df[df['algorithm'] == 'LocalSearch_new']['acc_err'].tolist()
# local_new_delay_err = df[df['algorithm'] == 'LocalSearch_new']['delay_err'].tolist()
# local_new_satisfied_err = df[df['algorithm'] == 'LocalSearch_new']['satisfied_err'].tolist()
#
# ILS_time = df[df['algorithm'] == 'ILS']['time'].tolist()
# ILS_objective = df[df['algorithm'] == 'ILS']['Normalized objective'].tolist()
# ILS_cpu = df[df['algorithm'] == 'ILS']['CPU usage'].tolist()
# ILS_memory = df[df['algorithm'] == 'ILS']['Memory usage'].tolist()
# ILS_accuracy = df[df['algorithm'] == 'ILS']['Average accuracy'].tolist()
# ILS_delay = df[df['algorithm'] == 'ILS']['Average delay'].tolist()
# ILS_satisfied = df[df['algorithm'] == 'ILS']['Satisfied workflows'].tolist()
# ILS_time_err = df[df['algorithm'] == 'ILS']['time_err'].tolist()
# ILS_obj_err = df[df['algorithm'] == 'ILS']['obj_err'].tolist()
# ILS_cpu_err = df[df['algorithm'] == 'ILS']['cpu_err'].tolist()
# ILS_memory_err = df[df['algorithm'] == 'ILS']['mem_err'].tolist()
# ILS_acc_err = df[df['algorithm'] == 'ILS']['acc_err'].tolist()
# ILS_delay_err = df[df['algorithm'] == 'ILS']['delay_err'].tolist()
# ILS_satisfied_err = df[df['algorithm'] == 'ILS']['satisfied_err'].tolist()
#
# ODP_LS_objective = df[df['algorithm'] == 'ODP-LS']['Normalized objective'].tolist()
# ODP_LS_time = df[df['algorithm'] == 'ODP-LS']['time'].tolist()
# ODP_LS_cpu = df[df['algorithm'] == 'ODP-LS']['CPU usage'].tolist()
# ODP_LS_memory = df[df['algorithm'] == 'ODP-LS']['Memory usage'].tolist()
# ODP_LS_accuracy = df[df['algorithm'] == 'ODP-LS']['Average accuracy'].tolist()
# ODP_LS_delay = df[df['algorithm'] == 'ODP-LS']['Average delay'].tolist()
# ODP_LS_satisfied = df[df['algorithm'] == 'ODP-LS']['Satisfied workflows'].tolist()
# ODP_LS_time_err = df[df['algorithm'] == 'ODP-LS']['time_err'].tolist()
# ODP_LS_obj_err = df[df['algorithm'] == 'ODP-LS']['obj_err'].tolist()
# ODP_LS_cpu_err = df[df['algorithm'] == 'ODP-LS']['cpu_err'].tolist()
# ODP_LS_memory_err = df[df['algorithm'] == 'ODP-LS']['mem_err'].tolist()
# ODP_LS_acc_err = df[df['algorithm'] == 'ODP-LS']['acc_err'].tolist()
# ODP_LS_delay_err = df[df['algorithm'] == 'ODP-LS']['delay_err'].tolist()
# ODP_LS_satisfied_err = df[df['algorithm'] == 'ODP-LS']['satisfied_err'].tolist()
#
# ODP_TS_objective = df[df['algorithm'] == 'ODP-TS']['Normalized objective'].tolist()
# ODP_TS_time = df[df['algorithm'] == 'ODP-TS']['time'].tolist()
# ODP_TS_cpu = df[df['algorithm'] == 'ODP-TS']['CPU usage'].tolist()
# ODP_TS_memory = df[df['algorithm'] == 'ODP-TS']['Memory usage'].tolist()
# ODP_TS_accuracy = df[df['algorithm'] == 'ODP-TS']['Average accuracy'].tolist()
# ODP_TS_delay = df[df['algorithm'] == 'ODP-TS']['Average delay'].tolist()
# ODP_TS_satisfied = df[df['algorithm'] == 'ODP-TS']['Satisfied workflows'].tolist()
# ODP_TS_time_err = df[df['algorithm'] == 'ODP-TS']['time_err'].tolist()
# ODP_TS_obj_err = df[df['algorithm'] == 'ODP-TS']['obj_err'].tolist()
# ODP_TS_cpu_err = df[df['algorithm'] == 'ODP-TS']['cpu_err'].tolist()
# ODP_TS_memory_err = df[df['algorithm'] == 'ODP-TS']['mem_err'].tolist()
# ODP_TS_acc_err = df[df['algorithm'] == 'ODP-TS']['acc_err'].tolist()
# ODP_TS_delay_err = df[df['algorithm'] == 'ODP-TS']['delay_err'].tolist()
# ODP_TS_satisfied_err = df[df['algorithm'] == 'ODP-TS']['satisfied_err'].tolist()
#
# bar_width = 0.2
#
# # satisfied workflows
# plt.bar(x , local_new_satisfied, yerr=local_new_satisfied_err ,width=bar_width, label='MSP-LS')
# plt.bar(x + 1*bar_width, ILS_satisfied, yerr=ILS_satisfied_err ,width=bar_width, label='MSP-ILS')
# plt.bar(x + 2*bar_width, ODP_LS_satisfied, yerr=ODP_LS_satisfied_err, width=bar_width, label='ODP-LS')
# plt.bar(x + 3*bar_width, ODP_TS_satisfied, yerr=ODP_TS_satisfied_err ,width=bar_width, label='ODP-TS')
# # Add labels and title
# plt.xlabel('Number of Workflows', fontsize=label_fontsize)
# plt.ylabel('Satisfied Workflows', fontsize=label_fontsize)
# plt.xticks(x + 2*bar_width / 2, categories, fontsize=tick_fontsize)
# plt.yticks(fontsize=tick_fontsize)
# plt.legend()
# filename = "results/figures/simulation/bar/satisfied.eps"
# foo_fig = plt.gcf()  # 'get current figure'
# foo_fig.savefig(filename, format='eps', dpi=2000)
# # Add a legend
# plt.show()
# plt.clf()