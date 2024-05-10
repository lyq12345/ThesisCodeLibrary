import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# Sample data
# y = [0.0109, 0.3787, 0.7894, 12.4716, 90.7383, 194.47]
x_specify = []
for i in range(1, 50, 2):
    x_specify.append(i)
df = pd.read_csv('results/evaluation_22_real.csv')
df = df[df['group'].isin(x_specify)]
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100]
categories = sorted(df['group'].unique())
# x = np.arange(len(categories))
x = [1, 3, 5, 7, 9, 11, 13, 15]
label_fontize=34
tick_fontsize=25
legend_size = 18
fig_size=(11,8.2)

# solvers = ["LocalSearch_new", "ILS", "Greedy_accfirst", "Greedy_delayfirst", "Greedy_multi", "ODP-LS", "ODP-TS"]
solvers = ["LocalSearch_new", "ILS", "ODP-LS", "ODP-TS"]
# metrics = ["Normalized objective", "Average accuracy", "Average delay", "CPU usage", "Memory usage", "Satisfied workflows"]
metrics = ["Normalized objective", "Average accuracy", "Average delay"]
label_dict = {
  "LocalSearch_new": {
    "label": "AMPHI-LS",
    "marker": "o",
    "linestyle": "-"
  },
  "ILS": {
    "label": "AMPHI-ILS",
    "marker": "^",
    "linestyle": "-"
  },
"Greedy_accfirst": {
    "label": "AccFirst",
    "marker": "x",
    "linestyle": "-"
  },
  "Greedy_delayfirst": {
    "label": "DelFirst",
    "marker": "x",
    "linestyle": "-"
  },
"Greedy_multi": {
    "label": "GreedyMulti",
    "marker": "x",
    "linestyle": "-"
},
"ODP-LS": {
    "label": "ODP-LS",
    "marker": "0",
    "linestyle": "--"
  },
  "ODP-TS": {
    "label": "ODP-TS",
    "marker": "^",
    "linestyle": "--"
  }
}

metrics_mapping = {
  "Normalized objective": {
    "label": "Acc-Del Utility",
    "name": "objective"
  },
  "Average accuracy": {
    "label": "Accuracy (%)",
    "name": "accuracy"
  },
  "Average delay":{
    "label": "Average Delay (s)",
    "name": "delay"
  },
  "CPU usage": {
    "label": "CPU Usage (%)",
    "name": "cpu_usage"
  },
  "Memory usage": {
    "label": "Memory Usage (%)",
    "name": "memory_usage"
  },
  "Satisfied workflows": {
    "label": "Satisfied Workflow No.",
    "name": "satisfied"
  }
}

# greedy_acc_time = df[df['algorithm'] == 'Greedy_accfirst']['time'].tolist()
# greedy_acc_objective = df[df['algorithm'] == 'Greedy_accfirst']['Normalized objective'].tolist()
# greedy_acc_cpu = df[df['algorithm'] == 'Greedy_accfirst']['CPU usage'].tolist()
# greedy_acc_memory = df[df['algorithm'] == 'Greedy_accfirst']['Memory usage'].tolist()
# greedy_acc_accuracy = df[df['algorithm'] == 'Greedy_accfirst']['Average accuracy'].tolist()
# greedy_acc_delay = df[df['algorithm'] == 'Greedy_accfirst']['Average delay'].tolist()
# greedy_acc_satisfied = df[df['algorithm'] == 'Greedy_accfirst']['Satisfied workflows'].tolist()
# greedy_acc_time_err = df[df['algorithm'] == 'Greedy_accfirst']['time_err'].tolist()
# greedy_acc_obj_err = df[df['algorithm'] == 'Greedy_accfirst']['obj_err'].tolist()
# greedy_acc_cpu_err = df[df['algorithm'] == 'Greedy_accfirst']['cpu_err'].tolist()
# greedy_acc_memory_err = df[df['algorithm'] == 'Greedy_accfirst']['mem_err'].tolist()
# greedy_acc_acc_err = df[df['algorithm'] == 'Greedy_accfirst']['acc_err'].tolist()
# greedy_acc_delay_err = df[df['algorithm'] == 'Greedy_accfirst']['delay_err'].tolist()
# greedy_acc_satisfied_err = df[df['algorithm'] == 'Greedy_accfirst']['satisfied_err'].tolist()
# greedy_acc_cpu = [x * 100 for x in greedy_acc_cpu]
# greedy_acc_memory = [x * 100 for x in greedy_acc_memory]
# greedy_acc_accuracy = [x * 100 for x in greedy_acc_accuracy]
#
# greedy_delay_time = df[df['algorithm'] == 'Greedy_delayfirst']['time'].tolist()
# greedy_delay_objective = df[df['algorithm'] == 'Greedy_delayfirst']['Normalized objective'].tolist()
# greedy_delay_cpu = df[df['algorithm'] == 'Greedy_delayfirst']['CPU usage'].tolist()
# greedy_delay_memory = df[df['algorithm'] == 'Greedy_delayfirst']['Memory usage'].tolist()
# greedy_delay_accuracy = df[df['algorithm'] == 'Greedy_delayfirst']['Average accuracy'].tolist()
# greedy_delay_delay = df[df['algorithm'] == 'Greedy_delayfirst']['Average delay'].tolist()
# greedy_delay_satisfied = df[df['algorithm'] == 'Greedy_delayfirst']['Satisfied workflows'].tolist()
# greedy_delay_time_err = df[df['algorithm'] == 'Greedy_delayfirst']['time_err'].tolist()
# greedy_delay_obj_err = df[df['algorithm'] == 'Greedy_delayfirst']['obj_err'].tolist()
# greedy_delay_cpu_err = df[df['algorithm'] == 'Greedy_delayfirst']['cpu_err'].tolist()
# greedy_delay_memory_err = df[df['algorithm'] == 'Greedy_delayfirst']['mem_err'].tolist()
# greedy_delay_acc_err = df[df['algorithm'] == 'Greedy_delayfirst']['acc_err'].tolist()
# greedy_delay_delay_err = df[df['algorithm'] == 'Greedy_delayfirst']['delay_err'].tolist()
# greedy_delay_satisfied_err = df[df['algorithm'] == 'Greedy_delayfirst']['satisfied_err'].tolist()
# greedy_delay_cpu = [x * 100 for x in greedy_delay_cpu]
# greedy_delay_memory = [x * 100 for x in greedy_delay_memory]
# greedy_delay_accuracy = [x * 100 for x in greedy_delay_accuracy]
#
# greedy_multi_time = df[df['algorithm'] == 'Greedy_multi']['time'].tolist()
# greedy_multi_objective = df[df['algorithm'] == 'Greedy_multi']['Normalized objective'].tolist()
# greedy_multi_cpu = df[df['algorithm'] == 'Greedy_multi']['CPU usage'].tolist()
# greedy_multi_memory = df[df['algorithm'] == 'Greedy_multi']['Memory usage'].tolist()
# greedy_multi_accuracy = df[df['algorithm'] == 'Greedy_multi']['Average accuracy'].tolist()
# greedy_multi_delay = df[df['algorithm'] == 'Greedy_multi']['Average delay'].tolist()
# greedy_multi_satisfied = df[df['algorithm'] == 'Greedy_multi']['Satisfied workflows'].tolist()
# greedy_multi_time_err = df[df['algorithm'] == 'Greedy_multi']['time_err'].tolist()
# greedy_multi_obj_err = df[df['algorithm'] == 'Greedy_multi']['obj_err'].tolist()
# greedy_multi_cpu_err = df[df['algorithm'] == 'Greedy_multi']['cpu_err'].tolist()
# greedy_multi_memory_err = df[df['algorithm'] == 'Greedy_multi']['mem_err'].tolist()
# greedy_multi_acc_err = df[df['algorithm'] == 'Greedy_multi']['acc_err'].tolist()
# greedy_multi_delay_err = df[df['algorithm'] == 'Greedy_multi']['delay_err'].tolist()
# greedy_multi_satisfied_err = df[df['algorithm'] == 'Greedy_multi']['satisfied_err'].tolist()
# greedy_multi_cpu = [x * 100 for x in greedy_multi_cpu]
# greedy_multi_memory = [x * 100 for x in greedy_multi_memory]
# greedy_multi_accuracy = [x * 100 for x in greedy_multi_accuracy]
#
# local_new_time = df[df['algorithm'] == 'LocalSearch_new']['time'].tolist()
# local_new_objective = df[df['algorithm'] == 'LocalSearch_new']['Normalized objective'].tolist()
# local_new_cpu = df[df['algorithm'] == 'LocalSearch_new']['CPU usage'].tolist()
# local_new_cpu = [x * 100 for x in local_new_cpu]
# local_new_memory = df[df['algorithm'] == 'LocalSearch_new']['Memory usage'].tolist()
# local_new_memory = [x * 100 for x in local_new_memory]
# local_new_accuracy = df[df['algorithm'] == 'LocalSearch_new']['Average accuracy'].tolist()
# local_new_accuracy = [x * 100 for x in local_new_accuracy]
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
# ILS_cpu = [x * 100 for x in ILS_cpu]
# ILS_memory = df[df['algorithm'] == 'ILS']['Memory usage'].tolist()
# ILS_memory = [x * 100 for x in ILS_memory]
# ILS_accuracy = df[df['algorithm'] == 'ILS']['Average accuracy'].tolist()
# ILS_accuracy = [x * 100 for x in ILS_accuracy]
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
# ODP_LS_cpu = [x * 100 for x in ODP_LS_cpu]
# ODP_LS_memory = df[df['algorithm'] == 'ODP-LS']['Memory usage'].tolist()
# ODP_LS_memory = [x * 100 for x in ODP_LS_memory]
# ODP_LS_accuracy = df[df['algorithm'] == 'ODP-LS']['Average accuracy'].tolist()
# ODP_LS_accuracy = [x * 100 for x in ODP_LS_accuracy]
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
# ODP_TS_cpu = [x * 100 for x in ODP_TS_cpu]
# ODP_TS_memory = df[df['algorithm'] == 'ODP-TS']['Memory usage'].tolist()
# ODP_TS_memory = [x * 100 for x in ODP_TS_memory]
# ODP_TS_accuracy = df[df['algorithm'] == 'ODP-TS']['Average accuracy'].tolist()
# ODP_TS_accuracy = [x * 100 for x in ODP_TS_accuracy]
# ODP_TS_delay = df[df['algorithm'] == 'ODP-TS']['Average delay'].tolist()
# ODP_TS_satisfied = df[df['algorithm'] == 'ODP-TS']['Satisfied workflows'].tolist()
# ODP_TS_time_err = df[df['algorithm'] == 'ODP-TS']['time_err'].tolist()
# ODP_TS_obj_err = df[df['algorithm'] == 'ODP-TS']['obj_err'].tolist()
# ODP_TS_cpu_err = df[df['algorithm'] == 'ODP-TS']['cpu_err'].tolist()
# ODP_TS_memory_err = df[df['algorithm'] == 'ODP-TS']['mem_err'].tolist()
# ODP_TS_acc_err = df[df['algorithm'] == 'ODP-TS']['acc_err'].tolist()
# ODP_TS_delay_err = df[df['algorithm'] == 'ODP-TS']['delay_err'].tolist()
# ODP_TS_satisfied_err = df[df['algorithm'] == 'ODP-TS']['satisfied_err'].tolist()

def export_legend(legend, fname="legend.png"):
  fig = legend.figure
  fig.canvas.draw()
  bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  fig.savefig(fname, bbox_inches=bbox)


for met in metrics:
  plt.figure(figsize=fig_size)
  for sol in solvers:
    value = df[df['algorithm'] == sol][met].tolist()
    if met == "CPU usage" or met == "Memory usage":
      value = [x * 100 for x in value]
    plt.plot(x, value, label=label_dict[sol]["label"], marker='o', linestyle='-')
    # Add labels and title
    plt.xlabel('Number of Workflows', fontsize=label_fontize)
    plt.ylabel(metrics_mapping[met]["label"], fontsize=label_fontize)
    plt.xticks(range(1, 16, 2), fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    # plt.legend(loc='lower left', ncol=2, fontsize=legend_size)
  filename_pre = f"results/figures/real/line/preview/{metrics_mapping[met]['name']}.png"
  filename = f"results/figures/real/line/{metrics_mapping[met]['name']}.eps"
  foo_fig = plt.gcf()  # 'get current figure'
  foo_fig.savefig(filename_pre, format='png', dpi=2000)
  foo_fig.savefig(filename, format='eps', dpi=2000)
  plt.show()
  plt.clf()

# # objective
# plt.figure(figsize=fig_size)
# plt.plot(x, local_new_objective, label=solvers[0], marker='o', linestyle='-')
# plt.plot(x, ILS_objective, label=solvers[1], marker='^', linestyle='-')
# plt.plot(x, greedy_acc_objective, label=solvers[4], marker='x', linestyle='-')
# plt.plot(x, greedy_delay_objective, label=solvers[5], marker='x', linestyle='-')
# plt.plot(x, greedy_multi_objective, label=solvers[6], marker='x', linestyle='-')
# plt.plot(x, ODP_LS_objective, label=solvers[2], marker='o', linestyle='--')
# plt.plot(x, ODP_TS_objective, label=solvers[3], marker='^', linestyle='--')
# # Add labels and title
# plt.xlabel('Number of Workflows', fontsize=label_fontize)
# plt.ylabel('Average Objective', fontsize=label_fontize)
# plt.xticks(range(1, 16, 2), fontsize=tick_fontsize)
# plt.yticks(fontsize=tick_fontsize)
# # plt.legend(loc='lower left', ncol=2, fontsize=legend_size)
# filename_pre = "results/figures/real/line/preview/objective.png"
# filename = "results/figures/real/line/objective.eps"
# foo_fig = plt.gcf()  # 'get current figure'
# foo_fig.savefig(filename_pre, format='png', dpi=2000)
# foo_fig.savefig(filename, format='eps', dpi=2000)
# plt.show()
# plt.clf()
#
# # cpu consumption
# plt.figure(figsize=fig_size)
# plt.plot(x, local_new_cpu, label=solvers[0], marker='o', linestyle='-')
# plt.plot(x, ILS_cpu, label=solvers[1], marker='^', linestyle='-')
# plt.plot(x, greedy_acc_cpu, label=solvers[4], marker='x', linestyle='-')
# plt.plot(x, greedy_delay_cpu, label=solvers[5], marker='x', linestyle='-')
# plt.plot(x, greedy_multi_cpu, label=solvers[6], marker='x', linestyle='-')
# plt.plot(x, ODP_LS_cpu, label=solvers[2], marker='o', linestyle='--')
# plt.plot(x, ODP_TS_cpu, label=solvers[3], marker='^', linestyle='--')
#
# # Add labels and title
# plt.xlabel('Number of Workflows', fontsize=label_fontize)
# plt.ylabel('CPU Usage(%)', fontsize=label_fontize)
# plt.xticks(range(1, 16, 2), fontsize=tick_fontsize)
# plt.yticks(fontsize=tick_fontsize)
# # plt.legend(fontsize=legend_size)
# filename = "results/figures/real/line/cpu_usage.eps"
# foo_fig = plt.gcf()  # 'get current figure'
# foo_fig.savefig(filename, format='eps', dpi=2000)
# # Add a legend
#
# plt.show()
# plt.clf()
# #
# # memory consumption
# plt.figure(figsize=fig_size)
# plt.plot(x, local_new_memory, marker='o', label=solvers[0],linestyle='-')
# plt.plot(x, ILS_memory, marker='^', label=solvers[1], linestyle='-')
# plt.plot(x, greedy_acc_memory, label=solvers[4], marker='x', linestyle='-')
# plt.plot(x, greedy_delay_memory, label=solvers[5], marker='x', linestyle='-')
# plt.plot(x, greedy_multi_memory, label=solvers[6], marker='x', linestyle='-')
# plt.plot(x, ODP_LS_memory, marker='o', label=solvers[2], linestyle='--')
# plt.plot(x, ODP_TS_memory, marker='^', label=solvers[3], linestyle='--')
#
# # Add labels and title
# plt.xlabel('Number of Workflows', fontsize=label_fontize)
# plt.ylabel('Memory Usage(%)', fontsize=label_fontize)
# plt.xticks(range(1, 16, 2), fontsize=tick_fontsize)
# plt.yticks(fontsize=tick_fontsize)
# # Add a legend
# # plt.legend(fontsize=legend_size)
#
# ax = plt.gca()
#
# # 设置 y 轴刻度格式
# ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.f'))
# filename = "results/figures/real/line/memory_usage.eps"
#
# foo_fig = plt.gcf()  # 'get current figure'
# foo_fig.savefig(filename, format='eps', dpi=2000)
# plt.show()
# plt.clf()
# #
# # accuracy
# plt.figure(figsize=fig_size)
# plt.plot(x, local_new_accuracy, marker='o', label=solvers[0],linestyle='-')
# plt.plot(x, ILS_accuracy, marker='^', label=solvers[1], linestyle='-')
# plt.plot(x, greedy_acc_accuracy, label=solvers[4], marker='x', linestyle='-')
# plt.plot(x, greedy_delay_accuracy, label=solvers[5], marker='x', linestyle='-')
# plt.plot(x, greedy_multi_accuracy, label=solvers[6], marker='x', linestyle='-')
# plt.plot(x, ODP_LS_accuracy, marker='o', label=solvers[2], linestyle='--')
# plt.plot(x, ODP_TS_accuracy, marker='^', label=solvers[3], linestyle='--')
#
# # Add labels and title
# plt.xlabel('Number of Workflows', fontsize=label_fontize)
# plt.ylabel('Accuracy(%)', fontsize=label_fontize)
# plt.xticks(range(1, 16, 2), fontsize=tick_fontsize)
# plt.yticks(fontsize=tick_fontsize)
# # Add a legend
# # plt.legend(bbox_to_anchor=(0.6,0.36), fontsize=legend_size)
# filename = "results/figures/real/line/accuracy.eps"
# foo_fig = plt.gcf()  # 'get current figure'
# foo_fig.savefig(filename, format='eps', dpi=2000)
# plt.show()
# plt.clf()
# #
# # delay
# plt.figure(figsize=fig_size)
# plt.plot(x, local_new_delay, marker='o', label=solvers[0],linestyle='-')
# plt.plot(x, ILS_delay, marker='^', label=solvers[1], linestyle='-')
# plt.plot(x, greedy_acc_delay, label=solvers[4], marker='x', linestyle='-')
# plt.plot(x, greedy_delay_delay, label=solvers[5], marker='x', linestyle='-')
# plt.plot(x, greedy_multi_delay, label=solvers[6], marker='x', linestyle='-')
# plt.plot(x, ODP_LS_delay, marker='o', label=solvers[2], linestyle='--')
# plt.plot(x, ODP_TS_delay, marker='^', label=solvers[3], linestyle='--')
# plt.xlabel('Number of Workflows', fontsize=label_fontize)
# plt.ylabel('Average Delay(s)', fontsize=label_fontize)
# plt.xticks(range(1, 16, 2), fontsize=tick_fontsize)
# plt.yticks(fontsize=tick_fontsize, )
# # plt.legend(bbox_to_anchor=(0.6, 0.3), ncol=2, fontsize=legend_size)
# filename = "results/figures/real/line/delay.eps"
# foo_fig = plt.gcf()  # 'get current figure'
# foo_fig.savefig(filename, format='eps', dpi=2000)
# # Add a legend
# plt.show()
# plt.clf()
#
# # satisfied workflows
# plt.figure(figsize=fig_size)
# plt.plot(x, local_new_satisfied, marker='o', label=solvers[0],linestyle='-')
# plt.plot(x, ILS_satisfied, marker='^', label=solvers[1], linestyle='-')
# plt.plot(x, greedy_acc_satisfied, label=solvers[4], marker='x', linestyle='-')
# plt.plot(x, greedy_delay_satisfied, label=solvers[5], marker='x', linestyle='-')
# plt.plot(x, greedy_multi_satisfied, label=solvers[6], marker='x', linestyle='-')
# plt.plot(x, ODP_LS_satisfied, marker='o', label=solvers[2], linestyle='--')
# plt.plot(x, ODP_TS_satisfied, marker='^', label=solvers[3], linestyle='--')
# # Add labels and title
# plt.xlabel('Number of Workflows', fontsize=label_fontize)
# plt.ylabel('Satisfied Workflows', fontsize=label_fontize)
# plt.xticks(range(1, 16, 2), fontsize=tick_fontsize)
# plt.yticks(fontsize=tick_fontsize)
# # Add a legend
# # plt.legend(fontsize=legend_size)
# filename = "results/figures/real/line/satisfied.eps"
# foo_fig = plt.gcf()  # 'get current figure'
# foo_fig.savefig(filename, format='eps', dpi=2000)
# plt.show()
# plt.clf()

export_legend(plt.legend(fontsize=legend_size), "results/figures/real/legend.eps")
def export_legend(legend, fname="legend.eps"):
  fig = legend.figure
  fig.canvas.draw()
  bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  fig.savefig(fname, bbox_inches=bbox)

