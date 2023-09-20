import pandas as pd
import matplotlib.pyplot as plt

def draw_emp_dist(filename):
    df = pd.read_csv(filename)
    data_list = df['proc_time'].to_list()
    sorted_list = sorted(data_list)

    plt.figure(figsize=(6, 4))
    last, i = min(sorted_list), 0
    while i < len(sorted_list):
        plt.plot([last, sorted_list[i]], [i / len(sorted_list), i / len(sorted_list)], 'k')
        if i < len(sorted_list):
            last = sorted_list[i]
        i += 1
    plt.show()

def draw_hist():
    devices = ['pi', 'nano', 'xaiver']
    versions = ['tinyyolov3', 'yolov3']
    fig, axes = plt.subplots(2, 3, sharex='col', sharey='row')
    for i in range(2):
        version_name = versions[i]
        for j in range(3):
            device_name = devices[j]
            file_name = f"results/raw_{device_name}_human_{version_name}.csv"
            df = pd.read_csv(file_name)
            data = df['proc_time'].to_list()
            axes[i, j].hist(data[i * 3 + j], bins=10, edgecolor='k')
            axes[i, j].set_title(f'{version_name} on {device_name}')
    plt.show()

# draw_emp_dist("results/raw_nano_human_tinyyolov3.csv")
draw_hist()