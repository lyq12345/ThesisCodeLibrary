import matplotlib.pyplot as plt
import pandas as pd
# Sample data
# y = [0.0109, 0.3787, 0.7894, 12.4716, 90.7383, 194.47]
df = pd.read_csv('results/evaluation_8.csv')
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x = sorted(df['group'].unique())

ortools_time = df[df['algorithm'] == 'ORTools']['time'].tolist()
ortools_objective = df[df['algorithm'] == 'ORTools']['Normalized objective'].tolist()
# topsis_time = df[df['algorithm'] == 'TOPSIS']['time'].tolist()
# topsis_objective = df[df['algorithm'] == 'TOPSIS']['Normalized objective'].tolist()
local_time = df[df['algorithm'] == 'LocalSearch']['time'].tolist()
local_objective = df[df['algorithm'] == 'LocalSearch']['Normalized objective'].tolist()

# # objective
# plt.plot(x, ortools_objective, label='ORTools', marker='o', linestyle='-')
# # plt.plot(x, topsis_objective, label='TOPSIS', marker='o', linestyle='-')
# plt.plot(x, local_objective, label='LocalSearch', marker='o', linestyle='-')
#
# # Add labels and title
# plt.xlabel('Number of requests')
# plt.ylabel('Objective')
# # plt.title('Objective')

# time
plt.plot(x, ortools_time, label='ORTools', marker='o', linestyle='-')
# plt.plot(x, topsis_time, label='TOPSIS', marker='o', linestyle='-')
plt.plot(x, local_time, label='LocalSearch', marker='o', linestyle='-')

# Add labels and title
plt.xlabel('Number of requests')
plt.ylabel('Decision Making Time(s)')
# plt.title('Time')

# Add a legend
plt.legend()

# Display the chart
plt.show()