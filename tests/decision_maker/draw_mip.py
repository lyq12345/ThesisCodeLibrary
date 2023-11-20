import matplotlib.pyplot as plt
import pandas as pd
# Sample data
x1 = [1, 2, 3, 4, 5, 6]
# y = [0.0109, 0.3787, 0.7894, 12.4716, 90.7383, 194.47]
df = pd.read_csv('results/evaluation_4.csv')
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
ortools_time = [0.0109, 0.3787, 0.7894, 12.4716, 90.7383, 194.47]
ortools_objective = df[df['algorithm'] == 'MIP']['Normalized objective'].tolist()

topsis_time = df[df['algorithm'] == 'TOPSIS']['time'].tolist()
topsis_objective = df[df['algorithm'] == 'TOPSIS']['Normalized objective'].tolist()
local_time = df[df['algorithm'] == 'LocalSearch']['time'].tolist()
local_objective = df[df['algorithm'] == 'LocalSearch']['Normalized objective'].tolist()

print(topsis_time)
print(topsis_objective)

# Create the line chart
plt.plot(x1, ortools_objective, label='ORTools', marker='o', linestyle='-')
plt.plot(x, topsis_objective, label='TOPSIS', marker='o', linestyle='-')
plt.plot(x, local_objective, label='LocalSearch', marker='o', linestyle='-')

# Add labels and title
plt.xlabel('Number of requests')
plt.ylabel('Normalized Objective')
plt.title('Objective')

# Add a legend
plt.legend()

# Display the chart
plt.show()