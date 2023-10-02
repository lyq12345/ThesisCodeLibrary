import cvxpy as cp
import numpy as np
# Define parameters
num_servers = 3
num_apps = 4
server_capacity = np.array([10, 8, 6])
app_requirements = np.array([4, 3, 5, 2])
deployment_cost = np.array([[1, 2, 3],
                           [2, 1, 2],
                           [3, 2, 1],
                           [2, 3, 2]])

# Define decision variables (binary variables)
x = cp.Variable((num_apps, num_servers), boolean=True)

# Define the objective function (minimize cost)
cost = cp.sum(cp.multiply(deployment_cost, x))
objective = cp.Minimize(cost)

# Define constraints (server capacity and app requirements)
constraints = [
    cp.sum(x, axis=1) == 1,
    app_requirements@x <= server_capacity
]

# Create the optimization problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()

# Print the results
if problem.status == cp.OPTIMAL:
    print("Optimal deployment plan:")
    print(x.value)
    print("Total cost:", cost.value)
else:
    print("Problem couldn't be solved.")
