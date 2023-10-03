import cvxpy as cp
import numpy as np

# Define parameters
num_devices = 3
num_operators = 4
server_capacity = np.array([10, 8, 6]) # xaiver, nano, pi
operator_requirements = np.array([3, 5, 3, 4]) #humantiny, humanyolo, firetiny, fireyolo
operator_accuracy = np.array([0.45, 0.68, 0.45, 0.68])
processing_speed = np.array([[0.46, 0.58, 1.09],
                           [3.86, 4.48, 7.21],
                           [0.42, 0.55, 1.07],
                           [2.62, 4.36, 7.08]])

transmission_rate = np.random.uniform(5, 20, size=(3, 3))

deployment_cost = np.array([[1, 2, 3],
                           [2, 1, 2],
                           [3, 2, 1],
                           [2, 3, 2]])

# Define decision variables (binary variables)
x = cp.Variable((num_operators, num_devices), boolean=True) # operator - device
y = cp.Variable((num_devices, num_operators), boolean=True) # device(data source) - operator

# Define the objective function (minimize cost)
cost = cp.sum(cp.multiply(deployment_cost, x))
delay = cp.sum(y.T@cp.multiply(y@x, transmission_rate), axis=1) + cp.sum(cp.multiply(x, processing_speed), axis=1)
accuracy = cp.multiply(cp.sum(x, axis=1), operator_accuracy.T)
objective = cp.Minimize(cost)

# Define constraints (server capacity and app requirements)
constraints = [
    cp.sum(x, axis=1) == 1,
    operator_requirements@x <= server_capacity
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
