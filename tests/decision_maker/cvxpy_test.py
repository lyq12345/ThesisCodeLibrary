import cvxpy as cp
import numpy as np

# Define parameters
num_devices = 6
num_operators = 3
server_capacity = np.array([10, 8, 6, 6, 6, 6]) # xaiver, nano, pi, pi, pi, pi
operator_requirements = np.array([3, 5, 4]) # humantinyv3, humanyolov3, firetinyv3, fireyolov3
operator_accuracy = np.array([0.45, 0.68, 0.45])
processing_speed = np.array([[0.46, 0.58, 1.09, 1.09, 1.09, 1.09],
                           [3.86, 4.48, 7.21, 7.21, 7.21, 7.21],
                           [0.42, 0.55, 1.07, 1.07, 1.07, 1.07]])

transmission_rate = np.random.uniform(5, 20, size=(6, 6))

# Define decision variables (binary variables)
X = cp.Variable((num_operators, num_devices), boolean=True) # operator - device
Y = cp.Variable((num_devices, num_operators), boolean=True) # device(data source) - operator

# Define the objective function
transmission_delay = cp.sum(Y.T@cp.multiply(Y@X, transmission_rate), axis=1) # ?????????????
processing_delay = cp.sum(cp.multiply(X, processing_speed), axis=1)
delay = transmission_delay + processing_delay
accuracy = cp.multiply(cp.sum(X, axis=1), operator_accuracy.T)

utility = accuracy - cp.maximum(0, (delay - 10)/delay)

objective = cp.Maximize(cp.sum(utility))

# Define constraints
constraints = [
    cp.sum(X, axis=1) == 1, # each operator deployed only once
    operator_requirements@X <= server_capacity, # resource not exceeded
    cp.sum(Y) == cp.sum(X), # operator number consistent
    cp.sum(Y, axis=0) == 1, # an operator can serve one data source
]

# Create the optimization problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()

# Print the results
if problem.status == cp.OPTIMAL:
    print("Optimal deployment plan:")
    print(Y.value)
    print(X.value)
    print("Total cost:", utility.value)
else:
    print("Problem couldn't be solved.")
