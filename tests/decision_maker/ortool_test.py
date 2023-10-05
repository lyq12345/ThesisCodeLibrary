from ortools.linear_solver import pywraplp

# Dimensions
D = 6
O = 3

def create_operator_model():
    data = {}
    data["operator_accuracies"] = [0.45, 0.68, 0.45, 0.68]
    data["resource_requirements"] = [3, 5, 4, 5] # humantinyv3, humanyolov3, firetinyv3, fireyolov3
    data["processing_speed"] = [[0, 0.58, 1.09, 1.09, 1.09, 1.09],
                                [3.86, 4.48, 7.21, 7.21, 7.21, 7.21],
                                [0.42, 0.55, 1.07, 1.07, 1.07, 1.07]]
    return data

def inverse(x):
    if x == 0:
        return 0
    else:
        return 1/x

def create_device_model():
    """Stores the data for the problem."""
    data = {}
    data["transmission_speed"] = [[ 0,  6.43163439, 19.35637777, 15.13368045, 13.53071896, 11.69206187],
                                 [ 6.43163439, 0, 17.22729904, 18.4682481,   5.52004585,  8.88291174],
                                 [ 19.35637777, 17.22729904,   0, 16.86937148, 10.66980807, 13.09753162],
                                 [15.13368045, 18.4682481, 10.66980807,  0, 14.25380426, 14.14231382],
                                 [ 13.53071896, 5.52004585, 13.09753162, 14.25380426, 0,   6.70848206],
                                 [11.69206187,  8.88291174,  9.19870435,  14.14231382,  6.70848206,  0]]
    data["resource_capability"] = [10, 8, 6, 6, 6, 6]

    return data

# Create the MIP solver
solver = pywraplp.Solver.CreateSolver('SCIP')

operator_data = create_operator_model()
device_data = create_device_model()

# Create binary decision variables x_ij for the binary matrix x
x = {} #x_jk - operator j on device k
for j in range(O):
    for k in range(D):
        x[j, k] = solver.IntVar(0, 1, f'x_{j}_{k}')

# Create binary decision variables y_jk for the binary vector y
y = {} #y_ij - data source i to operator j
for i in range(D):
    for j in range(O):
        y[i, j] = solver.IntVar(0, 1, f'y_{i}_{j}')

z = {}
for i in range(D):
    for j in range(O):
        for k in range(D):
            z[i, j, k] = solver.IntVar(0, 1, f'z_{i}_{j}_{k}')

# Add constraints to represent x_ij * y_jk as binary variables
for i in range(D):
    for j in range(O):
        for k in range(D):
            solver.Add(z[i, j, k] <= x[j, k])
            solver.Add(z[i, j, k] <= y[i, j])
            solver.Add(z[i, j, k] >= x[j, k] + y[i, j] - 1)

# Each operator is assigned to at most 1 device.
for j in range(O):
    solver.Add(solver.Sum([x[j, k] for k in range(D)]) <= 1)

# Each data source transmit to at most one operator
for i in range(D):
    solver.Add(solver.Sum([y[i, j] for j in range(O)]) <= 1)

# operator requirement sum in each device should not exceed its capacity
for k in range(D):
    solver.Add(solver.Sum([x[j, k]*operator_data["resource_requirements"][j] for j in range(O)]) <= device_data["resource_capability"][k])


utilities = []
for i in range(D):
    for j in range(O):
        for k in range(D):
            utilities.append(z[i, j, k]*operator_data["operator_accuracies"][j]
                             - z[i, j, k]*max((1 - 10*inverse((device_data["transmission_speed"][i][k] + operator_data["processing_speed"][j][k]))), 0))

# transmission_times = {}
# for i in range(D):
#     for j in range(O):
#         for k in range(D):
#             transmission_times[i, j, k] = z[i, j, k]*device_data["transmission_speed"][i][k]


# for i in range(D):
#     for j in range(O):
#         for k in range(D):
#             utilities.append(z[i, j, k]*accuracy[i, j, k]-z[i, j, k]*transmission_times[i, j, k])

solver.Maximize(solver.Sum(utilities))

status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print(f"The maximized utility sum = {solver.Objective().Value()}\n")
    # Print the values of x_ij
    print("Values of decision variable Y:")
    for i in range(D):
        for j in range(O):
            if y[i, j].solution_value() != 0:
                # print(f"y_{i}_{j} =", y[i, j].solution_value())
                print(f"data source {i} transmits to operator {j}")

    # Print the values of y_jk
    print("Values of decision variable X:")
    for j in range(O):
        for k in range(D):
            if x[j, k].solution_value() != 0:
                print(f"operator {j} is deployed on device {k}")
            # print(f"x_{j}_{k} =", x[j, k].solution_value())

    # print("Values of z_ijk:")
    # for i in range(D):
    #     for j in range(O):
    #         for k in range(D):
    #             if z[i, j, k].solution_value() != 0:
    #                 print(f"z_{i}_{j}_{k} =", z[i, j, k].solution_value())
else:
    print("No optimal solution found.")


# # Create binary decision variable z to model max(0, sum(x_ij * y_jk))
# z = solver.IntVar(0, 100000, 'z')
#
# # Additional binary decision variables to model x_ij * y_jk
# xy = {}
# for i in range(D):
#     for j in range(O):
#         for k in range(D):
#             xy[i, j, k] = solver.IntVar(0, 1, f'xy_{i}_{j}_{k}')
#
# # Add constraints to represent x_ij * y_jk as binary variables
# for i in range(D):
#     for j in range(O):
#         for k in range(D):
#             solver.Add(xy[i, j, k] <= x[i, j])
#             solver.Add(xy[i, j, k] <= y[j, k])
#             solver.Add(xy[i, j, k] >= x[i, j] + y[j, k] - 1)
#
# # Add constraint to ensure z is 1 if and only if max(0, sum(x_ij * y_jk)) > 0
# # solver.Add(z <= max( 200, sum(2*xy[i, j, k] for i in range(I) for j in range(J) for k in range(K))))
#
# # Add constraint to ensure z is equal to max(100, sum(x_ij * y_jk))
# # solver.Add(z >= 100)
# # solver.Add(z >= sum(xy[i, j, k] for i in range(I) for j in range(J) for k in range(K)))
# # max(100,sum(xy[i, j, k]) for all i, j, k)
# M = 10  # A large constant
# for i in range(D):
#     for j in range(O):
#         for k in range(D):
#             solver.Add(z >= 10 - M * (1 - xy[i, j, k]))
#             solver.Add(
#                 z <= sum(xy[i, j, k] for i in range(D) for j in range(O) for k in range(D)) - M * (xy[i, j, k] - 1))

# Add constraint sum_j x_ij <= 1 for each i
# for i in range(D):
#     solver.Add(sum(x[i, j] for j in range(O)) <= 1)
#
# # Maximize z
# solver.Maximize(z)
# # Solve the problem
# status = solver.Solve()
# # Print the objective function
# print("Objective Function:")
# print("Maximize z")  # Changed from "Maximize z"
# # Print the constraints
# print("z <= max(10,sum(xy[i, j, k]) for all i, j, k)")
#
# print("\nConstraints:")
# for i in range(D):
#     print(f"sum_j x_{i}_j <= 1")
#
# # Check the solver's result status
# if status == pywraplp.Solver.OPTIMAL:
#     print("Optimal Solution Found:")
#     print("z =", z.solution_value())
#
#     # Print the values of x_ij
#     print("Values of x_ij:")
#     for i in range(D):
#         for j in range(O):
#             print(f"x_{i}_{j} =", x[i, j].solution_value())
#
#     # Print the values of y_jk
#     print("Values of y_jk:")
#     for j in range(O):
#         for k in range(D):
#             print(f"y_{j}_{k} =", y[j, k].solution_value())
# else:
#     print("No optimal solution found.")