from ortools.linear_solver import pywraplp

# Dimensions
I = 4
J = 5
K = 3

# Create the MIP solver
solver = pywraplp.Solver.CreateSolver('SCIP')

# Create binary decision variables x_ij for the binary matrix x
x = {}
for i in range(I):
    for j in range(J):
        x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')

# Create binary decision variables y_jk for the binary vector y
y = {}
for j in range(J):
    for k in range(K):
        y[j, k] = solver.IntVar(0, 1, f'y_{j}_{k}')

# Create binary decision variable z to model max(0, sum(x_ij * y_jk))
z = solver.IntVar(0, 100000, 'z')

# Additional binary decision variables to model x_ij * y_jk
xy = {}
for i in range(I):
    for j in range(J):
        for k in range(K):
            xy[i, j, k] = solver.IntVar(0, 1, f'xy_{i}_{j}_{k}')

# Add constraints to represent x_ij * y_jk as binary variables
for i in range(I):
    for j in range(J):
        for k in range(K):
            solver.Add(xy[i, j, k] <= x[i, j])
            solver.Add(xy[i, j, k] <= y[j, k])
            solver.Add(xy[i, j, k] >= x[i, j] + y[j, k] - 1)

# Add constraint to ensure z is 1 if and only if max(0, sum(x_ij * y_jk)) > 0
#solver.Add(z <= max( 200, sum(2*xy[i, j, k] for i in range(I) for j in range(J) for k in range(K))))

# Add constraint to ensure z is equal to max(100, sum(x_ij * y_jk))
# solver.Add(z >= 100)
# solver.Add(z >= sum(xy[i, j, k] for i in range(I) for j in range(J) for k in range(K)))
# max(100,sum(xy[i, j, k]) for all i, j, k)
M = 10  # A large constant
for i in range(I):
    for j in range(J):
        for k in range(K):
            solver.Add(z >= 10 - M * (1 - xy[i, j, k]))
            solver.Add(z <= sum(xy[i, j, k] for i in range(I) for j in range(J) for k in range(K)) - M * (xy[i, j, k] - 1))

# Add constraint sum_j x_ij <= 1 for each i
for i in range(I):
    solver.Add(sum(x[i, j] for j in range(J)) <= 1)

# Maximize z
solver.Maximize(z)
# Solve the problem
status = solver.Solve()
# Print the objective function
print("Objective Function:")
print("Maximize z")  # Changed from "Maximize z"
# Print the constraints
print("z <= max(10,sum(xy[i, j, k]) for all i, j, k)")

print("\nConstraints:")
for i in range(I):
    print(f"sum_j x_{i}_j <= 1")

# Check the solver's result status
if status == pywraplp.Solver.OPTIMAL:
    print("Optimal Solution Found:")
    print("z =", z.solution_value())
    
    # Print the values of x_ij
    print("Values of x_ij:")
    for i in range(I):
        for j in range(J):
            print(f"x_{i}_{j} =", x[i, j].solution_value())
    
    # Print the values of y_jk
    print("Values of y_jk:")
    for j in range(J):
        for k in range(K):
            print(f"y_{j}_{k} =", y[j, k].solution_value())
else:
    print("No optimal solution found.")