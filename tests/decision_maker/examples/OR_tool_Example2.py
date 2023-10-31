from ortools.linear_solver import pywraplp

# Create the MIP solver
solver = pywraplp.Solver.CreateSolver('SCIP')

# Create binary decision variables x and y
x = solver.IntVar(0, 1, 'x')
y = solver.IntVar(0, 1, 'y')

# x*y
# Define the objective function to maximize max(0, x*y)
# We can model max(0, x*y) as a binary variable z, where z = 1 if x*y > 0, and z = 0 otherwise.
z = solver.IntVar(0, 1, 'z')

# Add constraints to make z = x*y
solver.Add(z >= x + y - 1)
solver.Add(z <= x)
solver.Add(z <= y)

# Maximize z
solver.Maximize(z*10)

# # Solve the problem
# solver.Solve()
# Check the status of the solver
status = solver.Solve()
# Print the results
if status == pywraplp.Solver.OPTIMAL:
    print("Optimal Solution Found:")
    print("x =", x.solution_value())
    print("y =", y.solution_value())
    print("x*y =", z.solution_value())
else:
    print("No optimal solution found.")
