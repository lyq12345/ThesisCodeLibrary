from ortools.linear_solver import pywraplp

# Create the MIP solver
solver = pywraplp.Solver.CreateSolver('SCIP')

# Create binary decision variables x and y
x = solver.IntVar(0, 1, 'x')
y = solver.IntVar(0, 1, 'y')

# Create a binary decision variable z to represent max(x, y)
z = solver.IntVar(0, 1, 'z')

# Add constraints to represent max(x, y)
solver.Add(z >= x)
solver.Add(z >= y)

# Set the objective function to maximize z (max(x, y))
solver.Maximize(z)

# Solve the problem
status = solver.Solve()

# Check the solver's result status
if status == pywraplp.Solver.OPTIMAL:
    print("Optimal Solution Found:")
    print("x =", x.solution_value())
    print("y =", y.solution_value())
    print("max(x, y) =", z.solution_value())
else:
    print("No optimal solution found.")
