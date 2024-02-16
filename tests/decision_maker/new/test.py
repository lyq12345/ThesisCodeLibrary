from ortools.sat.python import cp_model


def minimize_communication_cost(job_sequence, communication_costs):
    num_jobs = len(job_sequence)
    num_devices = len(communication_costs)

    model = cp_model.CpModel()

    # Variables
    assignments = {}
    for i in range(num_jobs):
        for j in range(num_devices):
            assignments[(i, j)] = model.NewBoolVar(f'job_{i}_on_device_{j}')

    # Constraints
    for i in range(num_jobs):
        model.Add(
            sum(assignments[(i, j)] for j in range(num_devices)) == 1)  # Each job is assigned to exactly one device

    for j in range(num_devices):
        for t in range(num_jobs - 1):
            model.Add(assignments[(job_sequence[t], j)] == assignments[
                (job_sequence[t + 1], j)])  # Jobs must maintain their sequence

    # Objective
    total_communication_cost = sum(
        communication_costs[j] * assignments[(i, j)] for i in range(num_jobs) for j in range(num_devices))
    model.Minimize(total_communication_cost)

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        assignment_solution = {}
        for i in range(num_jobs):
            for j in range(num_devices):
                if solver.Value(assignments[(i, j)]) == 1:
                    assignment_solution[i] = j
        min_communication_cost = solver.ObjectiveValue()
        return assignment_solution, min_communication_cost
    else:
        return None, None


# Example usage
job_sequence = [0, 1, 2, 3]  # Example job sequence
communication_costs = [[1, 2], [3, 1], [2, 4], [1, 3]]  # Example communication costs between devices

assignment_solution, min_communication_cost = minimize_communication_cost(job_sequence, communication_costs)
if assignment_solution is not None:
    print("Optimal Assignment:")
    for job, device in assignment_solution.items():
        print(f"Job {job} assigned to Device {device}")
    print(f"Minimum Communication Cost: {min_communication_cost}")
else:
    print("No solution found.")