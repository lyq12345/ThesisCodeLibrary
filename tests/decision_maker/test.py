import random

# Define the problem - a list of customer locations and a list of potential warehouse locations
customers = [(1, 2), (3, 4), (5, 6), (7, 8)]
warehouses = [(2, 3), (6, 7)]

# Objective function - minimize the total transportation cost
def objective_function(customers, warehouses):
    total_cost = 0
    for c in customers:
        min_dist = min([abs(c[0]-w[0]) + abs(c[1]-w[1]) for w in warehouses])
        total_cost += min_dist
    return total_cost

# Randomly initialize the initial solution
def random_initial_solution(customers, num_warehouses):
    return random.sample(customers, num_warehouses)

# Local search algorithm
def local_search(customers, num_warehouses, max_iterations):
    current_solution = random_initial_solution(customers, num_warehouses)
    current_cost = objective_function(current_solution, warehouses)

    for _ in range(max_iterations):
        # Generate neighboring solutions by moving one customer to a different warehouse
        neighbors = []
        for i in range(num_warehouses):
            for j in range(len(customers)):
                if customers[j] not in current_solution:
                    neighbor_solution = current_solution[:i] + [customers[j]] + current_solution[i+1:]
                    neighbors.append(neighbor_solution)

        # Evaluate the neighbors and select the best one
        best_neighbor = min(neighbors, key=lambda solution: objective_function(solution, warehouses))

        # If the best neighbor is better than the current solution, update the solution
        if objective_function(best_neighbor, warehouses) < current_cost:
            current_solution = best_neighbor
            current_cost = objective_function(current_solution, warehouses)

    return current_solution, current_cost

# Run the local search
best_solution, best_cost = local_search(customers, len(warehouses), max_iterations=1000)

print("Best solution:", best_solution)
print("Best cost:", best_cost)
