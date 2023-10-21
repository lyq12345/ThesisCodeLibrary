import random
import sys


# function to generate new population
def Generate_Initial_Population(problem_size, population_size) -> list:
    """Generate list of random chromosome
    Parameters
    ----------
    problem_size : int
        size of the problem i.e no of location/facilites
    population_size : int
        number of data we want in our list
    Returns
    -------
    list
        return list of chromosome
    """

    population = []

    for i in range(population_size):
        # create list with size == problem size and random values ranging from 0 to problem_size
        x = random.sample(range(problem_size), problem_size)

        # add list x to population. The item in the second index (0) is the fitness score we'll use later
        population.append([x, 0])

    return population

# cost function to get cost of each chromosome
def Cost_Function(population, distances, flows) -> list:
    """Gets the fitness score for each data in a population using the formula minϕ∈Sn ∑ni=1 ∑nj=1 fij⋅dϕ(i)ϕ(j)
    Parameters
    ----------
    population : list
        list of chromosomes
    distances : list
        list of distance mapping for each data in the population
    flows : list
        list of flow mappings for each data in the population
    Returns
    -------
    list
        list of chromosomes with updated fitness score
    """
    for chromosome in population:

        cost = 0

        searched_list = []

        for j in chromosome[0]:
            for k in chromosome[0]:

                # since problem is a one-to-one type, mapping (1,2) == (2,1).
                if (k, j) in searched_list or (j, k) in searched_list: continue

                # cost function = cost + flow(f1, f2) * distance(d1, d2) for every f1, f2, d1, d2.
                cost += Get_Distance_Or_Flow(j,k, distances) * Get_Distance_Or_Flow(chromosome[0][j], chromosome[0][k], flows)

                # append mapping to searched list to save time.
                searched_list.append((j, k))


        chromosome[1] = cost


    return population

# selection function
def Selection_Function(population) -> list:
    """Select data with the minimum fitness score from a population using the tournament selection technique
    Parameters
    ----------
    population : list
        list of chromosome
    Returns
    -------
    list
        data with the minimum fitness score from the population
    """

    # return random list size population_size/5 from population
    random_k_list = random.sample(population, int(len(population)/5))

    # sort random list using their fitness score
    random_k_list.sort(key = lambda x: x[1])

    # return first element after sort
    return random_k_list[0]


# crossover function
def Crossover_Function(data1, data2):
    """Perform modified version of uniform crossover on 2 chromosomes
    Parameters
    ----------
    data1 : list
         list containing chromosome and fitness score
    data2 : list
         list containing chromosome and fitness score
    Returns
    -------
    list
        return list containing 2 data with modified chromosome
    """

    # for this function, I modified the uniform crossover function to take care of duplicates after crossover.

    data1[1] = 0
    data2[1] = 0
    chromosome1 = list.copy(data1[0])
    chromosome2 = list.copy(data2[0])

    # for each index in both chromosomes, use a coin toss to determine which index is crossed over
    for i in range(len(chromosome1)):

        cointoss = random.randrange(2)
        if cointoss == 0:
            chromosome1[i], chromosome2[i] = chromosome2[i], chromosome1[i]

    # find duplicates after crossing over
    dupes_in_ch1 = list(duplicates(chromosome1))
    dupes_in_ch2 = list(duplicates(chromosome2))

    # handle duplicates if any are found
    for i in dupes_in_ch1:
        if i in chromosome1: chromosome1.remove(i)
        chromosome2.append(i)

    for i in dupes_in_ch2:
        if i in chromosome2: chromosome2.remove(i)
        chromosome1.append(i)

    # replaced the modified chromosomes in the data
    data1[0] = chromosome1
    data2[0] = chromosome2

    return [data1, data2]

# It gets 2 random indexes in the chromosome and switches their value.

def Mutation_Function(data) -> list:
    """Modifies the chromosome in a data
    Parameters
    ----------
    gene : list
        specific data that needs modification
    Returns
    -------
    list
        returns the Modified data
    """

    chromosome = data[0]

    randomNum1 = random.randint(0, len(chromosome) - 1)
    randomNum2 = random.randint(0, len(chromosome) - 1)

    # exchange values at 2 random indexes
    chromosome[randomNum1], chromosome[randomNum2] = chromosome[randomNum2], chromosome[randomNum1]

    return data

def GeneticAlgorithm(problem_size, population_size, distances, flows, number_of_iterations):
    # generate initial population
    population = Generate_Initial_Population(problem_size, population_size)

    solution = int(sys.maxsize)
    next_generation = []
    n = 0

    while n < number_of_iterations:

        # get cost function for each data in population
        population = Cost_Function(population=population, distances=distances, flows=flows)

        # sort population according to fitness score
        population.sort(key=lambda x: x[1])

        # get fittest data
        fittest_data = list.copy(population[0])

        # check for the fittest data and print it out
        if fittest_data[1] < solution:
            result = list.copy(fittest_data)
            solution = fittest_data[1]
            print("\nSolution for iteration - " + str(n))
            print(result)

        while len(next_generation) < len(population):
            # use selection fucntion to get 2 fit chromosomes
            data1 = Selection_Function(population)
            data2 = Selection_Function(population)

            # crossover the 2 chromosome
            crossed_over_data = Crossover_Function(data1, data2)

            # mutate both chromosomes
            offspring1 = Mutation_Function(crossed_over_data[0])
            offspring2 = Mutation_Function(crossed_over_data[1])

            # add offsprings to next generation
            next_generation.append(offspring1)
            next_generation.append(offspring2)

        # repeat iteration with new generation
        population = next_generation
        next_generation = []
        n += 1

    # print final result
    print("Final solution after " + str(n) + " iterations = ")
    print(result)

    return result


# helper function to help generate random distance and flow values
distances = Generate_Distance_Or_Flow(6, 20)
flows = Generate_Distance_Or_Flow(6, 4)

# Test run an exmaple with input size of 6, population size of 30 and to perform 1000 iterations
GeneticAlgorithm(6, 30, distances, flows, 1000)