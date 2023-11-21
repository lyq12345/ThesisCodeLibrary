# # Python3 program to create target string, starting from
# # random string using Genetic Algorithm
#
# import random
#
# # Number of individuals in each generation
# POPULATION_SIZE = 100
#
# # Valid genes
# GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP
# QRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''
#
# # Target string to be generated
# TARGET = "I love GeeksforGeeks"
#
#
# class Individual(object):
#     '''
#     Class representing individual in population
#     '''
#
#     def __init__(self, chromosome):
#         self.chromosome = chromosome
#         self.fitness = self.cal_fitness()
#
#     @classmethod
#     def mutated_genes(self):
#         '''
#         create random genes for mutation
#         '''
#         global GENES
#         gene = random.choice(GENES)
#         return gene
#
#     @classmethod
#     def create_gnome(self):
#         '''
#         create chromosome or string of genes
#         '''
#         global TARGET
#         gnome_len = len(TARGET)
#         return [self.mutated_genes() for _ in range(gnome_len)]
#
#     def mate(self, par2):
#         '''
#         Perform mating and produce new offspring
#         '''
#
#         # chromosome for offspring
#         child_chromosome = []
#         for gp1, gp2 in zip(self.chromosome, par2.chromosome):
#
#             # random probability
#             prob = random.random()
#
#             # if prob is less than 0.45, insert gene
#             # from parent 1
#             if prob < 0.45:
#                 child_chromosome.append(gp1)
#
#             # if prob is between 0.45 and 0.90, insert
#             # gene from parent 2
#             elif prob < 0.90:
#                 child_chromosome.append(gp2)
#
#             # otherwise insert random gene(mutate),
#             # for maintaining diversity
#             else:
#                 child_chromosome.append(self.mutated_genes())
#
#             # create new Individual(offspring) using
#         # generated chromosome for offspring
#         return Individual(child_chromosome)
#
#     def cal_fitness(self):
#         '''
#         Calculate fitness score, it is the number of
#         characters in string which differ from target
#         string.
#         '''
#         global TARGET
#         fitness = 0
#         for gs, gt in zip(self.chromosome, TARGET):
#             if gs != gt: fitness += 1
#         return fitness
#
#     # Driver code
#
#
# def main():
#     global POPULATION_SIZE
#
#     # current generation
#     generation = 1
#
#     found = False
#     population = []
#
#     # create initial population
#     for _ in range(POPULATION_SIZE):
#         gnome = Individual.create_gnome()
#         population.append(Individual(gnome))
#
#     while not found:
#
#         # sort the population in increasing order of fitness score
#         population = sorted(population, key=lambda x: x.fitness)
#
#         # if the individual having lowest fitness score ie.
#         # 0 then we know that we have reached to the target
#         # and break the loop
#         if population[0].fitness <= 0:
#             found = True
#             break
#
#         # Otherwise generate new offsprings for new generation
#         new_generation = []
#
#         # Perform Elitism, that mean 10% of fittest population
#         # goes to the next generation
#         s = int((10 * POPULATION_SIZE) / 100)
#         new_generation.extend(population[:s])
#
#         # From 50% of fittest population, Individuals
#         # will mate to produce offspring
#         s = int((90 * POPULATION_SIZE) / 100)
#         for _ in range(s):
#             parent1 = random.choice(population[:50])
#             parent2 = random.choice(population[:50])
#             child = parent1.mate(parent2)
#             new_generation.append(child)
#
#         population = new_generation
#
#         print("Generation: {}\tString: {}\tFitness: {}". \
#               format(generation,
#                      "".join(population[0].chromosome),
#                      population[0].fitness))
#
#         generation += 1
#
#     print("Generation: {}\tString: {}\tFitness: {}". \
#           format(generation,
#                  "".join(population[0].chromosome),
#                  population[0].fitness))
#
#
# if __name__ == '__main__':
#     main()

# -*- coding: utf-8 -*-
import pandas as pd
import random


def tournament_select(pops, popsize, fit, tournament_size):
    new_pops = []
    while len(new_pops) < len(pops):
        tournament_list = random.sample(range(0, popsize), tournament_size)
        tournament_fit = [fit[i] for i in tournament_list]
        # 转化为df方便索引
        tournament_df = pd.DataFrame([tournament_list, tournament_fit]).transpose().sort_values(by=1).reset_index(
            drop=True)
        new_pop = pops[int(tournament_df.iloc[0, 0])]
        new_pops.append(new_pop)

    return new_pops

def is_valid(gene):
    pass


def crossover(popsize, parent1_pops, parent2_pops, pc):
    child_pops = []
    for i in range(popsize):
        parent1 = parent1_pops[i]
        parent2 = parent2_pops[i]
        child = [-1 for i in range(len(parent1))]

        if random.random() >= pc:
            child = parent1.copy()  # 随机生成一个
            random.shuffle(child)

        else:
            # parent1
            start_pos = random.randint(0, len(parent1) - 1)
            end_pos = random.randint(0, len(parent1) - 1)
            if start_pos > end_pos: start_pos, end_pos = end_pos, start_pos
            child[start_pos:end_pos + 1] = parent1[start_pos:end_pos + 1].copy()
            # parent2 -> child
            list_index = list(range(end_pos + 1, len(parent2))) + list(range(0, start_pos))
            j = -1
            for i in list_index:
                for j in range(j + 1, len(parent2)):
                    if parent2[j] not in child:
                        child[i] = parent2[j]
                        break

        child_pops.append(child)
    return child_pops


def mutate(pops, pm):
    pops_after_mutate = []
    mutate_time = 0
    for i in range(len(pops)):
        pop = pops[i].copy()
        if random.random() < pm:
            while mutate_time < 3:
                mut_pos1 = random.randint(0, len(pop) - 1)
                mut_pos2 = random.randint(0, len(pop) - 1)
                if mut_pos1 != mut_pos2: pop[mut_pos1], pop[mut_pos2] = pop[mut_pos2], pop[mut_pos1]
                mutate_time += 1
        pops_after_mutate.append(pop)

    return pops_after_mutate


def package_calFitness(cargo_df, pop, max_v, max_m):
    '''
    输入：cargo_df-货物信息,pop-个体,max_v-箱子容积,max_m-箱子在载重
    输出：适应度-fit，boxes-解码后的个体
    '''
    box_num = 0  # 装满的箱子数
    v_sum, m_sum = 0, 0
    v, m = 0, 0
    boxes, box = [], []
    for j in pop:
        v_j = cargo_df[cargo_df['货物序号'] == j]['体积'].iloc[0]
        m_j = cargo_df[cargo_df['货物序号'] == j]['重量'].iloc[0]
        v += v_j
        m += m_j
        if (v_sum + v_j <= max_v) and (m_sum + m_j <= max_m):
            box.append(j)
            v_sum = v_sum + v_j
            m_sum = m_sum + m_j
        else:
            boxes.append(sorted(box))
            box_num += 1
            box = []
            box.append(j)
            v_sum = v_j
            m_sum = m_j

    # 最后一个箱子
    boxes.append(sorted(box))

    # fit=总体积/(已使用车辆容积和+最后一辆车使用容积)*总重量/(已使用车辆载重和+最后一辆车使用载重)
    fit = (v / (v_sum + box_num * max_v)) * (m / (m_sum + box_num * max_m))
    return round(fit, 4), boxes


def package_GA(cargo_df, generations, popsize, tournament_size, pc, pm, max_v, max_m):
    # 初始化种群
    cargo_list = list(cargo_df['货物序号'])
    pops = [random.sample(cargo_list, len(cargo_list)) for i in range(popsize)]  # 种群初始化
    fit, boxes = [-1] * popsize, [-1] * popsize

    for i in range(popsize):
        fit[i], boxes[i] = package_calFitness(cargo_df, pops[i], max_v, max_m)

    best_fit = max(fit)
    best_pop = pops[fit.index(max(fit))]
    best_box = boxes[fit.index(max(fit))]

    if best_fit == 1: return best_pop  # 1说明除最后一辆车都装满，已是最优解

    iter = 0  # 迭代计数
    while iter < generations:
        pops1 = tournament_select(pops, popsize, fit, tournament_size)
        pops2 = tournament_select(pops, popsize, fit, tournament_size)
        new_pops = crossover(popsize, pops1, pops2, pc)
        new_pops = mutate(new_pops, pm)
        iter += 1
        new_fit, new_boxes = [-1] * popsize, [-1] * popsize  # 初始化
        for i in range(popsize):
            new_fit[i], new_boxes[i] = package_calFitness(cargo_df, new_pops[i], max_v, max_m)  # 计算适应度
        for i in range(len(pops)):
            if fit[i] < new_fit[i]:
                pops[i] = new_pops[i]
                fit[i] = new_fit[i]
                boxes[i] = new_boxes[i]

        if best_fit < max(fit):  # 保留历史最优
            best_fit = max(fit)
            best_pop = pops[fit.index(max(fit))]
            best_box = boxes[fit.index(max(fit))]

        print("第", iter, "代适应度最优值：", best_fit)
    return best_pop, best_fit, best_box


if __name__ == '__main__':
    # 数据
    num = list(range(100))  # 货物编号
    volumns = [1, 6, 7, 8, 1, 2, 3, 1, 8, 8, 10, 1, 9, 3, 4, 3, 5, 7, 4, 6, 5, 5, 9, 5, 6, 3, 9, 9, 6, 3, 4, 2, 1, 3, 5,
               9, 6, 6, 8, 5, 6, 2, 7, 9, 5, 1, 7, 5, 10, 6,
               4, 6, 9, 7, 2, 4, 3, 7, 5, 4, 5, 10, 2, 1, 4, 10, 9, 6, 10, 10, 10, 2, 10, 2, 4, 6, 4, 1, 7, 6, 1, 10, 1,
               3, 4, 1, 7, 3, 6, 5, 3, 10, 6, 8, 1, 6, 4, 4, 10, 3]  # 体积
    weight = [3, 5, 3, 8, 10, 4, 7, 2, 10, 1, 9, 2, 1, 9, 7, 1, 7, 1, 4, 2, 5, 9, 1, 6, 1, 4, 2, 1, 2, 1, 5, 5, 6, 8, 3,
              6, 7, 4, 9, 7, 7, 4, 8, 3, 9, 4, 1, 1, 9, 5, 8,
              4, 10, 3, 5, 1, 7, 8, 8, 2, 8, 7, 1, 10, 3, 3, 8, 2, 4, 6, 8, 3, 5, 8, 10, 5, 7, 5, 7, 1, 9, 1, 5, 9, 9,
              2, 10, 2, 9, 3, 7, 10, 5, 1, 2, 1, 9, 8, 6, 9]  # 重量
    cargo_df = pd.DataFrame({'货物序号': num, "体积": volumns, "重量": weight})

    M, V = 100, 100  # 箱子载重容积

    # GA参数
    generations = 50
    popsize = 40
    tournament_size = 4
    pc = 0.9
    pm = 0.1

    pop, fit, box = package_GA(cargo_df, generations, popsize, tournament_size, pc, pm, V, M)
    print("最优解：", box)


