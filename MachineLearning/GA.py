import random
import numpy as np
import matplotlib.pyplot as plt

n = 7
m = 10
generations = 20

def generate_indv():
    return np.random.randint(2, size=n)
def compute_fitness(indv):
    return indv.sum()
def sort_population(population):
    for i in range(0,m):
        for j in range(i+1, m):
            if(compute_fitness(population[i]) > compute_fitness(population[j])):
                temp = population[i]
                population[i] = population[j]
                population[j] = temp
    return population

def selection(sorted_population):
    index_1 = random.randint(0, m-1)
    while(True):
        index_2 = random.randint(0,m-1)
        if index_2 != index_1:
            break
    if index_2 > index_1:
        return sorted_population[index_2]
    return sorted_population[index_1]

def cross_over(indv_1, indv_2, cross_over_rate = 0.8):
    mask = np.random.random(n)
    mask = mask < cross_over_rate
    indv_c1 = indv_1.copy()
    indv_c2 = indv_2.copy()
    indv_c1[mask] = indv_2[mask]
    indv_c2[mask] = indv_1[mask]
    return indv_c1, indv_c2

def mutation(indv, mutation_rate= 0.05):

    mask = np.random.random(n)
    mask = mask < mutation_rate
    indv_m = indv.copy()
    indv_m[mask] = random.randint(0,1)
    return indv_m

fitnesses = []
def GA(population, elim = 2):
    population = sort_population(population)
    new_population = []
    fitnesses.append(compute_fitness(population[-1]))
    while len(new_population) < m - elim:
        indv_s1 = selection(population)
        indv_s2 = selection(population)

        indv_c1, indv_c2 = cross_over(indv_s1, indv_s2)

        indv_m1 = mutation(indv_c1)
        indv_m2 = mutation(indv_c2)

        new_population.append(indv_m1)
        new_population.append(indv_m2)
    for indv in population[m - elim:]:
        new_population.append(indv)

    return new_population 



population = []
for i in range(m):
    population.append(generate_indv())

for i in range(generations):
    print(population[-1])
    population = GA(population)


plt.plot(fitnesses)
plt.show()

