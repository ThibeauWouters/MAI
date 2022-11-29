import random
import numpy as np
import time
import matplotlib.pyplot as plt


class KnapsackProblem:
    """Initialize the knapsack problem"""

    def __init__(self, number):
        self.numberOfObjects = number
        self.objectValues = 2 ** (np.random.normal(0, 1, self.numberOfObjects))
        self.objectWeights = 2 ** (np.random.normal(0, 1, self.numberOfObjects))
        self.capacity = 0.25 * np.sum(self.objectWeights)
        self.populationSize = 50
        self.offspringSize = self.populationSize
        self.population = [list(np.random.permutation(self.numberOfObjects)) for i in range(self.populationSize)]
        self.averageFitness = 0
        self.populationValues = [0] * self.populationSize
        self.populationWeights = [0] * self.populationSize
        self.set_fitness_population()
        self.offspring = [[0 for i in range(self.numberOfObjects)] for j in range(self.offspringSize)]
        self.allIndividuals = [[0 for i in range(self.numberOfObjects)] for j in range(self.offspringSize + self.populationSize)]
        self.numberOfIterations = 100
        self.k = 5
        self.alpha = 0.05
        self.beta = 0.50 # probability to choose object in child during crossover in symmetric difference
        print("Welcome to KP. We have " + str(self.numberOfObjects) + " objects.")
        print("Their values are: ", self.objectValues)
        print("Their weights are: ", self.objectWeights)
        print("Capacity is: ", self.capacity)
        #print("Population is: ", self.population)

    def estimate_optimal_fitness(self):
        """Use value/weight ratio as heuristic to estimate optimal fitness"""

        # Get best order according to value/weight ratio
        value_to_weight = []
        for i in range(self.numberOfObjects):
            value_to_weight.append(self.objectValues[i]/self.objectWeights[i])
        sorting_indices = list(np.argsort(value_to_weight))
        sorting_indices.reverse()

        # Fill knapsack according to this
        estimate = self.fitness(sorting_indices,return_weight=False)
        print("We estimate that the optimal fitness is around ", estimate)

        return estimate


    def optimize(self):
        """The main evolutionary algorithm loop"""
        mean_fitness_list = []
        max_fitness_list = []

        for i in range(self.numberOfIterations):
            start_time = time.time()
            self.getOffspring()
            self.mutation()
            self.joinPopulations()
            self.elimination()
            #print("The current population is ", self.population)
            self.set_fitness_population()
            iteration_time = time.time() - start_time

            # Save and report on progress:
            mean_fitness = np.mean(self.populationValues)
            max_fitness = max(self.populationValues)

            print("Iteration %d with mean fitness %0.3f, best fitness %0.3f and time %0.2f"
                  % (i+1, mean_fitness, max_fitness, iteration_time))
            mean_fitness_list.append(mean_fitness)
            max_fitness_list.append(max_fitness)

        # At the end of optimization loop, make a plot
        estimate = self.estimate_optimal_fitness()
        plt.figure(figsize = (12, 10))
        plt.plot(mean_fitness_list, '-o', label='Mean', color='red')
        plt.plot(max_fitness_list, '-o', label='Max', color='blue')
        plt.axhline(estimate, linestyle='--', label='Heuristic', color="black")
        plt.grid()
        plt.legend()
        plt.xlabel('Iteration number')
        plt.ylabel('Fitness')
        plt.title('Knapsack problem for %d items' % self.numberOfObjects)
        plt.show()

    def selection(self):
        """Selects a single individual with k-tournament"""

        sampled_indices = random.sample(list(range(self.populationSize)), self.k)
        sampled_fitness = [self.populationValues[index] for index in sampled_indices]
        best = np.argmin(sampled_fitness)
        best_index = sampled_indices[best]
        best_individual = self.population[best_index]

        return best_individual

    def crossover(self, parent1, parent2):
        """TO DO: Performs crossover after selection"""

        # Base crossover on knapsacks of parents
        first_knapsack = self.inKnapsack(parent1)
        second_knapsack = self.inKnapsack(parent2)
        #print("first knapsack ", first_knapsack)
        #print("second knapsack ", second_knapsack)

        # Get what is certainly inside the other knapsacks as list
        common_knapsack = first_knapsack.intersection(second_knapsack)
        numberOfCommon = len(common_knapsack)
        #print("common knapsack ", common_knapsack)

        # If they fully overlap, we are done already
        if len(common_knapsack) == self.numberOfObjects:
            return list(common_knapsack)

        # If not, add remaining objects: first, get symmetric difference between the knapsacks
        symmetric_difference = first_knapsack.symmetric_difference(second_knapsack)

        # Add these items with 50% probability
        child = common_knapsack
        for item in symmetric_difference:
            dice_throw = np.random.uniform()
            if dice_throw <= self.beta:
                child.add(item)

        # See which items are still remaining
        remainder = set(range(self.numberOfObjects)).difference(child)

        # Combine them both
        new_individual = list(child.union(remainder))

        # Randomize the order of the offspring & remainder part of this child separately
        new_individual[0:numberOfCommon-1] = list(np.random.permutation(new_individual[0:numberOfCommon-1]))
        new_individual[numberOfCommon:] = list(np.random.permutation(new_individual[numberOfCommon:]))
        #print(new_individual)

        ### TO DO implement self-adaptibility: change alpha of EACH individual
        return new_individual

    def getOffspring(self):
        """Fills the offspring array by performing crossover on pairs of parents"""
        for j in range(self.offspringSize):
            parent1 = self.selection()
            parent2 = self.selection()
            # Make sure the parents are different ### IS THIS NECESSARY?
            #while parent1 == parent2:
            #    parent2 = self.selection()
            self.offspring[j] = self.crossover(parent1, parent2)

    def mutate_individual(self, individual):
        dice_throw = np.random.uniform()
        if dice_throw <= self.alpha:
            # Get two random integers:
            first_index, second_index = random.sample(range(1, self.numberOfObjects), 2)
            # Get their values and swap
            temp = individual[first_index]
            individual[first_index] = individual[second_index]
            individual[second_index] = temp

        return individual

    def mutation(self):
        """Mutates ALL individuals. Selects two indices in the individual's permutation and swaps them"""

        # Mutate the original population
        for index, individual in enumerate(self.population):
            self.population[index] = self.mutate_individual(individual)

        # Mutate the offspring
        for index, individual in enumerate(self.offspring):
            self.offspring[index] = self.mutate_individual(individual)

    def elimination(self):
        """Eliminates individuals from the population and offspring, with lambda + mu"""
        all_fitness = []

        # Get fitness values in joined population and sort:
        for individual in self.allIndividuals:
            all_fitness.append(self.fitness(individual, return_weight=False))
        sorting_indices = np.argsort(all_fitness)

        # Get the best lambda individuals, save them in population
        for index, sorting_index in enumerate(sorting_indices[-self.populationSize:]):
            self.population[index] = self.allIndividuals[sorting_index]

    def joinPopulations(self):
        """Joins original population with the offspring"""
        self.allIndividuals = self.population + self.offspring

    def fitness(self, individual, return_weight=True):
        """Computes the fitness of a single individual. Go through loop as far as possible!"""
        value = 0
        weight = 0

        for i in individual:
            next_weight = weight + self.objectWeights[i]
            if next_weight < self.capacity:
                value += self.objectValues[i]
                weight += self.objectWeights[i]

        if return_weight:
            return value, weight
        else:
            return value

    def inKnapsack(self, individual):
        """Returns a set of items chosen by individual in the knapsack"""
        items = []
        weight = 0

        for i in individual:
            next_weight = weight + self.objectWeights[i]
            if next_weight < self.capacity:
                items.append(i)
                weight += self.objectWeights[i]
        return set(items)


    def set_fitness_population(self, return_weight=True):
        """Computes the fitness of a population of individuals"""
        total_values = []
        total_weights = []

        for individual in self.population:
            value, weight = self.fitness(individual, return_weight=return_weight)
            total_values.append(value)
            total_weights.append(weight)

        self.populationValues = total_values
        self.populationWeights = total_weights


# Test area:
knapsack = KnapsackProblem(10)
knapsack.optimize()

# Plot the solution