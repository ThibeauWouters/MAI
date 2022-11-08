import Reporter
import numpy as np
import math
import random
from random import sample
import matplotlib.pyplot as plt
import time

# route_matrix=[[0,3,math.inf,math.inf,1],[math.inf,0,3,math.inf,2],[math.inf,math.inf,0,2,2],[3,math.inf,math.inf,0,3],[math.inf,1,math.inf,3,0]]

class r0708518:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)
		self.lambdaa = 100
		self.population = []
		# max number of optimization cycles
		self.numberOfIterations = 100
		self.dMatrix = None
		self.tournament_size = 10
		self.mu = 100

	def optimize(self, filename, plot_name='plot'):
		"""The evolutionary algorithm's main loop"""
		# Create empty lists to be filled with values to report progress
		mean_fit_values = []
		best_fit_values = []

		# Read distance matrix from file.
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()
		self.dMatrix = distanceMatrix

		# Your code here.
		self.initialize()
		self.population.sort(key=lambda x: x[1])

		# Loop:
		start = time.time()
		for _ in range(self.numberOfIterations):
			fitnesses = list(map(lambda x: x[1], self.population))
			meanObjective = sum(fitnesses)/self.lambdaa
			bestObjective = fitnesses[0]
			bestSolution = self.population[0][0]

			# Add to the list for later on:
			mean_fit_values.append(meanObjective)
			best_fit_values.append(bestObjective)

			# Your code here.
			offspring = []
			for _ in range(self.mu):
				p1, p2 = self.selection()
				child = self.recombination(p1[0], p2[0])
				child = self.mutation(child)
				offspring.append(child)
			self.elimination(offspring)


			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break
		end = time.time()
		time_diff = end - start
		number_of_minutes = time_diff // 60
		number_of_seconds = time_diff % 60
		print(f"The algorithm took {number_of_minutes} minutes and {number_of_seconds} seconds.")
		print(f"The final best fitness value was {bestObjective}")

		# Plot the results
		plt.figure(figsize=(7, 5))
		plt.plot(mean_fit_values, '--o', color='red', label="Mean")
		plt.plot(best_fit_values, '--o', color='blue', label="Best")
		plt.grid()
		plt.legend()
		plt.xlabel('Iteration step')
		plt.ylabel('Fitness')
		plt.title('TSP for ' + str(filename))
		# Save the plots (as PNG and PDF)
		plt.savefig('Plots/' + plot_name + '.png', bbox_inches='tight')
		plt.savefig('Plots/' + plot_name + '.pdf', bbox_inches='tight')
		#plt.show()
		plt.close()

		# Your code here.
		# Return the best fitness value & best solution
		return bestObjective, bestSolution

	def initialize(self):
		size = np.size(self.dMatrix[0])
		for i in range(self.lambdaa):
			# Random permutation, but always start in 'city 0'
			# TODO: can this be optimized?
			tail_chromosome = np.random.permutation(size-1) + 1
			chromosome = np.insert(tail_chromosome, 0, 0)
			fitness = self.fitness(chromosome)
			individual = (chromosome, fitness)
			self.population.append(individual)

	def recombination(self, parent1, parent2):
		"""Parent1 and parent2 should be chromosomes"""

		# Save number of cities and parents for convenience
		number_of_cities = len(parent1)
		parents = [parent1, parent2]

		# TODO: can this be achieved faster?

		# Initialize a new child with appropriate length
		child = np.zeros_like(parent1)

		# Save the indices that were not assigned yet, and cities being assigned, throughout the process
		indices_not_assigned = [i for i in range(number_of_cities)]
		cities_not_assigned = [i for i in range(number_of_cities)]

		# Iterate over parents, save entries common in both parents
		for i in range(len(parent1)):
			if parent1[i] == parent2[i]:
				# Save value, and remove from remaining cities to be assigned
				child[i] = parent1[i]
				indices_not_assigned.remove(i)
				cities_not_assigned.remove(parent1[i])

		# If all indices were assigned (parents are identical chromosomes), return
		if len(indices_not_assigned) == 0:
			fit = self.fitness(child)
			return child, fit

		# If not, randomly assign remaining cities
		while len(indices_not_assigned) > 0:
			random_city = random.choice(cities_not_assigned)
			next_index = indices_not_assigned[0]
			child[next_index] = random_city
			indices_not_assigned.remove(next_index)
			cities_not_assigned.remove(random_city)

		# Now, child should be completed - return it
		fit = self.fitness(child)
		return child, fit

	def mutation(self, individual):
		"""Swaps two entries in the cycle"""
		# Get the cities of the tour first
		individual = individual[0]

		# Only change from index 1: always keep index 0 equal to 0
		position_list = random.sample(range(1, len(individual)), 2)
		temp = individual[position_list[0]]
		individual[position_list[0]] = individual[position_list[1]]
		individual[position_list[1]] = temp

		# Get fitness and return
		fit = self.fitness(individual)
		return individual, fit

	def selection(self):

		competitors_1 = sample(self.population, self.tournament_size)
		competitors_2 = sample(self.population, self.tournament_size)
		competitors_1.sort(key=lambda x: x[1])
		competitors_2.sort(key=lambda x: x[1])
		father, mother = competitors_1[0], competitors_2[0]
		return father, mother

	"""lambda+mu elimination"""
	def elimination(self, offspring):
		self.population.extend(offspring)
		self.population.sort(key=lambda x: x[1])
		self.population = self.population[:self.lambdaa]

	def fitness(self, tour):
		"""Computes the fitness value of an individual"""
		fitness = 0

		# For the 'body' of the tour:
		for i in range(len(tour) - 1):
			fitness += self.dMatrix[tour[i], tour[i+1]]

		# 'Close' the tour:
		fitness += self.dMatrix[tour[-1], tour[0]]
		return fitness


if __name__=="__main__":
	mytest = r0708518()
	mytest.optimize('./tour50.csv')
	pass
