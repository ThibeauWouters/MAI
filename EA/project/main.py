import Reporter
import numpy as np
import pandas as pd
import csv
import random
from random import sample
import matplotlib.pyplot as plt
import time


class r0708518:

	def __init__(self, which_mutation="DM", which_recombination="OX2"):
		self.reporter = Reporter.Reporter(self.__class__.__name__)
		self.dMatrix = None
		# Size of population and offspring
		self.lambdaa = 100
		# TODO - is there some better way for this?
		self.population = []
		self.mu = 100
		# self.offspring = []
		# Hyperparameters:
		self.tournament_size = 10
		# max number of optimization cycles
		self.numberOfIterations = 9999999999999
		# TODO - remove the following?
		self.iterationCounter = 0
		self.no_improvement_max = self.numberOfIterations/10
		# Delta is the threshold for improvement between consecutive iterations (check convergence)
		self.delta = -1

		# Choose which algorithm we are going to use for the various steps of the optimization loop
		self.which_mutation = which_mutation
		print("Mutation operator: " + self.which_mutation)
		self.which_recombination = which_recombination
		print("Recombination operator: " + self.which_recombination)

	def optimize(self, filename):
		"""The evolutionary algorithm's main loop"""
		# Create empty lists to be filled with values to report progress
		mean_fit_values = []
		best_fit_values = []

		# Read distance matrix from file.
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()
		# Replace inf values with just a penalty term
		# TODO - what are good values for penalty term?
		# First, replace inf values by -1
		distanceMatrix = np.where(distanceMatrix == float('inf'), -1, distanceMatrix)
		# Get max value, use it to construct a penalty term, and replace again
		max_distance = np.max(distanceMatrix.flatten())
		penalty_term = 2*max_distance
		distanceMatrix = np.where(distanceMatrix == -1, penalty_term, distanceMatrix)

		self.dMatrix = distanceMatrix

		# Initialize the population
		self.initialize()
		self.population.sort(key=lambda x: x[1])

		# Initialize certain variables that keep track of the convergence
		global_counter = 0
		no_improvement_counter = 0
		current_best = 0
		previous_best = 0

		# The main evolutionary algorithm loop comes here:
		start = time.time()
		while global_counter < self.numberOfIterations and no_improvement_counter < self.no_improvement_max:
			# Old best fitness value is previous 'current' one
			previous_best = current_best

			# For reporting progess: get mean and best values
			fitnesses = list(map(lambda x: x[1], self.population))
			meanObjective = sum(fitnesses)/self.lambdaa
			bestObjective = fitnesses[0]
			bestSolution = self.population[0][0]

			# Update the current best fitness value
			current_best = bestObjective

			# If not enough improvement was seen this iteration, count up
			if abs(previous_best - current_best) < self.delta:
				no_improvement_counter += 1
			else:
				# If enough improvement was seen: reset counter
				no_improvement_counter = 0

			# Add fitness values to a list to plot them later on:
			mean_fit_values.append(meanObjective)
			best_fit_values.append(bestObjective)

			# Get offspring
			offspring = [0]*self.mu
			for i in range(self.mu):
				p1, p2 = self.selection()
				# TODO - is there a redundant fitness calculation here?
				child = self.recombination(p1[0], p2[0])
				child = self.mutation(child[0])
				offspring[i] = child

			# TODO - Mutate the original population as well?

			# Extend population
			self.population.extend(offspring)
			# Eliminate
			self.elimination()

			global_counter += 1
			self.iterationCounter += 1

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
		print(f"The algorithm took {number_of_minutes} minutes and {round(number_of_seconds)} seconds and {global_counter} iterations.")
		print(f"The final best fitness value was {round(bestObjective)}")
		print(f"The final best individual was {bestSolution}")
		all_different = (len(pd.unique(bestSolution)) == len(bestSolution))
		print(f"All different? {all_different}")

		# Plot the results
		plt.figure(figsize=(7, 5))
		plt.plot(mean_fit_values, '--o', color='red', label="Mean")
		plt.plot(best_fit_values, '--o', color='blue', label="Best")
		plt.grid()
		plt.legend()
		plt.xlabel('Iteration step')
		plt.ylabel('Fitness')
		plt.title('TSP for ' + str(filename))
		plot_name = "plot_mut_" + self.which_mutation + "_rec_" + self.which_recombination
		# Save the plots (as PNG and PDF)
		plt.savefig('Plots/' + plot_name + '.png', bbox_inches='tight')
		plt.savefig('Plots/' + plot_name + '.pdf', bbox_inches='tight')
		plt.close()

		# Return the best fitness value & best solution
		# TODO - change this again
		# return bestObjective, bestSolution
		return bestObjective, meanObjective, self.iterationCounter

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

	#########################
	# --- RECOMBINATION --- #
	#########################

	def recombination(self, parent1, parent2):
		"""Performs the chosen recombination operator."""

		if self.which_recombination == "PMX":
			return self.partially_mapped_crossover(parent1, parent2)
		elif self.which_recombination == "SCX":
			return self.single_cycle_crossover(parent1, parent2)
		elif self.which_recombination == "OX":
			return self.order_crossover(parent1, parent2)
		elif self.which_recombination == "OX2":
			return self.order_based_crossover(parent1, parent2)
		elif self.which_recombination == "CX":
			return self.cycle_crossover(parent1, parent2)
		elif self.which_recombination == "EX":
			return self.edge_crossover(parent1, parent2)
		elif self.which_recombination == "AX":
			return self.alternating_crossover(parent1, parent2)

		elif self.which_recombination == "GROUP":
			return self.group_recombination(parent1, parent2)

		# Default choice
		else:
			return self.partially_mapped_crossover(parent1, parent2)

	def order_based_crossover(self, parent1, parent2):
		"""(OX2) Performs the order based crossover operator."""

		# Initialize child
		child = np.full(len(parent1), -1)

		# Generate a sample of indices of length k
		# TODO - choose a different version? Now, k is fixed to half the parent length, so more of a mutation really.
		k = len(parent1)//7
		indices = np.sort(np.random.choice([i for i in range(1, len(parent1))], size=k, replace=False))

		# Get cities at those positions at parent 2, look up their indices in parent1
		cities = parent2[indices]
		new_indices = np.in1d(parent1, cities)

		# Copy parent1 at places which are not the new indices, copy cities in preserved order at selected indices
		child[~new_indices] = parent1[~new_indices]
		child[new_indices] = cities

		fit = self.fitness(child)
		return child, fit



	def alternating_crossover(self, parent1, parent2):
		"""(AX) Performs the alternating position crossover operator."""

		child = np.empty(2*len(parent1), dtype=parent1.dtype)
		child[0::2] = parent1
		child[1::2] = parent2
		child = pd.unique(child)

		fit = self.fitness(child)
		return child, fit

	def edge_crossover(self, parent1, parent2):
		"""(EX) Performs the edge crossover operator."""

		# TODO - only consider rolls in 1D

		# STEP 1: construct the edge table

		# Initialize the edge table
		edge_table = [np.array([], dtype='int') for i in range(len(parent1))]

		# Do numpy roll on parents to easily get the edges
		roll_p1_left = np.roll(parent1, 1)
		roll_p1_right = np.roll(parent1, -1)

		roll_p2_left = np.roll(parent2, 1)
		roll_p2_right = np.roll(parent2, -1)

		# TODO - can be done faster?
		for i in range(len(parent1)):
			# Look at edges of allele at i in parent 1
			index = parent1[i]
			edge_table[index] = np.concatenate([edge_table[index], np.array([roll_p1_left[i], roll_p1_right[i]])], dtype='int')

			# Same for parent2
			index = parent2[i]
			edge_table[index] = np.concatenate([edge_table[index], np.array([roll_p2_left[i], roll_p2_right[i]])], dtype='int')

		# STEP 2: do the loop

		# First element: choose 0 to guarantee constraint is met
		child = np.full(len(parent1), -1)
		child[0] = 0
		current_element = 0
		unassigned = [i for i in range(1, len(parent1))]

		for i in range(1, len(parent1)):
			# Remove all references to current element in the edge table
			for j in range(len(edge_table)):
				if current_element in edge_table[j]:
					ids = np.in1d(edge_table[j], np.array([current_element]))
					remainder = edge_table[j][~ids]
					edge_table[j] = remainder

			# Consider edges of current element
			edges = edge_table[current_element]

			# In case this list is empty:
			if len(edges) == 0:
				# Pick a random element
				# current_element += 1
				current_element = random.sample(unassigned, 1)[0]
				# print("random sample")
				# continue
			else:
				# OPTION 1 --- if there is a common edge, choose that element
				unique, counts = np.unique(edges, return_counts=True)
				if 2 in counts:
					# print("option1")
					current_element = unique[np.argwhere(counts == 2)[0][0]]
				else:

					# OPTION 2 --- pick entry with itself having the smallest list
					lengths = []
					for element in edges:
						lengths.append(len(edge_table[element]))
					# Note, argmin returns smallest index, so like a random choice
					# print("option2")
					current_element = edges[np.argmin(lengths)]

			# Fill in value and repeat loop
			child[i] = current_element
			unassigned.remove(current_element)

		fit = self.fitness(child)
		return child, fit

	def order_crossover(self, parent1, parent2):
		"""(OX) Performs the order crossover operator."""

		# Introduce two random cut points to get subtour of parent1
		a, b = np.sort(np.random.permutation(np.arange(1, len(parent1)))[:2])

		# Find the remaining cities, and use their order as given in second parent
		ids = np.in1d(parent2, parent1[a:b])
		remainder = parent2[~ids]

		# Add these two together, and make sure 0 is first element
		child = np.concatenate([parent1[a:b], remainder])
		idzero = np.argwhere(child == 0)[0][0]
		child = np.roll(child, -idzero)

		fit = self.fitness(child)
		return child, fit

	def single_cycle_crossover(self, parent1, parent2):
		"""(SCX) Performs the cycle crossover, but only performs one such cycle."""

		# Initialize child, make sure to start at zero
		child = np.full(len(parent1), -1)
		child[0] = 0

		# Initialize information for a 'cycle'
		index = 1
		child[index] = parent1[index]
		first_index = index

		while True:
			# Look at the allele with the same position in P2
			allele = parent2[index]

			# Go to the position with the same allele in P1
			next_index = np.argwhere(parent1 == allele)[0][0]

			# Add this allele to the cycle
			child[next_index] = parent1[next_index]

			index = next_index
			# In case we completed the cycle, start the next 'cycle' -- swap order of parents
			if index == first_index:
				child = np.where(child == -1, parent2, child)
				fit = self.fitness(child)
				return child, fit

	def cycle_crossover(self, parent1, parent2):
		"""(CX) Performs the cycle crossover, but only performs one such cycle."""

		# Initialize child, make sure to start at zero
		child = np.full(len(parent1), -1)
		child[0] = 0

		# Initialize information for a 'cycle'
		index = 1
		first_index = index
		value_parent = parent1
		which_value_parent = "1"
		child[index] = value_parent[index]

		while True:
			# Look at the allele with the same position in P2
			allele = parent2[index]

			# Go to the position with the same allele in P1
			next_index = np.argwhere(parent1 == allele)[0][0]

			# Add this allele to the cycle
			child[next_index] = value_parent[next_index]

			index = next_index
			# In case we completed the cycle, start the next 'cycle' -- swap order of parents
			if index == first_index:
				if -1 not in child:
					fit = self.fitness(child)
					return child, fit
				else:
					# Start a new cycle
					index = np.argwhere(child == -1)[0][0]
					first_index = index
					if which_value_parent == "1":
						value_parent = parent2
						which_value_parent = "2"
					else:
						value_parent = parent1
						which_value_parent = "1"

	def partially_mapped_crossover(self, parent1, parent2):
		""" (PMX) Implements the partially mapped crossover."""

		# Initialize two children we are going to create
		child1 = np.zeros_like(parent1)
		# child2 = np.zeros_like(parent1)

		# Generate cut points a and b: these are two random indices, sorted
		# TODO - make sure that a and b differ by "enough"?
		a, b = np.sort(np.random.permutation(np.arange(1, len(parent1)))[:2])

		# Get the cuts from the two parents
		# NOTE: parent[a:b] gives elements parent[a], ..., parent[b-1] !!!
		cut1 = np.array(parent1[a:b])
		cut2 = np.array(parent2[a:b])

		# Cross the cuts
		child1[a:b] = cut2
		# child2[a:b] = cut1

		# Check which indices remain to be assigned
		remaining_indices = np.where(child1 == 0)

		# Iterate over the remaining entries
		for i in np.nditer(remaining_indices):
			# --- Fill child 1
			# Get the value we WISH to fill in:
			value = parent1[i]

			# If this element, or any we will now find, was already copied from parent 2:
			while value in cut2:
				# look up index of this element in parent2
				index = np.where(parent2 == value)
				# Then use the mapping cut1 <-> cut2 to get new value. Check if new value also in cut2 (while loop)
				value = parent1[index]

			# if not, just use the value of parent 1
			child1[i] = value

		# print("Child ", child1)

		fit = self.fitness(child1)
		return child1, fit

	def group_recombination(self, parent1, parent2):
		"""Copies the intersection of two parents. Distributes the remaining cities of first parent to child after
			permutation. First implementation of recombination algorithm."""

		# TODO - give this the correct name :P

		# Child starts off with the intersection of the parents. Fill remaining with -1 (to recognize it later).
		# Since cities start in 0, this constraint will automatically copy over to child.
		child = np.where(parent1 == parent2, parent1, -1)

		# Get the indices of child which were not assigned yet.
		leftover_indices = np.where(child == -1)[0]

		# Get the cities that appear in one of the parents, permute them
		# TODO does it matter WHICH parent, maybe choose the one with best fitness?
		leftover_cities_permuted = np.random.permutation(parent1[leftover_indices])

		# Store permuted cities in the child
		child[leftover_indices] = leftover_cities_permuted

		# Compute the fitness value and return
		fit = self.fitness(child)
		return child, fit

	####################
	# --- MUTATION --- #
	####################

	def mutation(self, individual):
		"""Performs the chosen mutation (chosen at initialization of self)"""
		if self.which_mutation == "EM":
			return self.exchange_mutation(individual)
		elif self.which_mutation == "DM":
			return self.displacement_mutation(individual)
		elif self.which_mutation == "SIM":
			return self.simple_inversion_mutation(individual)
		elif self.which_mutation == "ISM":
			return self.insertion_mutation(individual)
		elif self.which_mutation == "IVM":
			return self.inversion_mutation(individual)
		elif self.which_mutation == "SM":
			return self.scramble_mutation(individual)
		elif self.which_mutation == "SDM":
			return self.scrambled_displacement_mutation(individual)

		# Default choice (or: no mutation operator is given)
		else:
			return self.simple_inversion_mutation(individual)

	def scrambled_displacement_mutation(self, individual):
		"""(SDM) Takes a random subtour and inserts it, in 'scrambled' (permuted) order, at a random place.
			Note that this extends both the scramble and displacement mutation, where the subtour gets scrambled."""

		# Randomly introduce two cuts in the individual
		a, b = np.sort(np.random.permutation(np.arange(1, len(individual)))[:2])
		subtour = individual[a:b]
		# Delete the subtour from individual
		individual = np.delete(individual, np.arange(a, b, 1))
		# Insert it at a random position, but reverse the order
		insertion_point = random.randint(1, len(individual))
		individual = np.insert(individual, insertion_point, np.random.permutation(subtour))

		fit = self.fitness(individual)
		return individual, fit

	def scramble_mutation(self, individual):
		"""(SM) Takes a random subtour of the individual, and reverses that subtour at that location."""

		# Randomly introduce two cuts in the individual, and reverse that part of individual
		a, b = np.sort(np.random.permutation(np.arange(1, len(individual)))[:2])
		individual[a:b] = np.random.permutation(individual[a:b])

		fit = self.fitness(individual)
		return individual, fit

	def inversion_mutation(self, individual):
		"""(IVM) Takes a random subtour and inserts it, in reversed order, at a random place.
			Note that this is an extension of the displacement mutation, where the subtour gets reversed."""

		# Randomly introduce two cuts in the individual
		a, b = np.sort(np.random.permutation(np.arange(1, len(individual)))[:2])
		subtour = individual[a:b]
		# Delete the subtour from individual
		individual = np.delete(individual, np.arange(a, b, 1))
		# Insert it at a random position, but reverse the order
		insertion_point = random.randint(1, len(individual))
		individual = np.insert(individual, insertion_point, subtour[::-1])

		fit = self.fitness(individual)
		return individual, fit

	def insertion_mutation(self, individual):
		"""(ISM) Takes a random city and inserts it at a random place.
			Note that this is a special case of displacement mutation, where the subtour has length 1."""

		# Randomly take two different index positions
		a, b = np.sort(np.random.permutation(np.arange(1, len(individual)))[:2])
		subtour = individual[a]
		# Delete the subtour from individual
		individual = np.delete(individual, a)
		# Insert it at a random position
		individual = np.insert(individual, b, subtour)

		fit = self.fitness(individual)
		return individual, fit

	def simple_inversion_mutation(self, individual):
		"""(SIM) Takes a random subtour of the individual, and reverses that subtour at that location."""

		# Randomly introduce two cuts in the individual, and reverse that part of individual
		a, b = np.sort(np.random.permutation(np.arange(1, len(individual)))[:2])
		individual[a:b] = individual[a:b][::-1]

		fit = self.fitness(individual)
		return individual, fit

	def displacement_mutation(self, individual):
		"""Cuts a subtour of the individual, and places it in a random place"""
		# TODO - enforce certain length of subtour??? Or make sure don't return same?

		# Randomly introduce two cuts in the individual
		a, b = np.sort(np.random.permutation(np.arange(1, len(individual)))[:2])
		subtour = individual[a:b]
		# Delete the subtour from individual
		individual = np.delete(individual, np.arange(a, b, 1))
		# Insert it at a random position
		insertion_point = random.randint(1, len(individual))
		individual = np.insert(individual, insertion_point, subtour)

		fit = self.fitness(individual)
		return individual, fit

	def exchange_mutation(self, individual):
		"""Randomly swaps two entries in the cycle."""

		# idea: https://stackoverflow.com/questions/22847410/swap-two-values-in-a-numpy-array
		# more efficient to use numpy: See comments below this answer: https://stackoverflow.com/a/9755548/13331858
		# old method: # random.sample(range(1, len(individual)), 2)

		# Note: we only change from index 1: always keep index 0 equal to 0

		# Get two indices at which we will do a swap
		indices = np.random.permutation(np.arange(1,len(individual)))[:2]

		# Flip cities at those locations. Compute fitness and return
		individual[indices] = individual[np.flip(indices)]
		fit = self.fitness(individual)
		return individual, fit

	#####################
	# --- SELECTION --- #
	#####################

	def selection(self):
		return self.k_tournament_selection()

	def k_tournament_selection(self):

		competitors_1 = sample(self.population, self.tournament_size)
		competitors_2 = sample(self.population, self.tournament_size)
		competitors_1.sort(key=lambda x: x[1])
		competitors_2.sort(key=lambda x: x[1])
		father, mother = competitors_1[0], competitors_2[0]
		return father, mother

	#####################
	# --- ELMINATION --- #
	#####################

	"""lambda+mu elimination"""
	def elimination(self):
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


def analyze_operators():
	"""Given two sets of mutation and recombination operators, run the optimization loop for each combination
		of mutation and recombination operators and save important values to a CSV to analyze the performance."""
	print("Analyzing the performance of operators . . . ")
	print("------------------------")

	all_mutations = ["EM", "DM", "SIM", "ISM", "IVM", "SM", "SDM"]
	# "CX" and "EX" not included --- they are too slow to outperform any combination!
	all_recombinations = ["PMX", "SCX", "OX", "OX2", "AX", "GROUP"]

	for mut in all_mutations:
		for rec in all_recombinations:
			mytest = r0708518(which_mutation=mut, which_recombination=rec)
			bestObjective, meanObjective, iterationCounter = mytest.optimize('./tour50.csv')
			with open('Data/Analysis_mutation_recombination.csv', 'a', newline='') as file:
				writer = csv.writer(file)
				# write data (Mutation, Recombination, Best, Mean, Iterations)
				data = [mut, rec, bestObjective, meanObjective, iterationCounter]
				writer.writerow(data)
			print("--------------------------------")

if __name__=="__main__":
	# mytest = r0708518()
	# mytest.optimize('./tour50.csv')
	
	# parent1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
	# parent1 = np.array([0, 2, 6, 7, 1, 5, 4, 8, 3])
	# parent2 = np.array([0, 8, 7, 3, 6, 5, 2, 4, 1])
	# print(parent1)
	# print(parent2)
	# mytest.order_based_crossover(parent1, parent2)

	"""Analyzing the performance of mutation and crossover operators"""
	analyze_operators()

	pass
