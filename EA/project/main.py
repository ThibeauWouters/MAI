import Reporter
import numpy as np
import pandas as pd
import csv
import random
import matplotlib.pyplot as plt
import time

# Fix random seeds:
np.random.seed(0)
random.seed(0)


# --- Auxiliary functions ---

def check_subset_numpy_arrays(a, b):
	"""Returns true if a is a sub array of b"""
	b = np.unique1d(b)
	c = np.intersect1d(a, b)
	return np.size(c) == np.size(b)


def make_plot(mean_fit_values, best_fit_values, filename):
	# --- Plot the results
	# TODO - delete this at the end
	plt.figure(figsize=(7, 5))
	start = len(mean_fit_values) // 10
	plt.plot([i for i in range(start, len(mean_fit_values))], mean_fit_values[start:], '--o', ms=2, color='red',
			 label="Mean")
	plt.plot([i for i in range(start, len(best_fit_values))], best_fit_values[start:], '--o', ms=2, color='blue',
			 label="Best")
	plt.legend()
	plt.grid()
	plt.xlabel('Iteration step')
	plt.ylabel('Fitness')
	# plt.yscale('log')
	plt.title('TSP for ' + str(filename))
	# plot_name = "plot_mut_" + self.which_mutation + "_rec_" + self.which_recombination
	plot_name = "plot_test_run"
	# Save the plots (as PNG and PDF)
	plt.savefig('Plots/' + plot_name + '.png', bbox_inches='tight')
	plt.savefig('Plots/' + plot_name + '.pdf', bbox_inches='tight')
	plt.close()


class r0708518:

	def __init__(self, user_params_dict=None):
		self.reporter = Reporter.Reporter(self.__class__.__name__)
		self.distance_matrix = None

		# Store default parameter values here
		self.default_params_dict = {'lambdaa': 50, 'mu': 50, "which_mutation": "DM", "which_recombination": "OX2",
							   "which_elimination": "lambda plus mu", "which_selection": "k tournament",
								"tournament_size": 5, "number_of_iterations": 100,
							   "round_robin_size": 10, "random_perm_init_number": 100, "random_road_init_number": 0,
							   "greedy_road_init_number": 0, "nnb_road_init_number": 0, "which_lso": "2-opt",
							   "lso_pivot_rule": "greedy descent", "lso_init_sample_size": 100,
							   "lso_rec_sample_size": 10, "lso_mut_sample_size": 10, "lso_init_depth": 1,
							   "lso_rec_depth": 1, "lso_mut_depth": 1, "use_lso": False,
							   "alpha": 1, "delta": 0.1}

		# TODO - declare all fields with explanation?

		# TODO - check constraints, is the sum of init numbers equal to lambda? What if it's too large/small?

		# Set the default (hyper)parameters
		for key in self.default_params_dict.keys():
			setattr(self, key, self.default_params_dict[key])
		# Overwrite the (hyper)parameters set by the user:
		if user_params_dict is not None:
			for key in user_params_dict:
				setattr(self, key, user_params_dict[key])
		self.iterationCounter = 0
		self.no_improvement_max = 500
		# TODO - remove the following?
		# Delta is the threshold for improvement between consecutive iterations (check convergence)
		# self.delta = 0.1
		# Alpha is the mutation rate we start off with, but gets adapted
		# self.alpha = 1

		# Meta: here, the functions that we implemented in the algorithm are saved, in order to check that the user
		# selected a method which is implemented, and if not, we can use the "default" choices in methods defined below
		self.implemented_mutation_operators      = ["EM", "DM", "SIM", "ISM", "IVM", "SM", "SDM"]
		self.implemented_recombination_operators = ["PMX", "SCX", "OX", "OX2", "CX", "EX", "AX", "GROUP"]
		self.implemented_selection_operators     = ["k tournament"]
		self.implemented_elimination_operators   = ["lambda plus mu", "lambda and mu", "round robin"]

		# Probability distributions that one can use to select items from a numpy array: see self.get_probabilities
		self.implemented_probability_distributions = ["geom"]

		# For testing: print the chosen operators etc, and check if they are implemented, otherwise select a default
		# TODO - make sure the defaults are implemented correctly and conveniently?

		# Show which implementations were chosen by the user:
		print("Mutation operator:      %s" % self.which_mutation)
		print("Recombination operator: %s" % self.which_recombination)
		print("Elimination operator:   %s" % self.which_elimination)
		print("Selection operator:     %s" % self.which_selection)

		# TODO - allow for more general or check if condition is valid?
		if self.which_elimination == "lambda and mu":
			# For lambda and mu, where we replace current generation with offspring, make sure they are same size
			smallest = min(self.lambdaa, self.mu)
			self.lambdaa = smallest
			self.mu = smallest

	def optimize(self, filename):
		"""The evolutionary algorithm's main loop"""
		start = time.time()
		# Create empty lists to be filled with values to report progress
		mean_fit_values = []
		best_fit_values = []

		# Read distance matrix from file.
		file = open(filename)
		distance_matrix = np.loadtxt(file, delimiter=",")
		file.close()
		# Replace inf values with just a penalty term
		# TODO - what are good values for penalty term?
		# First, replace inf values by -1
		distance_matrix = np.where(distance_matrix == float('inf'), -1, distance_matrix)
		# Get max value, use it to construct a penalty term, and replace again
		max_distance = np.max(distance_matrix.flatten())
		penalty_value = 2 * max_distance
		distance_matrix = np.where(distance_matrix == -1, penalty_value, distance_matrix)
		# Also apply the penalty for roads going from a city to itself (diagonal terms)
		distance_matrix = np.where(distance_matrix == 0, penalty_value, distance_matrix)

		# Save the penalty term for convenience for later on
		self.penalty_value = penalty_value
		# Save
		self.distance_matrix = distance_matrix
		# Save n, the size of the problem (number of cities)
		self.n = np.size(self.distance_matrix[0])

		# Also construct the connections matrix
		self.construct_connections_matrix()

		# If LSO uses 2-opt, then generate all (i, j) pairs: this also determines the size of the neighbourhood
		# TODO - 2-opt without constructing this list as it's likely too expensive memory-wise for the larger problems?
		if self.which_lso == "2-opt":
			self.two_opt_ij_pairs = np.array(
				[np.array([i, j]) for i in range(1, self.n - 1) for j in range(i + 1, self.n)])

		# Make sure that the number of nb we want to sample does not exceed the size of the nb hood
		self.lso_init_sample_size = max(self.lso_init_sample_size, len(self.two_opt_ij_pairs))
		self.lso_rec_sample_size = max(self.lso_rec_sample_size, len(self.two_opt_ij_pairs))
		self.lso_mut_sample_size = max(self.lso_mut_sample_size, len(self.two_opt_ij_pairs))

		# If we do steepest descent, this sample size is the nb size
		if self.lso_pivot_rule == "steepest descent":
			self.lso_init_sample_size = len(self.two_opt_ij_pairs)
			self.lso_rec_sample_size = len(self.two_opt_ij_pairs)
			self.lso_mut_sample_size = len(self.two_opt_ij_pairs)

		# Allocate memory for population and offspring
		self.population = np.empty((self.lambdaa, self.n + 1))
		self.offspring = np.empty((self.mu, self.n + 1))

		# Initialize the population, and sort based on the final column, which contains the fitness value
		self.initialize()
		self.population = self.population[self.population[:, -1].argsort()]

		# Initialize certain variables that keep track of the convergence
		global_counter = 0
		no_improvement_counter = 0
		current_best = 0
		previous_best = 0

		"""This is where the main evolutionary algorithm loop comes:"""
		while global_counter < self.number_of_iterations and no_improvement_counter < self.no_improvement_max:
			# Adapt hyperparameters:
			alpha = self.alpha ** (global_counter)

			# Old best fitness value is previous 'current' one
			previous_best = current_best

			# For reporting progess: get mean and best values
			# Get the mean value of the fitnesses, which is the final column of each individual
			mean_objective = np.mean(self.population[:, -1])
			best = self.population[self.population[:, -1].argmin()]
			# worst = self.population[self.population[:, -1].argmax()]
			best_objective = best[-1]
			best_solution = best[:-1].astype('int')

			# Update the current best fitness value
			current_best = best_objective

			# If not enough improvement was seen this iteration, count up
			if abs(previous_best - current_best) < self.delta:
				no_improvement_counter += 1
			else:
				# If enough improvement was seen: reset counter
				no_improvement_counter = 0

			# Add fitness values to a list to plot them later on:
			mean_fit_values.append(mean_objective)
			best_fit_values.append(best_objective)

			# Generate new offspring
			for i in range(self.mu):
				p1, p2 = self.parents_selection()
				# TODO - is there a redundant fitness calculation here?
				child = self.recombination(p1[:-1], p2[:-1])
				child = self.lso(child, depth=self.lso_rec_depth, sample_size=self.lso_rec_sample_size)
				if random.uniform(0, 1) < alpha:
					child = self.mutation(child[:-1])
					child = self.lso(child, depth=self.lso_mut_depth, sample_size=self.lso_mut_sample_size)
				self.offspring[i] = child

			# TODO - Mutate the original population as well?
			# Eliminate
			self.elimination()

			global_counter += 1
			# self.iterationCounter += 1

			# TODO - what are some clever LSO activation rules?
			### Check if we should activate the LSO
			# if global_counter >= 0.99*self.number_of_iterations and self.use_lso == False:
			# 	print("Before activating LSO: best %f" % best_objective)
			# 	print("Before activating LSO: avrg %f" % mean_objective)
			# 	self.use_lso = True

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(mean_objective, best_objective, best_solution)
			if timeLeft < 0:
				break

		# --- Report on progress within this window
		# TODO - delete this at the end
		end = time.time()
		time_diff = end - start
		number_of_minutes = time_diff // 60
		number_of_seconds = time_diff % 60
		print(
			f"The algorithm took {number_of_minutes} minutes and {round(number_of_seconds)} seconds and {global_counter} iterations.")
		print(f"The final best fitness value was {round(best_objective)}")
		print(f"The final average fitness value was {round(mean_objective)}")
		print(f"The final best individual was {best_solution}")
		all_different = (len(pd.unique(best_solution)) == len(best_solution))
		print(f"All different? {all_different}")

		# For our own purpose, make a plot as well:
		make_plot(mean_fit_values, best_fit_values, filename)

		# Return the best fitness value & best solution
		# TODO - change this again
		# return bestObjective, bestSolution
		return best_objective, mean_objective, global_counter

	##########################
	# --- INITIALIZATION --- #
	##########################

	def initialize(self):
		"""Initializes the population using several techniques and concatenates them."""

		# Empty list to save all the subpopulations we are going to construct
		subpopulations = []
		# TODO - check if this is possible automatically but I guess not...

		# Initialize by random permutation:
		if self.random_perm_init_number != 0:
			subpopulations.append(self.random_perm_initialize(self.random_perm_init_number))

		# Initialize by random selection of roads, which avoids the infs:
		if self.random_road_init_number != 0:
			subpopulations.append(self.road_initialize(self.random_road_init_number, method="random"))

		# Initialize by stochastically, greedily selecting the roads:
		if self.greedy_road_init_number != 0:
			subpopulations.append(self.road_initialize(self.greedy_road_init_number, method="greedy"))

		# Initialize by nearest neighbours, but starting location is random:
		if self.nnb_road_init_number != 0:
			subpopulations.append(self.road_initialize(self.nnb_road_init_number, method="nearest nb"))

		# Append all the subpopulations together in one big population array, save it
		self.population = np.concatenate(subpopulations)

	def construct_connections_matrix(self):
		"""Creates a matrix from the distance matrix of the problem, where each row i specifies which cities
			one can reach from i, avoiding the roads that give inf (or equivalently, large penalty values). Used
			in constructing initial populations in a more informed manner. In each rows, these possible cities
			are ordered based on their cost, ranging from low to high. The result is saved as instance."""
		# Get the connections matrix, showing the valid routes
		connections_matrix = []

		for i in range(self.n):
			# Get all the roads
			all_roads = np.array([i for i in range(self.n)])
			# Look at the next row
			row = self.distance_matrix[i]
			# Get the indices where the distance is not equal to the penalty value (inf in matrix, illegal road)
			indices = np.where((row != self.penalty_value) & (row > 0))[0]
			# Select these roads
			selected_roads = all_roads[indices]
			# Get the cost of the selected roads:
			road_values = row[indices]
			# Sort these values, and save the index positions used for this sorting procedure
			sort_indices = np.argsort(road_values)
			# Sort the selected roads based on the order dictated by the values, low to high
			sorted_selected_roads = selected_roads[sort_indices]
			# Append those roads which do not have "inf" to the connections matrix
			connections_matrix.append(sorted_selected_roads)

		self.connections_matrix = connections_matrix

	def get_probabilities(self, values, distr="geom"):
		"""Generates a numpy array of probabilities to select the entries of that numpy array.
			values: numpy array containing the values for which we want to get probabilities
			distr: string indiciating which distribution should be used for getting the probabilities. Default is
			the geometric distribution, which is a discrete version of the exponential distribution."""

		# TODO - take these values into account when getting the probabilities?
		# TODO - these things such as "geom": get them as fields, to easily adapt to new things?
		if distr not in self.implemented_probability_distributions:
			distr = "geom"

		# Geometric distribution:
		if distr == "geom":
			# Set the probability of success on each trial (p)
			p = 0.4
			# Set the number of failures before the first success (k)
			k = np.array([i for i in range(len(values))])
			# Calculate the probabilities for each value of k using the geometric probability distribution
			probabilities = p * (1 - p) ** (k - 1)
			return probabilities / np.sum(probabilities)

	def construct_tour(self, current, final_index, final_city, method="random"):
		"""Starting from an individual which is only constructed partially, continue constructing a full individual,
			taking into account the possible connections (no infs in distance matrix) and possibly also take into
			account the magnitude of the cost of taking a road. This function is called recursively, in order to
			be able to backtrack in case we get "stuck", i.e., we end up in a city connected to cities which were
			already selected in the construction of the individual.
			Parameters:
			current: individual, numpy array of size n, possibly containing -1 for cities which were not assigned yet
			final_index: the index at which the final city in individual was assigned in a previous call
			final_city: the city last assigned in the previous function call."""

		# print(current)

		# If the individual is finished, return it
		if final_index == (self.n - 1):
			return current

		# Check where we can go from here, but permute randomly to avoid determinism in this function
		# For the random method, we permute the connections in order to make it random
		if method == "random":
			current_connections = np.random.permutation(self.connections_matrix[final_city])
		# For any other method, we keep the order
		else:
			current_connections = self.connections_matrix[final_city]
		# From these possibilities, delete those that are already used
		possible_connections = np.setdiff1d(current_connections, current, assume_unique=True)
		# If no more options exist, break outside the loop (retry)
		if len(possible_connections) == 0:
			# print("Backtrack . . . ")
			return None
		# If OK: generate the next city randomly among possibilities
		elif method == "random":
			# Try each city in the connections until we finish the individual, then return
			for current_city in possible_connections:
				# If we had a successful addition, save into individual
				current[final_index + 1] = current_city
				# Continue the construction (recursive call!)
				end = self.construct_tour(current, final_index + 1, current_city)
				# If the recursive call ends up with an individual, then return it and break all loops
				if end is not None:
					return end
			# If we end up here, the for loop was unsuccesful, and backtrack "above":
			return None

		elif method == "greedy":
			# Try each city in the connections until we finish the individual, then return
			while len(possible_connections != 0):
				# Get the probabilities to select cities
				probabilities = self.get_probabilities(possible_connections)
				# Select one
				current_city = np.random.choice(possible_connections, p=probabilities)
				selected_index = np.where(possible_connections == current_city)[0]
				# Delete it from the possible connections in order to be able to backtrack afterwards
				possible_connections = np.delete(possible_connections, selected_index)
				# Save into individual
				current[final_index + 1] = current_city
				# Continue the construction (recursive call!)
				end = self.construct_tour(current, final_index + 1, current_city, method=method)
				# If the recursive call ends up with an individual, then return it and break all loops
				if end is not None:
					return end
			# If we end up here, we were unsuccesful and we need to backtrack above
			return None

		elif method == "nearest nb":
			# Try each city in the connections until we finish the individual, then return
			while len(possible_connections != 0):
				# For nearest neighbours, choose the first city as this is closest
				# print("Current connections: ", possible_connections)
				# print("All cost values : ", self.distance_matrix[final_city][possible_connections])
				current_city = possible_connections[0]
				# print("Chosen cost: ", self.distance_matrix[final_city][current_city])
				# Delete it from the possible connections in order to be able to backtrack afterwards
				possible_connections = possible_connections[1:]
				# Save into individual
				current[final_index + 1] = current_city
				# Continue the construction (recursive call!)
				end = self.construct_tour(current, final_index + 1, current_city, method=method)
				# If the recursive call ends up with an individual, then return it and break all loops
				if end is not None:
					return end
			# If we end up here, we were unsuccesful and we need to backtrack above
			return None

	def road_initialize(self, number=10, method="random"):
		if method not in ["random", "greedy", "nearest nb"]:
			print("Initialization method not recognized. Defaulting to random.")
			method = "random"

		# Generate a "subpopulation" based on how many are generated using this method
		result = np.empty((number, self.n + 1))

		# Construct "number" amount of individuals
		counter = 0
		while counter < number:
			# Initialize a new individual, start filling at zero
			individual = np.full(self.n, -1)
			# Pick a random starting point
			starting_point = np.random.choice(self.n)
			individual[0] = starting_point
			# Call construct_tour, which recursively makes a road
			individual = self.construct_tour(individual, 0, starting_point, method=method)
			if individual is not None:
				# Make sure that the genome starts with city 0 because of our convention
				idzero = np.argwhere(individual == 0)[0][0]
				individual = np.roll(individual, -idzero)
				# Compute its fitness and return it
				# print("Individual constructed:")
				# print(individual)
				fit = self.fitness(individual)
				individual = np.append(individual, fit)
				# Improve individual with LSO operator (nothing changed if lso_init_depth = 0)
				individual = self.lso(individual, depth=self.lso_init_depth)
				result[counter] = individual
				counter += 1
			else:
				print("I might be stuck here...")

		return result

	@DeprecationWarning
	def road_initialize_old(self, number=10, method="random"):
		"""At each city, selects randomly along the valid roads to another city to generate the initial population.
			Number: integer indicating how many individuals should be constructed in this way."""

		if method not in ["random", "greedy"]:
			print("Initialization method not recognized. Defaulting to random.")
			method = "random"

		# Generate a "subpopulation" based on how many are generated using this method
		result = np.empty((number, self.n + 1))

		# Construct "number" amount of individuals
		counter = 0
		while counter < number:
			# Initialize a new individual, start filling at zero
			individual = np.full(self.n, -1)
			individual[0] = 0
			seen = np.array([0])
			current_city = 0
			# Fill up the individual
			i = 1
			while i < self.n:
				# for i in range(1, self.n):
				# Check where we can go from here
				current_connections = self.connections_matrix[current_city]
				# From these possibilities, delete those that are already used
				possible_connections = np.setdiff1d(current_connections, seen, assume_unique=True)
				# If no more options exist, break outside the loop (retry)
				if len(possible_connections) == 0:
					print("We got stuck in the smart initialization! Wasted resources!")
					break
				# Else: generate the next city randomly among possibilities
				if method == "random":
					current_city = np.random.choice(possible_connections)
				elif method == "greedy":
					probabilities = self.get_probabilities(possible_connections)
					current_city = np.random.choice(possible_connections, p=probabilities)
				# Save into seen
				seen = np.append(seen, current_city)
				# Save it in the individual
				individual[i] = current_city

			# Check if we used all cities (completed the individual) or not, to increase counter
			if len(seen) == self.n:
				# Save the individual
				# Compute its fitness and append at the end
				fit = self.fitness(individual)
				individual = np.append(individual, fit)
				# Save it in the result, and increase counter
				result[counter] = individual
				counter += 1

		# We are done
		return result

	def random_perm_initialize(self, number=10):
		"""Initialize individuals by a random permutation of the cities. Quick and easy, but may end up selecting
			illegal roads, both inf roads as well as roads connecting a city to itself which may mislead the
			algorithm since those have a low cost."""
		# size = np.size(self.distance_matrix[0]) #old#

		result = np.empty((number, self.n + 1))
		for i in range(number):
			# Random permutation, but always start in 'city 0'
			# TODO: can this be optimized?
			# Make sure that a zero remains that the very first position:
			individual = np.zeros(self.n + 1)
			# Permute the remaining cities 1, ..., n:
			random_permutation = np.random.permutation(self.n - 1) + 1
			# Save both parts in the individual
			individual[1:self.n] = random_permutation
			# Compute the fitness and append to individual
			fitness = self.fitness(random_permutation)
			individual[-1] = fitness
			result[i] = individual

		return result

	#########################
	# --- RECOMBINATION --- #
	#########################

	def recombination(self, parent1, parent2):
		"""Performs the chosen recombination operator."""

		# Make sure the given option is implemented, otherwise use default
		if self.which_recombination not in self.implemented_recombination_operators:
			default = self.default_params_dict["which_recombination"]
			print("Recombination operator not recognized. Using default: %s" % default)
			self.which_recombination = default

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

		# Initialize child with -1 everywhere
		child = np.full(len(parent1), -1)

		# Generate a sample of indices of length k
		# TODO - choose a different version? Now, k is fixed to certain ration of parent length, so close to mutation
		k = len(parent1) // 7
		indices = np.sort(np.random.choice([i for i in range(1, len(parent1))], size=k, replace=False))

		# Get cities at those positions at parent 2, look up their indices in parent1
		cities = parent2[indices]
		new_indices = np.in1d(parent1, cities)

		# Copy parent1 at places which are not the new indices, copy cities in preserved order at selected indices
		child[~new_indices] = parent1[~new_indices]
		child[new_indices] = cities

		fit = self.fitness(child)
		child = np.append(child, fit)
		return child

	def alternating_crossover(self, parent1, parent2):
		"""(AX) Performs the alternating position crossover operator."""

		child = np.empty(2 * len(parent1), dtype=parent1.dtype)
		child[0::2] = parent1
		child[1::2] = parent2
		child = pd.unique(child)

		fit = self.fitness(child)
		child = np.append(child, fit)
		return child

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
			edge_table[index] = np.concatenate([edge_table[index], np.array([roll_p1_left[i], roll_p1_right[i]])],
											   dtype='int')

			# Same for parent2
			index = parent2[i]
			edge_table[index] = np.concatenate([edge_table[index], np.array([roll_p2_left[i], roll_p2_right[i]])],
											   dtype='int')

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
		child = np.append(child, fit)
		return child

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
		child = np.append(child, fit)
		return child

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
				child = np.append(child, fit)
				return child

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
					child = np.append(child, fit)
					return child
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
		child1 = np.append(child1, fit)
		return child1

	def group_recombination(self, parent1, parent2):
		"""Copies the intersection of two parents. Distributes the remaining cities of first parent to child after
			permutation. First implementation of recombination algorithm."""

		# TODO - give this the correct name!

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
		child = np.append(child, fit)
		return child

	####################
	# --- MUTATION --- #
	####################

	def mutation(self, individual):
		"""Performs the chosen mutation (chosen at initialization of self)"""

		# Make sure the given option is implemented, otherwise use default
		if self.which_mutation not in self.implemented_mutation_operators:
			default = self.default_params_dict["which_mutation"]
			print("Mutation operator not recognized. Using default: %s" % default)
			self.which_mutation = default

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
		individual = np.append(individual, fit)
		return individual

	def scramble_mutation(self, individual):
		"""(SM) Takes a random subtour of the individual, and reverses that subtour at that location."""

		# Randomly introduce two cuts in the individual, and reverse that part of individual
		a, b = np.sort(np.random.permutation(np.arange(1, len(individual)))[:2])
		individual[a:b] = np.random.permutation(individual[a:b])

		fit = self.fitness(individual)
		individual = np.append(individual, fit)
		return individual

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
		individual = np.append(individual, fit)
		return individual

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
		individual = np.append(individual, fit)
		return individual

	def simple_inversion_mutation(self, individual):
		"""(SIM) Takes a random subtour of the individual, and reverses that subtour at that location."""

		# Randomly introduce two cuts in the individual, and reverse that part of individual
		a, b = np.sort(np.random.permutation(np.arange(1, len(individual)))[:2])
		individual[a:b] = individual[a:b][::-1]

		fit = self.fitness(individual)
		individual = np.append(individual, fit)
		return individual

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
		individual = np.append(individual, fit)
		return individual

	def exchange_mutation(self, individual):
		"""Randomly swaps two entries in the cycle."""

		# idea: https://stackoverflow.com/questions/22847410/swap-two-values-in-a-numpy-array
		# more efficient to use numpy: See comments below this answer: https://stackoverflow.com/a/9755548/13331858
		# old method: # random.sample(range(1, len(individual)), 2)

		# Note: we only change from index 1: always keep index 0 equal to 0

		# Get two indices at which we will do a swap
		indices = np.random.permutation(np.arange(1, len(individual)))[:2]

		# Flip cities at those locations. Compute fitness and return
		individual[indices] = individual[np.flip(indices)]
		fit = self.fitness(individual)
		individual = np.append(individual, fit)
		return individual

	#####################
	# --- SELECTION --- #
	#####################

	# Selecting parents to generate the offspring

	def parents_selection(self):
		# Make sure the given option is implemented, otherwise use default
		if self.which_selection not in self.implemented_selection_operators:
			default = self.default_params_dict["which_selection"]
			print("Selection operator not recognized. Using default: %s" % default)
			self.which_selection = default

		if self.which_selection == "k tournament":
			return self.k_tournament_selection()

	def k_tournament_selection(self):
		"""Performs k tournament selection. Returns two parents, including fitness value, to perform recombination."""
		# Sample competitors
		competitors_1 = self.population[np.random.choice(self.lambdaa, self.tournament_size)]
		competitors_2 = self.population[np.random.choice(self.lambdaa, self.tournament_size)]
		# Sort competitors based on fitness
		competitors_1 = competitors_1[competitors_1[:, -1].argsort()]
		competitors_2 = competitors_2[competitors_2[:, -1].argsort()]
		return competitors_1[0], competitors_2[0]

	#######################
	# --- ELIMINATION --- #
	#######################

	def elimination(self):
		"""Choose the algorithm to perform the elimination phase"""

		# Make sure the given option is implemented, otherwise use default
		if self.which_elimination not in self.implemented_elimination_operators:
			default = self.default_params_dict["which_elimination"]
			print("Elimination operator not recognized. Using default: %s" % default)
			self.which_elimination = default

		if self.which_elimination == "lambda plus mu":
			self.lambda_plus_mu_elimination()

		elif self.which_elimination == "lambda and mu":
			self.lambda_and_mu_elimination()

		elif self.which_elimination == "round robin":
			self.round_robin_elimination()

		# Default:
		else:
			self.lambda_plus_mu_elimination()

	def round_robin_elimination(self):
		"""Performs the round robin elimination."""
		# Join original population and offspring
		all_individuals = np.concatenate((self.population, self.offspring))

		# Make an empty array to store the wins
		wins = np.zeros(self.lambdaa + self.mu)

		# Do pairwise tournaments for each individual
		for i in range(self.lambdaa + self.mu):
			current_individual = all_individuals[i]
			for _ in range(self.round_robin_size):
				# Randomly sample a competitor
				random_index = np.random.choice(self.lambdaa + self.mu)
				competitor = all_individuals[random_index]
				# Compare their fitness values, decide who wins
				if current_individual[-1] < competitor[-1]:
					# Current individual wins, add +1 to his win count
					wins[i] += 1
				else:
					# Competitor wins, add +1 to his win count
					wins[random_index] += 1

		# Save the lambda individuals with the highest win counts
		top_indices = np.argpartition(wins, -self.lambdaa)[-self.lambdaa:]
		self.population = all_individuals[top_indices]

	def lambda_plus_mu_elimination(self):
		"""Performs lambda + mu elimination"""
		# Join original population and offspring
		all_individuals = np.concatenate((self.population, self.offspring))
		# Sort based on their fitness value (last column of chromosome)
		all_individuals = all_individuals[all_individuals[:, -1].argsort()]
		# Make sure population size is again lambda for next iteration
		self.population = all_individuals[:self.lambdaa]

	def lambda_and_mu_elimination(self):
		"""Performs (lambda, mu) elimination. Offspring becomes new population, old population discarded."""
		# TODO - generalize to mu larger than lambda
		self.population = self.offspring

	########################
	# --- LOCAL SEARCH --- #
	########################

	def lso(self, individual, depth=1, sample_size=10):
		"""Performs a LSO. Individual contains genome and the fitness."""

		# If we disabled the use of LSO, do nothing
		if not self.use_lso:
			return individual

		# Base case, if depth of search is zero, do nothing
		if depth == 0:
			return individual

		# Save initial values, ie fitness and corresponding genome, to "best"
		best_fitness = individual[-1]
		individual = individual[:-1]
		best_individual = individual

		if self.which_lso == "2-opt":
			# Randomly permute the (i, j) pairs, from which neighbours are constructed, for a specified sample size
			sampled_indices = np.random.choice(len(self.two_opt_ij_pairs), size=sample_size, replace=False)
			sampled_neighbours = self.two_opt_ij_pairs[sampled_indices]
			for (i, j) in sampled_neighbours:
				# Get the next neighbour
				neighbour = self.two_opt(individual, i, j)
				# Compute its fitness, check if better than current best
				fit = self.fitness(neighbour)
				# If improvement is seen, save it
				if fit < best_fitness:
					best_fitness = fit
					best_individual = neighbour
					# If the pivot rule is greedy descent: return at first improvement
					if self.lso_pivot_rule == "greedy descent":
						# Append fitness at the end of the individual
						best_individual = np.append(best_individual, best_fitness)
						return self.lso(best_individual, depth=depth - 1)

			# If we reach the end of the while loop, this means that the current individual was the best one
			best_individual = np.append(best_individual, best_fitness)
			return self.lso(best_individual, depth=depth - 1)

	def two_opt(self, individual, i, j):
		"""Applies the two-opt operator once to a single individual. That is, it takes a subtour and reverts it."""
		# Get a subtour and reverse it
		individual[i:j] = individual[i:j][::-1]
		return individual

	#################
	# --- OTHER --- #
	#################

	def fitness(self, tour):
		"""Computes the fitness value of an individual"""
		fitness = 0

		# Make sure that the entries are seen as integers:
		tour = tour.astype('int')

		# For the 'body' of the tour:
		for i in range(len(tour) - 1):
			fitness += self.distance_matrix[tour[i], tour[i + 1]]

		# 'Close' the tour:
		fitness += self.distance_matrix[tour[-1], tour[0]]
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
			params_dict = {"which_mutation": mut, "which_recombination": rec}
			mytest = r0708518(params_dict)
			bestObjective, meanObjective, iterationCounter = mytest.optimize('./tour50.csv')
			with open('Data/Analysis_mutation_recombination.csv', 'a', newline='') as file:
				writer = csv.writer(file)
				# write data (Mutation, Recombination, Best, Mean, Iterations)
				data = [mut, rec, bestObjective, meanObjective, iterationCounter]
				writer.writerow(data)
			print("--------------------------------")


if __name__ == "__main__":
	params_dict = {"mu": 50, "number_of_iterations": 200, "random_perm_init_number": 0, "random_road_init_number": 40,
				   "greedy_road_init_number": 50, "nnb_road_init_number": 10, "use_lso": False}
	mytest = r0708518(params_dict)

	mytest.optimize('./tour50.csv')

	# """Analyzing the performance of mutation and crossover operators"""
	# analyze_operators()

	pass
