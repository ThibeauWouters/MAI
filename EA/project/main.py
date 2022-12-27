import Reporter
import numpy as np
import pandas as pd
import csv
import random
import matplotlib.pyplot as plt
import time
import sys
from math import floor
from numba import njit, jit
from os import listdir
from os.path import isfile, join

sys.setrecursionlimit(10000)

plt.rcParams['figure.figsize'] = (7, 4)
plt.rcParams['font.size']      = 8

# Fix random seeds:
# np.random.seed(0)
# random.seed(0)

heuristic_dict = {"tour50": 66540, "tour100": 103436, "tour250": 405662, "tour500": 78579,
                      "tour750": 134752, "tour1000": 75446}


###############################
# --- AUXILIARY FUNCTIONS --- #
###############################

def check_subset_numpy_arrays(a, b):
    """Returns true if a is a sub array of b"""
    b = np.unique1d(b)
    c = np.intersect1d(a, b)
    return np.size(c) == np.size(b)


def check_unique(individual):
    """Test whether all cities of an individual are different. If not, then there is a bug."""
    ok = (len(pd.unique(individual.genome)) == len(individual.genome))
    if not ok:
        diff = len(individual.genome) - len(pd.unique(individual.genome))
        print("Not OK: difference by %d " % diff)
        print(individual.genome)
        exit()


########################################
# --- EVOLUTIONARY ALGORITHM CLASS --- #
########################################


class CandidateSolution:
    """Class to store the genome and fitness of a candidate solution."""
    def __init__(self, genome, fitness):
        self.fitness = fitness
        self.genome = genome
        # Note - the idea was to use this class to get a self-adaptive algorithm, but we did not have time for this
        # self.alpha = ...


class r0708518:

    def __init__(self, user_params_dict=None):
        # Get the reporter
        self.reporter = Reporter.Reporter(self.__class__.__name__)

        # --- BASIC ---
        # Size of the population
        self.lambdaa = 75
        # Size of the offspring
        self.mu = 50

        # --- OPERATORS ---
        # Below, we store a string indicating which operators are to be used in the EA loop
        self.which_mutation      = "DM"
        self.which_recombination = "random"
        self.which_elimination   = "lambda plus mu"
        self.which_selection     = "k tournament"
        # Below, the functions that we implemented in the algorithm are saved, in order to check that the user
        # selected a method which is implemented, and if not, we can use the "default" choices in methods defined below
        # Note: "CX", "EX", "GROUP" crossovers are not used anymore as they are inefficient compared to others!
        self.implemented_mutation_operators      = ["EM", "DM", "SIM", "ISM", "IVM", "SM", "SDM", "variable", "random"]
        self.implemented_recombination_operators = ["PMX", "SCX", "OX", "OX2", "AX", "variable", "random"]
        self.implemented_selection_operators     = ["k tournament"]
        self.implemented_elimination_operators   = ["lambda plus mu", "age", "round robin", "k tournament", "ranking"]
        # For REC and MUT operators, whenever we choose "random" or "variable", these are the operators we use:
        self.available_rec_operators = ["PMX", "SCX", "OX", "OX2", "AX"]
        self.available_mut_operators = ["DM", "SM"]

        # --- Hyperparameters for these operators:
        # Tournament size for selection: varies during  run
        self.first_tournament_size = 4
        self.last_tournament_size  = 12
        self.tournament_size       = self.first_tournament_size
        # Tournament size for elimination:
        self.tournament_size_elimination = max(3, int(self.lambdaa // 40))
        # Size for round robin elimination:
        self.round_robin_size = 10
        # Mutation rate: varies during run
        self.first_alpha = 0.1
        self.last_alpha  = 0.01
        self.alpha       = self.first_alpha

        # --- INITIALIZATION ---
        # We provide percentages of population initialized with the methods specified below:
        # Random permutations (not advised)
        self.random_perm_init_fraction = 0
        # Greedy roads (in between nearest neighbours and legal initialization)
        self.greedy_road_init_fraction = 0.2
        # Nearest neighbours
        self.nnb_road_init_fraction    = 0.2
        # Remaining uses default: legal initialization

        # --- LSO ---
        # Specify which LSO operator we are going to use and which are implemented:
        self.which_lso       = "random"
        self.implemented_lso = ["2-opt", "adjacent swaps", "swaps", "insertions", "random"]
        # These are the operators we use when "random" is selected
        self.available_lso_operators = ["2-opt", "adjacent swaps", "swaps", "insertions"]
        # --- Hyperparameters
        # For each phase of the EA, specify the sample size and the depth
        # By default, we do nothing in recombination and mutation phase
        self.lso_init_sample_size = 100
        self.lso_init_depth       = 3
        self.lso_rec_sample_size  = 0
        self.lso_rec_depth        = 0
        self.lso_elim_sample_size = 100
        self.lso_elim_depth       = 3
        # Specify to which percentage of individuals we apply the LSO before elimination takes place:
        self.lso_elim_percentage  = 0.5
        # Specify when to apply the LSO - cooldown timer
        self.lso_cooldown = 10
        self.use_lso = True

        # --- STOPPING CONDITIONS ---
        # Specify max. number of iterations that the algorithm can do:
        self.number_of_iterations = 9999999999999999
        # Specify after how many iterations we stop, if no improvement in best fitness was seen during those iterations
        self.delta                = 0.1
        self.no_improvement_max   = 1000

        # --- FINAL STAGE ---
        # After 80% of allowed time, enter a 'final stage' where we focus on exploitation
        self.final_param_timer            = (4 / 5) * self.reporter.allowedTime
        # We are in the final stage until time runs out OR after certain number of iterations:
        self.final_number_of_iterations   = 300
        self.final_stage                  = False
        # Save the index at which we entered the 'final stage'
        self.final_stage_entered          = 0
        # We change the population and offspring size for the final stage
        self.final_stage_lambdaa          = self.lambdaa // 2
        self.final_stage_mu               = self.mu // 2

        # --- DIVERSITY PROMOTION ---
        # Specify which metric we use and which are implemented (only Hamming was implemented)
        self.which_metric        = "Hamming"
        self.implemented_metrics = ["Hamming"]
        # Specify how often we compute the diversity index (by default, we never compute it in the final version)
        self.diversity_check_cooldown = 99999999
        # Specify the sample size with which we compute the diversity index.
        self.diversity_sample_size    = self.lambdaa // 4
        self.diversity_index          = 0
        # If wanted, only use crowding whenever the diversity index drops below a threshold value (by default, always
        # use crowding)
        self.diversity_threshold      = 1.01
        # Boolean to indicate whether we want to use crowding during elimination phase:
        self.use_crowding             = True
        # Sample size used when performing crowding
        self.crowding_sample_size     = self.lambdaa//25

        # --- PROBABILITIES ---
        # Probability distributions that one can use to select items from a numpy array (see self.get_probabilities)
        # We only implemented the geometric distribution
        self.implemented_probability_distributions = ["geom"]
        # The parameter p of the geometric distribution varies over time. Used to e.g. increase selective pressure
        self.first_p = 0.1
        self.last_p  = 0.6
        self.p       = self.first_p

        # --- OTHER ---
        # Indicate whether or not we want to make a plot of the results:
        self.make_plot  = False
        self.which_plot = "combined"

        # --- Empty fields - to be determined/stored later on ----
        # Matrices that we are going to read, and the penalty value we are going to give to illegal roads:
        self.distance_matrix    = None
        # n = number of cities in TSP problem
        self.n                  = None
        self.connections_matrix = None
        self.penalty_value      = None
        # Population, offspring and their merger 'all_individuals'
        self.population         = None
        self.offspring          = None
        self.all_individuals    = None

        # Save these default parameters into a dictionary to look them up later on
        self.default_params_dict = {}
        for attribute, value in self.__dict__.items():
            if attribute == "default_params_dict":
                continue
            self.default_params_dict[attribute] = value
        default_param_keys = self.default_params_dict.keys()

        # Overwrite the (hyper)parameters set by the user:
        if user_params_dict is not None:
            for key in user_params_dict:
                if key not in default_param_keys:
                    print("User specified parameter %s is not a valid parameter." % str(key))
                else:
                    setattr(self, key, user_params_dict[key])

        # Make sure that the chosen operators are actually implemented, otherwise: use default one
        if self.which_recombination not in self.implemented_recombination_operators:
            default = self.default_params_dict["which_recombination"]
            print("Recombination operator not recognized. Using default: %s" % default)
            self.which_recombination = default

        if self.which_mutation not in self.implemented_mutation_operators:
            default = self.default_params_dict["which_mutation"]
            print("Mutation operator not recognized. Using default: %s" % default)
            self.which_mutation = default

        if self.which_elimination not in self.implemented_elimination_operators:
            default = self.default_params_dict["which_elimination"]
            print("Elimination operator not recognized. Using default: %s" % default)
            self.which_elimination = default

        if self.which_selection not in self.implemented_selection_operators:
            default = self.default_params_dict["which_selection"]
            print("Selection operator not recognized. Using default: %s" % default)
            self.which_selection = default

        if self.which_lso not in self.implemented_lso:
            default = self.default_params_dict["which_lso"]
            print("LSO operator not recognized. Using default: %s" % default)
            self.which_lso = default

        # Convert fractions for the initialization to actual numbers:
        print("------------------------------------")
        print("Population size: %d, offspring size: %d" % (self.lambdaa, self.mu))
        total_so_far = 0
        self.random_perm_init_number = int(round(self.random_perm_init_fraction * self.lambdaa))
        total_so_far += self.random_perm_init_number
        self.greedy_road_init_number = int(round(self.greedy_road_init_fraction * self.lambdaa))
        total_so_far += self.greedy_road_init_number
        self.nnb_road_init_number = int(round(self.nnb_road_init_fraction * self.lambdaa))
        total_so_far += self.nnb_road_init_number
        # Fill the remainder with "default" choice:
        self.random_road_init_number = int(round(self.lambdaa - total_so_far))
        print("Initialized with %d random permutations, %d greedy roads, %d NN and %d legal roads." % (
            self.random_perm_init_number, self.greedy_road_init_number, self.nnb_road_init_number,
            self.random_road_init_number))

        # Print which implementations were chosen by the user:
        print("------------------------------------")
        print("Mutation operator:      %s" % self.which_mutation)
        print("Recombination operator: %s" % self.which_recombination)
        print("Elimination operator:   %s" % self.which_elimination)
        print("Selection operator:     %s" % self.which_selection)
        print("LSO operator:           %s" % self.which_lso)

        if self.which_recombination in ["variable", "random"]:
            print("Available REC operators: ", self.available_rec_operators)
            # Start off with uniform probabilities
            self.recombination_probabilities = 1 / len(self.available_rec_operators) * np.ones(
                len(self.available_rec_operators))

        if self.which_mutation in ["variable", "random"]:
            # Start off with uniform probabilities
            print("Available MUT operators: ", self.available_mut_operators)
            self.mutation_probabilities = 1 / len(self.available_mut_operators) * np.ones(
                len(self.available_mut_operators))

        if self.which_lso == "random":
            # Start off with uniform probabilities
            print("Available LSO operators: ", self.available_lso_operators)

        # For age-based elimination, lambda has to be equal to mu as we do a replacement.
        if self.which_elimination == "age":
            smallest = min(self.lambdaa, self.mu)
            self.lambdaa = smallest
            self.mu = smallest

    def optimize(self, filename):
        """The evolutionary algorithm's main loop"""

        # --- PREPARATIONS
        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Replace inf values with just a penalty term
        distance_matrix = np.where(distance_matrix == float('inf'), -1, distance_matrix)
        # Get max value, use it to construct a penalty term, and replace again
        max_distance = np.max(distance_matrix.flatten())
        penalty_value = 2 * max_distance
        distance_matrix = np.where(distance_matrix == -1, penalty_value, distance_matrix)
        # Also apply the penalty for roads going from a city to itself (diagonal terms)
        distance_matrix = np.where(distance_matrix == 0, penalty_value, distance_matrix)

        # Save the penalty term for convenience for later on
        self.penalty_value = penalty_value
        # Save distance matrix as a field
        self.distance_matrix = distance_matrix
        # Save n, the size of the problem (number of cities)
        self.n = np.size(self.distance_matrix[0])

        # Also construct the connections matrix (see report for details)
        self.construct_connections_matrix()

        # Check how much time we have
        timeLeft = self.reporter.allowedTime
        # Initialize iteration counters
        global_counter         = 1
        no_improvement_counter = 0
        # Initialize variables keeping track of the convergence
        current_best  = 0
        previous_best = 0

        # Create empty lists to be filled with values to report progress
        mean_fit_values = []
        best_fit_values = []

        # To check the diversity, save it and plot it at the end.
        diversities = []

        # Allocate memory for population and offspring
        self.population = np.empty(self.lambdaa, dtype='object')
        self.offspring  = np.empty(self.mu, dtype='object')

        # To keep track of variable probabilities for crossover and mutation (if desired)
        delta_f_crossover = [[] for i in range(len(self.available_rec_operators))]
        delta_f_mutation = [[] for i in range(len(self.available_mut_operators))]

        # INITIALIZATION
        self.initialize()

        """This is where the main evolutionary algorithm loop comes:"""
        while global_counter < self.number_of_iterations:
            # Increment the counter for the number of iterations
            global_counter += 1

            # Save the current timestamp (based on reporter) - used to get the values of variable hyperparameters
            self.t = self.reporter.allowedTime - timeLeft
            # If we are later than 80% of allowed time, we enter the so-called "final stage"
            # Also (for tour 50 mostly) if no improvement was seen
            if (self.t > self.final_param_timer and not self.final_stage) or (no_improvement_counter >= self.no_improvement_max):
                # Save the index at which we entered final stage (used in plots)
                self.final_stage_entered = global_counter
                # Perform at most X iterations from now on, or until time runs out:
                self.number_of_iterations = global_counter + self.final_number_of_iterations
                print("Entering final stage of the algorithm (%d s or %d iterations)" % (int(round(timeLeft)), self.final_number_of_iterations))
                # Disable no improvement counter
                no_improvement_counter = -99999999999999
                # No longer do crowding - we don't care about diversity anymore, go full exploitation
                self.use_crowding = False
                self.final_stage  = True
                # Make sure the crowding mechanism never gets activated again:
                self.diversity_threshold = -0.01
                # Crank up the LSO
                self.lso_elim_depth       += 2
                self.lso_elim_sample_size += 100
                self.lso_elim_percentage   = 1
                # Change the popualtion and offspring size
                self.lambdaa = self.final_stage_lambdaa
                self.mu      = self.final_stage_mu

            # Adapt the variable hyperparameters (linear interpolation between 2 specified values:
            self.alpha = self.interpolate_parameter(self.first_alpha, self.last_alpha)
            self.tournament_size = self.interpolate_parameter(self.first_tournament_size,
                                                              self.last_tournament_size)
            self.tournament_size = int(floor(self.tournament_size))
            if self.which_recombination == "variable" or self.which_mutation == "variable":
                self.p = self.interpolate_parameter(self.first_p, self.last_p)

            # Update fitness values for reporting the progress:
            previous_best  = current_best
            fitnesses      = [ind.fitness for ind in self.population]
            mean_objective = np.mean(fitnesses)
            best_index     = np.argmin(fitnesses)
            best_candidate = self.population[best_index]
            best_solution  = best_candidate.genome
            best_objective = best_candidate.fitness
            current_best   = best_objective

            # If not enough improvement was seen this iteration, count up
            if abs(previous_best - current_best) < self.delta:
                no_improvement_counter += 1
            else:
                # If enough improvement was seen: reset counter to zero
                no_improvement_counter = 0

            # Add fitness values to a list to plot them later on:
            mean_fit_values.append(mean_objective)
            best_fit_values.append(best_objective)

            # Keep track of diversity: (with certain cooldown, and if we're not in the final stage)
            if global_counter % self.diversity_check_cooldown == 0 and not self.final_stage:
                self.diversity_index = self.measure_diversity()
                diversities.append(self.diversity_index)

                # Check whether or not we want to promote diversity (in elimination phase)
                if self.diversity_index < self.diversity_threshold:
                    self.use_crowding = True
                else:
                    # If population is diverse enough, stop using crowding until next diversity measurement
                    self.use_crowding = False

            # CROSSOVER

            for i in range(self.mu):
                # Select two parents
                parent1, parent2 = self.parents_selection()
                # Note: recombination and mutation ONLY work with genome, NOT the fitness at the end
                # here, "child" has no actual fitness and is only refering to the genome!
                if self.which_recombination == "variable":
                    # If we are using a variable crossover operator, we also return the index of used operator
                    index, child = self.recombination(parent1, parent2)
                    # Save info about this recombination in the appropriate list
                    difference = self.difference_fitness(parent1.genome, child.genome)
                    delta_f_crossover[index] = np.append(delta_f_crossover[index], difference)
                else:
                    child = self.recombination(parent1, parent2)
                # Perform LSO on computed child if wanted (by default: no LSO at this stage)
                child.genome = self.lso(child.genome, depth=self.lso_rec_depth, sample_size=self.lso_rec_sample_size)
                self.offspring[i] = child

            # Merge all individuals for next part
            # Join original population and offspring
            self.all_individuals = np.concatenate((self.population, self.offspring))
            # Sort population based on fitness
            self.all_individuals = self.sort_fitness(self.all_individuals)

            # MUTATE

            for i in range(1, len(self.all_individuals)):
                individual = self.all_individuals[i]
                # "Flip a coin" to determine whether or not we mutate this individual
                if not self.final_stage and random.uniform(0, 1) <= self.alpha:
                    if self.which_mutation == "variable":
                        # If we are using a variable mutation operator, we also return the index of used operator
                        # and save the difference in fitness, in order to rank the operators later on
                        index, new = self.mutation(individual)
                        # Save info about this recombination in the appropriate list
                        difference = self.difference_fitness(individual.genome, new.genome)
                        individual = new
                        delta_f_mutation[index] = np.append(delta_f_mutation[index], difference)
                    else:
                        individual = self.mutation(individual)
                # Now, compute the fitness and save into offspring array
                individual.fitness = self.fitness(individual.genome)
                self.all_individuals[i] = individual

            # LSO

            # We only perform the lso to the top individuals (based on fitness)
            number_of_lso = int(round(self.lso_elim_percentage * len(self.all_individuals)))
            for i in range(number_of_lso):
                old, old_fitness = self.all_individuals[i].genome, self.all_individuals[i].fitness
                better_genome = self.lso(old, depth=self.lso_elim_depth, sample_size=self.lso_elim_sample_size)
                better_fitness = self.efficient_fitness(better_genome, old, old_fitness)
                better_individual = CandidateSolution(better_genome, better_fitness)
                self.all_individuals[i] = better_individual

            # ELIMINATION

            # Make a new, empty population
            self.population = np.empty(self.lambdaa, dtype='object')
            # Sort based on fitness (used in some elimination schemes + for elitism)
            self.all_individuals = self.sort_fitness(self.all_individuals)
            self.elimination()


            # For testing: see advancements:
            # Activate the LSO after certain number of iterations
            if (global_counter % self.lso_cooldown) == 0:
                self.use_lso = True
            else:
                self.use_lso = False

            # If we are using variable crossover/mutation, update their probabilities
            if self.which_recombination == "variable":
                average_improvements = np.array([np.mean(deltas) for deltas in delta_f_crossover])
                sort_indices = np.argsort(average_improvements)
                probabilities = self.get_probabilities(len(self.available_rec_operators), param=self.p)
                self.recombination_probabilities = probabilities[sort_indices]

            if self.which_mutation == "variable":
                average_improvements = np.array([np.mean(deltas) for deltas in delta_f_mutation])
                sort_indices = np.argsort(average_improvements)
                probabilities = self.get_probabilities(len(self.available_mut_operators), param=self.p)
                self.mutation_probabilities = probabilities[sort_indices]

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(mean_objective, best_objective, best_solution)

            # Our code gets killed if time is up!
            if timeLeft < 0:
                break

        # --- Post processing ---
        print("------------------------------------")
        print("We did %d iterations" % global_counter)
        print("The final best    fitness value was {:,}".format(round(best_objective)).replace(',', ' '))
        print("The final average fitness value was {:,}".format(round(mean_objective)).replace(',', ' '))

        if self.make_plot:
            if self.which_plot == "separate":
                make_fitness_plot(mean_fit_values, best_fit_values, filename, start=10,
                                  final_stage_entered=self.final_stage_entered)
                make_fitness_plot(mean_fit_values, best_fit_values, filename, plot_mean=False, plot_name="best_fitness",
                                  final_stage_entered=self.final_stage_entered)
            # By default, plot combined plot
            else:
                make_combined_fitness_plot(mean_fit_values, best_fit_values, filename, plot_name="test",
                                           final_stage_entered=self.final_stage_entered)

            # Plot diversity (if we measured it during the run)
            if len(diversities) > 0:
                self.make_diversity_plot(diversities, filename)

        # Return the best fitness value & best solution
        # TODO - change this again
        # return bestObjective, bestSolution
        return best_objective, mean_objective, global_counter

    ##########################
    # --- INITIALIZATION --- #
    ##########################

    @jit(forceobj=True)
    def initialize(self):
        """Initializes the population using several techniques and concatenates them."""

        # Empty list to save all the subpopulations we are going to construct
        subpopulations = []

        # Initialize by random permutation:
        if self.random_perm_init_number != 0:
            subpopulations.append(self.random_perm_initialize(self.random_perm_init_number))

        # Initialize by random selection of roads, which avoids the illegal roads:
        if self.random_road_init_number != 0:
            subpopulations.append(self.road_initialize(self.random_road_init_number, method="random"))

        # Initialize by randomly but greedily selecting the roads:
        if self.greedy_road_init_number != 0:
            subpopulations.append(self.road_initialize(self.greedy_road_init_number, method="greedy"))

        # Initialize by nearest neighbours, but starting location is random:
        if self.nnb_road_init_number != 0:
            subpopulations.append(self.road_initialize(self.nnb_road_init_number, method="nearest nb"))

        # Append all the subpopulations together in one big population array, save it
        self.population = np.concatenate(subpopulations)

    def construct_connections_matrix(self):
        """Creates a matrix from the distance matrix of the problem, where each row i specifies which cities one can
        reach from i, avoiding the roads that give inf (or equivalently, large penalty values). Used in constructing
        initial populations in a more informed manner. In each rows, these possible cities are ordered based on their
        cost, ranging from low to high. The result is saved as instance attribute. """

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

    def construct_tour(self, current, final_index, final_city, method="random"):
        """Starting from an individual which is only constructed partially, continue constructing a full individual,
        taking into account the possible connections (no infs in distance matrix) and possibly also take into account
        the magnitude of the cost of taking a road. This function is called recursively, in order to be able to
        backtrack in case we get "stuck", i.e., we end up in a city connected to cities which were already selected
        in the construction of the individual. Parameters: current: individual, numpy array of size n,
        possibly containing -1 for cities which were not assigned yet final_index: the index at which the final city
        in individual was assigned in a previous call final_city: the city last assigned in the previous function
        call. """

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
                probabilities = self.get_probabilities(len(possible_connections))
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
                current_city = possible_connections[0]
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
        """Initializes an individual using legal roads only. Can do random, greedy or NN (see report for details)."""

        # Check that the provided method is a valid one, otherwise resort to default
        if method not in ["random", "greedy", "nearest nb"]:
            print("Initialization method not recognized. Defaulting to random.")
            method = "random"

        # Generate a "subpopulation" based on how many are generated using this method
        result = []

        # Construct "number" amount of individuals
        counter = 0
        while counter < number:
            # Initialize a new individual, start filling at zero
            individual = np.full(self.n, -1)
            # Pick a random starting point
            starting_point = np.random.choice(self.n)
            individual[0] = starting_point
            # Call construct_tour, which recursively makes a road
            # Note: here, individual is a TSP tour, not yet a CandidateSolution
            individual = self.construct_tour(individual, 0, starting_point, method=method)
            if individual is not None:
                # Make sure that the genome starts with city 0 because of our convention
                idzero = np.argwhere(individual == 0)[0][0]
                individual = np.roll(individual, -idzero)
                # Improve individual with LSO operator (nothing changed if lso_init_depth = 0)
                individual = self.lso(individual, depth=self.lso_init_depth, sample_size=self.lso_init_sample_size)
                # Compute its fitness
                fit = self.fitness(individual)
                # Construct a new instance of candidate solution
                candidate = CandidateSolution(individual, fit)
                result.append(candidate)
                counter += 1
            else:
                # In case there is a bug, print something to the screen
                print("I might be stuck here...")

        return result

    def random_perm_initialize(self, number=10):
        """Initialize individuals by a random permutation of the cities. Quick and easy, but may end up selecting
        illegal roads, both inf roads as well as roads connecting a city to itself which may mislead the algorithm
        since those have a low cost. Not recommended to use this initialization method."""

        result = np.empty((number, self.n))
        for i in range(number):
            # Random permutation, but always start in 'city 0'
            individual = np.zeros(self.n)
            # Permute the remaining cities 1, ..., n:
            random_permutation = np.random.permutation(self.n - 1) + 1
            # Save both parts in the individual
            individual = random_permutation
            # Compute the fitness and append to individual
            fit = self.fitness(random_permutation)
            # individual = np.append(individual, fitness)
            candidate = CandidateSolution(individual, fit)
            result[i] = candidate

        return result

    #########################
    # --- RECOMBINATION --- #
    #########################

    def recombination(self, parent1, parent2):
        """Performs the chosen recombination operator."""

        # Special case: variable operator:
        if self.which_recombination == "variable":
            return self.variable_crossover(parent1, parent2)

        # Specify which operator was selected. If random, select a random one
        if self.which_recombination == "random":
            which = np.random.choice(self.available_rec_operators)
        else:
            which = self.which_recombination

        if which == "PMX":
            new_genome = self.partially_mapped_crossover(parent1, parent2)
        elif which == "SCX":
            new_genome = self.single_cycle_crossover(parent1, parent2)
        elif which == "OX":
            new_genome = self.order_crossover(parent1, parent2)
        elif which == "OX2":
            new_genome = self.order_based_crossover(parent1, parent2)
        elif which == "AX":
            new_genome = self.alternating_crossover(parent1, parent2)

        # Deprecated functions
        # elif self.which_recombination == "CX":
        # 	new_genome = self.cycle_crossover(parent1, parent2)
        # elif self.which_recombination == "EX":
        # 	new_genome = self.edge_crossover(parent1, parent2)
        # elif self.which_recombination == "GROUP":
        # 	new_genome = self.group_recombination(parent1, parent2)

        # Default choice: OX2
        else:
            new_genome = self.order_based_crossover(parent1, parent2)

        # Instantiate new CandidateSolution object, assign arbitrary fitness value (to be computed later)
        child = CandidateSolution(new_genome, 99999999999999)

        return child

    def variable_crossover(self, parent1, parent2):
        """Chooses a crossover operator at random, but taking into account their performance. Returns index as well."""

        # Flip a coin, take into account fitness improvements for this
        i = np.random.choice(len(self.available_rec_operators), p=self.recombination_probabilities)
        # Get the chosen operator from this
        chosen_operator = self.available_rec_operators[i]

        # Do the crossover (generate child) and return also the index.
        if chosen_operator == "PMX":
            return i, self.partially_mapped_crossover(parent1, parent2)
        elif chosen_operator == "SCX":
            return i, self.single_cycle_crossover(parent1, parent2)
        elif chosen_operator == "OX":
            return i, self.order_crossover(parent1, parent2)
        elif chosen_operator == "OX2":
            return i, self.order_based_crossover(parent1, parent2)
        elif chosen_operator == "AX":
            return i, self.alternating_crossover(parent1, parent2)

        # Default choice (as a failsafe): OX2
        else:
            return 0, self.order_based_crossover(parent1, parent2)

    @jit(forceobj=True)
    def order_based_crossover(self, parent1, parent2):
        """(OX2) Performs the order based crossover operator."""

        tour1 = parent1.genome
        tour2 = parent2.genome

        # Initialize child with -1 everywhere
        child = np.full(len(tour1), -1)

        # Generate a sample of indices of length k
        k = np.random.choice(np.arange(2, self.n - 2))
        indices = np.sort(np.random.choice([i for i in range(1, len(tour1))], size=k, replace=False))

        # Get cities at those positions at tour 2
        cities = tour2[indices]
        # Look up their indices in tour1
        new_indices = np.in1d(tour1, cities)
        # Copy 'cities' in preserved order at selected indices
        child[new_indices] = cities
        # Copy tour1 at places which are not these new indices (note: '~'is negation), and np.in1d returns
        # a mask containing true and false values, not integers as indices!
        child[~new_indices] = tour1[~new_indices]

        return child

    def alternating_crossover(self, parent1, parent2):
        """(AX) Performs the alternating position crossover operator."""

        tour1 = parent1.genome
        tour2 = parent2.genome

        child = np.empty(2 * len(tour1), dtype=tour1.dtype)
        child[0::2] = tour1
        child[1::2] = tour2
        child = pd.unique(child)
        return child

    @DeprecationWarning
    def edge_crossover(self, parent1, parent2):
        """(EX) Performs the edge crossover operator."""

        tour1 = parent1.genome
        tour2 = parent2.genome

        # STEP 1: construct the edge table

        # Initialize the edge table
        edge_table = [np.array([], dtype='int') for i in range(len(tour1))]

        # Do numpy roll on tours to easily get the edges
        roll_p1_left = np.roll(tour1, 1)
        roll_p1_right = np.roll(tour1, -1)

        roll_p2_left = np.roll(tour2, 1)
        roll_p2_right = np.roll(tour2, -1)

        for i in range(len(tour1)):
            # Look at edges of allele at i in tour 1
            index = tour1[i]
            edge_table[index] = np.concatenate([edge_table[index], np.array([roll_p1_left[i], roll_p1_right[i]])],
                                               dtype='int')

            # Same for tour2
            index = tour2[i]
            edge_table[index] = np.concatenate([edge_table[index], np.array([roll_p2_left[i], roll_p2_right[i]])],
                                               dtype='int')

        # STEP 2: do the loop

        # First element: choose 0 to guarantee constraint is met
        child = np.full(len(tour1), -1)
        child[0] = 0
        current_element = 0
        unassigned = [i for i in range(1, len(tour1))]

        for i in range(1, len(tour1)):
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
                current_element = random.sample(unassigned, 1)[0]
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
        return child

    def order_crossover(self, parent1, parent2):
        """(OX) Performs the order crossover operator."""

        tour1 = parent1.genome
        tour2 = parent2.genome

        # Introduce two random cut points to get subtour of tour1
        a, b = np.sort(np.random.permutation(np.arange(1, len(tour1)))[:2])

        # Find the remaining cities, and use their order as given in second tour
        ids = np.in1d(tour2, tour1[a:b])
        remainder = tour2[~ids]

        # Add these two together, and make sure 0 is first element
        child = np.concatenate([tour1[a:b], remainder])
        idzero = np.argwhere(child == 0)[0][0]
        child = np.roll(child, -idzero)
        return child

    def single_cycle_crossover(self, parent1, parent2):
        """(SCX) Performs the cycle crossover, but only performs one such cycle."""

        tour1 = parent1.genome
        tour2 = parent2.genome

        # Initialize child, make sure to start at zero
        child = np.full(len(tour1), -1)
        child[0] = 0

        # Initialize information for a 'cycle'
        index = 1
        child[index] = tour1[index]
        first_index = index

        while True:
            # Look at the allele with the same position in P2
            allele = tour2[index]

            # Go to the position with the same allele in P1
            next_index = np.argwhere(tour1 == allele)[0][0]

            # Add this allele to the cycle
            child[next_index] = tour1[next_index]

            index = next_index
            # In case we completed the cycle, DON'T start a next 'cycle' simply return (copy tour2)
            if index == first_index:
                child = np.where(child == -1, tour2, child)

                return child

    def cycle_crossover(self, parent1, parent2):
        """(CX) Performs the cycle crossover, but only performs one such cycle."""

        tour1 = parent1.genome
        tour2 = parent2.genome

        # Initialize child, make sure to start at zero
        child = np.full(len(tour1), -1)
        child[0] = 0

        # Initialize information for a 'cycle'
        index = 1
        first_index = index
        value_tour = tour1
        which_value_tour = "1"
        child[index] = value_tour[index]

        while True:
            # Look at the allele with the same position in P2
            allele = tour2[index]

            # Go to the position with the same allele in P1
            next_index = np.argwhere(tour1 == allele)[0][0]

            # Add this allele to the cycle
            child[next_index] = value_tour[next_index]

            index = next_index
            # In case we completed the cycle, start the next 'cycle' -- swap order of tours
            if index == first_index:
                if -1 not in child:
                    return child
                else:
                    # Start a new cycle
                    index = np.argwhere(child == -1)[0][0]
                    first_index = index
                    if which_value_tour == "1":
                        value_tour = tour2
                        which_value_tour = "2"
                    else:
                        value_tour = tour1
                        which_value_tour = "1"

    def partially_mapped_crossover(self, parent1, parent2):
        """ (PMX) Implements the partially mapped crossover."""

        tour1 = parent1.genome
        tour2 = parent2.genome

        # Initialize two children we are going to create
        child = np.full(len(tour1), -1)
        child[0] = 0
        # Generate cut points a and b: these are two random indices, sorted
        a, b = np.sort(np.random.permutation(np.arange(1, len(tour1)))[:2])
        # Get the cut from the 2nd tour
        cut = tour2[a:b]
        # "Cross the cut"
        child[a:b] = cut
        # Check which indices remain to be assigned
        remaining_indices = np.where(child == -1)[0]
        # Iterate over the remaining entries
        for i in remaining_indices:
            # Get the value we WISH to fill in:
            value = tour1[i]
            # If this element, or any we will now find, was already copied from tour 2:
            while value in cut:
                # look up index of this element in tour2
                index = np.where(tour2 == value)  # [0][0]
                # Then use the mapping cut1 <-> cut2 to get new value. Check if new value also in cut2 (while loop)
                value = tour1[index]
            # if not, just use the value of tour 1
            child[i] = value

        return child

    @DeprecationWarning
    def group_recombination(self, parent1, parent2):
        """Copies the intersection of two parents. Distributes the remaining cities of first parent to child after
        permutation. First implementation of recombination algorithm. """

        tour1 = parent1.genome
        tour2 = parent2.genome

        # Child starts off with the intersection of the parents. Fill remaining with -1 (to recognize it later).
        # Since cities start in 0, this constraint will automatically copy over to child.
        child = np.where(tour1 == tour2, tour1, -1)
        # Get the indices of child which were not assigned yet.
        leftover_indices = np.where(child == -1)[0]
        # Get the cities that appear in one of the tours, permute them
        leftover_cities_permuted = np.random.permutation(tour1[leftover_indices])
        # Store permuted cities in the child
        child[leftover_indices] = leftover_cities_permuted
        return child

    ####################
    # --- MUTATION --- #
    ####################

    def mutation(self, individual):
        """Performs the chosen mutation (chosen at initialization of self)"""

        # Special case: variable operator:
        if self.which_mutation == "variable":
            return self.variable_mutation(individual)

        if self.which_mutation == "random":
            which = np.random.choice(self.available_mut_operators)
        else:
            which = self.which_mutation

        if which == "EM":
            new_genome = self.exchange_mutation(individual)
        elif which == "DM":
            new_genome = self.displacement_mutation(individual)
        elif which == "SIM":
            new_genome = self.simple_inversion_mutation(individual)
        elif which == "ISM":
            new_genome = self.insertion_mutation(individual)
        elif which == "IVM":
            new_genome = self.inversion_mutation(individual)
        elif which == "SM":
            new_genome = self.scramble_mutation(individual)
        elif which == "SDM":
            new_genome = self.scrambled_displacement_mutation(individual)
        else:
            new_genome = self.simple_inversion_mutation(individual)

        # Instantiate new CandidateSolution object, assign arbitrary fitness value (compute later on)
        new_individual = CandidateSolution(new_genome, 99999999999999)

        return new_individual

    def variable_mutation(self, individual):

        # Flip a coin, take into account fitness improvements for this
        i = np.random.choice(len(self.available_mut_operators), p=self.mutation_probabilities)
        # Get the chosen operator from this
        chosen_operator = self.available_mut_operators[i]

        if chosen_operator == "EM":
            return i, self.exchange_mutation(individual)
        elif chosen_operator == "DM":
            return i, self.displacement_mutation(individual)
        elif chosen_operator == "SIM":
            return i, self.simple_inversion_mutation(individual)
        elif chosen_operator == "ISM":
            return i, self.insertion_mutation(individual)
        elif chosen_operator == "IVM":
            return i, self.inversion_mutation(individual)
        elif chosen_operator == "SM":
            return i, self.scramble_mutation(individual)
        elif chosen_operator == "SDM":
            return i, self.scrambled_displacement_mutation(individual)

    def scrambled_displacement_mutation(self, individual):
        """(SDM) Takes a random subtour and inserts it, in 'scrambled' (permuted) order, at a random place. Note that
        this extends both the scramble and displacement mutation, where the subtour gets scrambled. """

        tour = individual.genome

        # Randomly introduce two cuts in the tour
        a, b = np.sort(np.random.permutation(np.arange(1, len(tour)))[:2])
        subtour = tour[a:b]
        # Delete the subtour from tour
        tour = np.delete(tour, np.arange(a, b, 1))
        # Insert it at a random position, but reverse the order
        insertion_point = random.randint(1, len(tour))
        tour = np.insert(tour, insertion_point, np.random.permutation(subtour))
        return tour

    def scramble_mutation(self, individual):
        """(SM) Takes a random subtour of the individual, and permutes that subtour at that location."""

        tour = individual.genome

        # Randomly introduce two cuts in the tour, and reverse that part of tour
        a, b = np.sort(np.random.permutation(np.arange(1, len(tour)))[:2])
        tour[a:b] = np.random.permutation(tour[a:b])
        return tour

    def inversion_mutation(self, individual):
        """(IVM) Takes a random subtour and inserts it, in reversed order, at a random place."""

        tour = individual.genome

        # Randomly introduce two cuts in the tour
        a, b = np.sort(np.random.permutation(np.arange(1, len(tour)))[:2])
        subtour = tour[a:b]
        # Delete the subtour from tour
        tour = np.delete(tour, np.arange(a, b, 1))
        # Insert it at a random position, but reverse the order
        insertion_point = random.randint(1, len(tour))
        tour = np.insert(tour, insertion_point, subtour[::-1])
        return tour

    def insertion_mutation(self, individual):
        """(ISM) Takes a random city and inserts it at a random place. Note that this is a special case of
        displacement mutation, where the subtour has length 1. """

        tour = individual.genome

        # Randomly take two different index positions
        a, b = np.sort(np.random.permutation(np.arange(1, len(tour)))[:2])
        subtour = tour[a]
        # Delete the subtour from tour
        tour = np.delete(tour, a)
        # Insert it at a random position
        tour = np.insert(tour, b, subtour)
        return tour

    def simple_inversion_mutation(self, individual):
        """(SIM) Takes a random subtour of the individual, and reverses that subtour at that location."""

        tour = individual.genome

        # Randomly introduce two cuts in the tour, and reverse that part of tour
        a, b = np.sort(np.random.permutation(np.arange(1, len(tour)))[:2])
        tour[a:b] = tour[a:b][::-1]
        return tour

    @jit(forceobj=True)
    def displacement_mutation(self, individual):
        """(DM) Cuts a subtour of the individual, and places it in a random place"""

        tour = individual.genome

        # Randomly introduce two cuts in the tour
        a, b = np.sort(np.random.permutation(np.arange(1, len(tour)))[:2])
        subtour = tour[a:b]
        # Delete the subtour from tour
        tour = np.delete(tour, np.arange(a, b, 1))
        # Insert it at a random position
        insertion_point = random.randint(1, len(tour))
        tour = np.insert(tour, insertion_point, subtour)
        return tour

    def exchange_mutation(self, individual):
        """Randomly swaps two entries in the cycle."""

        tour = individual.genome

        # Get two indices at which we will do a swap
        indices = np.random.permutation(np.arange(1, len(tour)))[:2]

        # Flip cities at those locations. Compute fitness and return
        tour[indices] = tour[np.flip(indices)]
        return tour

    #####################
    # --- SELECTION --- #
    #####################

    def parents_selection(self):
        """Selects two parents in the population. Only k tournament implemented."""

        if self.which_selection == "k tournament":
            # Select two parents using k tournament
            p1 = self.k_tournament(self.population, self.tournament_size)
            p2 = self.k_tournament(self.population, self.tournament_size)
            return p1, p2

    def k_tournament(self, individuals, k):
        """Performs k tournament selection. Returns winner of the tournament."""
        # Sample competitors
        pop_size = len(individuals)
        # Failsafe: adapt tournament size if the number of individuals is small
        if k >= pop_size:
            k = pop_size // 2
        competitors = individuals[np.random.choice(pop_size, k, replace=False)]
        # Sort competitors based on fitness, return best one
        competitors = self.sort_fitness(competitors)
        return competitors[0]

    #######################
    # --- ELIMINATION --- #
    #######################

    def elimination(self, which=None):
        """Choose the algorithm to perform the elimination phase"""

        # Specify which operator was selected
        if which is None:
            which = self.which_elimination

        # Perform the chosen operator:
        if which == "lambda plus mu":
            self.lambda_plus_mu_elimination(self.all_individuals)
        elif which == "age":
            self.age_elimination()
        elif which == "k tournament":
            self.k_tournament_elimination(self.all_individuals)
        elif which == "round robin":
            self.round_robin_elimination(self.all_individuals)
        elif which == "ranking":
            self.ranking_elimination(self.all_individuals)

    def ranking_elimination(self, all_individuals):
        """Eliminates based on ranking. Note that all_individuals must be sorted."""

        # Elitism: make sure the best individual gets selected:
        self.population[0] = all_individuals[0]
        # the remainder of the new population gets chosen based on their "rank" (determined by fitness).
        # Note: here we make use of the fact that the all_individuals array is sorted based on their fitness.
        probs = self.get_probabilities(len(all_individuals) - 1, param=self.p)
        self.population[1:] = np.random.choice(all_individuals[1:], size=self.lambdaa - 1, replace=False, p=probs)

        return

    def k_tournament_elimination(self, all_individuals):
        """Performs k tournament to construct the offspring from the population."""

        # Sample size for the k tournaments (may get adjusted later on to avoid problems)
        tournament_sample_size = self.tournament_size_elimination
        # Crowding sample size (may get adjusted later on to avoid problems)
        crowding_sample_size = self.crowding_sample_size

        counter = 0
        while counter < self.lambdaa:
            if counter == 0:
                # Elitism: keep the best individual: first one in array (we sorted based on fitness)
                self.population[0] = all_individuals[0]
                # Delete that individual
                all_individuals = np.delete(all_individuals, 0, axis=0)
                counter += 1

            # Check termination condition: if number of already selected plus remaining is lambda, then "fill spots"
            if (len(all_individuals) + counter) == self.lambdaa:
                self.population[counter:] = all_individuals
                return

            # If we are still constructing,
            # Failsafe: prevent a crash if length of all individuals drops below provided sample size:
            if len(all_individuals) < tournament_sample_size:
                tournament_sample_size = len(all_individuals) // 2
            # Same for sampling for the crowding
            if len(all_individuals) < crowding_sample_size:
                crowding_sample_size = len(all_individuals) // 2
            # Do k tournament the sneaky way: since all_individuals is sorted on fitness, just sample indices
            sampled_indices = np.sort(
                np.random.choice(len(all_individuals), size=tournament_sample_size, replace=False))
            # Because we sorted all individuals based on fitness, the earlier the index the better the fitness!
            best_index = sampled_indices[0]
            chosen = all_individuals[best_index]
            all_individuals = np.delete(all_individuals, best_index, axis=0)
            # Hence add that individual to the population
            self.population[counter] = chosen

            if self.use_crowding:
                # If we do crowding, delete the individual closest to chosen one from the population
                # Search for the one closest to chosen among remaining individuals, but sample them:
                sampled_indices = np.random.choice(len(all_individuals), size=crowding_sample_size, replace=False)
                sampled_individuals = all_individuals[sampled_indices]
                # Get the closest in distance
                distances = [self.hamming_distance(chosen.genome, individual.genome) for individual in
                             sampled_individuals]
                # Get index of the one that has closest distance...
                best_index_distances = np.argmin(distances)
                # ... then find the index in all_individuals that corresponded to this
                best_index = sampled_indices[best_index_distances]
                # Delete it from the population
                all_individuals = np.delete(all_individuals, best_index, axis=0)
            # Increment counter to go to next round
            counter += 1

        return

    def round_robin_elimination(self, all_individuals):
        """Performs the round robin elimination."""

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
                if current_individual.genome < competitor.genome:
                    # Current individual wins, add +1 to his win count
                    wins[i] += 1
                else:
                    # Competitor wins, add +1 to his win count
                    wins[random_index] += 1

        # Save the lambda individuals with the highest win counts
        top_indices = np.argpartition(wins, -self.lambdaa)[-self.lambdaa:]
        self.population = all_individuals[top_indices]

    def lambda_plus_mu_elimination(self, all_individuals):
        """Performs lambda + mu elimination"""

        # Keep lambda best individuals. Note: individuals must be sorted!
        # If we are not going to use crowding, immediately return
        if not self.use_crowding:
            self.population = all_individuals[:self.lambdaa]
            return

        # If we are going to use crowding, first define the sample size (can be changed later on in the function)
        sample_size = self.crowding_sample_size
        counter = 0
        while counter < self.lambdaa:
            # Check termination condition: if number of already selected plus remaining is lambda, then "fill spots"
            if (len(all_individuals) + counter) == self.lambdaa:
                self.population[counter:] = all_individuals
                return
            # Next in line is the one with best fitness, i.e. first in remaining individuals
            chosen = all_individuals[0]
            # Add the chosen individual to the population under construction
            self.population[counter] = chosen
            all_individuals = np.delete(all_individuals, 0, axis=0)
            # Failsafe: prevent a crash if length of all individuals drops below provided sample size:
            if len(all_individuals) < sample_size:
                sample_size = 1
            # Search for the one closest to chosen among remaining individuals, but sample them:
            sampled_indices = np.random.choice(len(all_individuals), size=sample_size, replace=False)
            sampled_individuals = all_individuals[sampled_indices]
            # Get the closest in distance
            distances = [self.hamming_distance(chosen.genome, individual.genome) for individual in sampled_individuals]
            # Get index of the one that has closest distance...
            best_index_distances = np.argmin(distances)
            # ... then find the index in all_individuals that corresponded to this
            best_index = sampled_indices[best_index_distances]
            # Delete it from the population
            all_individuals = np.delete(all_individuals, best_index, axis=0)
            # Increment counter to go to next round
            counter += 1

    def age_elimination(self):
        """Performs age elimination. Offspring becomes new population, old population completely discarded."""
        self.population = self.offspring

    ########################
    # --- LOCAL SEARCH --- #
    ########################

    def lso(self, individual, depth=1, sample_size=10):
        """Performs a LSO. Individual contains genome but NOT the fitness."""

        # If we disabled the use of LSO, do nothing
        if not self.use_lso:
            return individual

        best_individual = individual
        if self.which_lso != "random":
            which = self.which_lso
        else:
            which = np.random.choice(self.available_lso_operators)

        if which == "2-opt":
            while depth > 0:
                sampled_indices = [np.sort(np.random.choice(np.arange(1, self.n), size=2, replace=False)) for i in
                                   range(sample_size)]

                for (i, j) in sampled_indices:
                    # Get the next neighbour, and compute the difference in fitness
                    neighbour = self.two_opt(individual, i, j)
                    difference = self.difference_fitness(individual, neighbour)
                    # If improvement is seen, save it
                    if difference < 0:
                        best_individual = neighbour
                        break
                # At the end of for loop (either completed or broken), reduce the depth
                depth -= 1
            return best_individual

        if which == "swaps":
            # Note: this has complexity order n, so check (up until) the full neighbourhood of each individual
            while depth > 0:
                sampled_indices = [np.sort(np.random.choice(np.arange(1, self.n), size=2, replace=False)) for i in
                                   range(sample_size)]

                for (i, j) in sampled_indices:
                    # Get the next neighbour, and compute the difference in fitness
                    # Do an "adjacent swap", i.e. swap entry with the next entry
                    neighbour = self.swap(individual, i, j)
                    difference = self.difference_fitness(individual, neighbour)
                    # If improvement is seen, save it
                    if difference < 0:
                        best_individual = neighbour
                        break
                # At the end of for loop (either completed or broken), reduce the depth
                depth -= 1
            return best_individual

        if which == "adjacent swaps":
            # Note: this has complexity order n, so check (up until) the full neighbourhood of each individual
            while depth > 0:
                # If the sample size is larger than the number of adjacent swaps we can do, limit it
                if sample_size > self.n - 2:
                    sample_size = self.n - 2
                sampled_indices = np.random.choice(np.arange(1, self.n - 1), size=sample_size, replace=False)

                for i in sampled_indices:
                    # Get the next neighbour, and compute the difference in fitness
                    # Do an "adjacent swap", i.e. swap entry with the next entry
                    neighbour = self.swap(individual, i, i + 1)
                    difference = self.difference_fitness(individual, neighbour)
                    # If improvement is seen, save it
                    if difference < 0:
                        best_individual = neighbour
                        break
                # At the end of for loop (either completed or broken), reduce the depth
                depth -= 1
            return best_individual

        if which == "insertions":
            # Note: this has complexity order n, so check (up until) the full neighbourhood of each individual
            while depth > 0:
                # If the sample size is larger than the number of adjacent swaps we can do, limit it
                if sample_size > self.n - 2:
                    sample_size = self.n - 2
                sampled_indices = [np.sort(np.random.choice(np.arange(1, self.n), size=2, replace=False)) for i in
                                   range(sample_size)]

                for (i, j) in sampled_indices:
                    # Get the next neighbour, and compute the difference in fitness
                    # Do an "adjacent swap", i.e. swap entry with the next entry
                    neighbour = self.insert(individual, i, j)
                    difference = self.difference_fitness(individual, neighbour)
                    # If improvement is seen, save it
                    if difference < 0:
                        best_individual = neighbour
                        break
                # At the end of for loop (either completed or broken), reduce the depth
                depth -= 1
            return best_individual

    @jit(forceobj=True)
    def insert(self, individual, i, j):
        """Takes element at index i, inserts it at index j"""

        # Copy the individual
        clone = np.copy(individual)
        # Delete the chosen city and save it
        city = clone[i]
        clone = np.delete(clone, i)
        # Insert it at chosen position j
        clone = np.insert(clone, j, city)
        return clone

    @jit(forceobj=True)
    def swap(self, individual, i, j):
        clone = np.copy(individual)
        temp = clone[i]
        clone[i] = clone[j]
        clone[j] = temp
        return clone

    @jit(forceobj=True)
    def two_opt(self, individual, i, j):
        """Applies the two-opt operator once to a single individual. That is, it takes a subtour and reverts it."""
        # Get a subtour and reverse it
        clone = np.copy(individual)
        clone[i:j] = individual[i:j][::-1]
        return clone

    #######################
    # DIVERSITY PROMOTION #
    #######################

    def measure_distance(self, first, second):
        """Measures the distance between two individuals of the population."""

        if self.which_metric not in self.implemented_metrics:
            self.which_metric = self.default_params_dict["which_metric"]
            print("Metric not recognized. Defaulting to: ", self.which_metric)

        if self.which_metric == "Hamming":
            return self.hamming_distance(first, second)

    def hamming_distance(self, first, second):
        return np.sum(np.where(first != second, 1, 0))

    @jit(forceobj=True)
    def measure_diversity(self, sample_size=None):
        """Measures the diversity of the population"""

        # Failsafe: if we have small lambda, adapt the sample size
        if sample_size is None:
            sample_size = self.diversity_sample_size
        if self.lambdaa < sample_size:
            sample_size = round(self.lambdaa / 2)

        # Sample indices randomly, to get a sample of the population from this
        sample_indices = np.random.choice(self.lambdaa, size=sample_size, replace=False)
        sample = self.population[sample_indices]

        counter = 0
        total_distance = 0

        for i in range(len(sample) - 1):
            for j in range(i + 1, len(sample)):
                # Note: the distance is divided by the problem size, such that the concept of "diversity" does not
                # depend on the problem size. This allows us to design diversity promotion techniques valid for all
                # sizes of the TSP.
                total_distance += (self.measure_distance(sample[i].genome, sample[j].genome)) / self.n
                counter += 1

        return total_distance / counter

    ###################
    # --- FITNESS --- #
    ###################

    def sort_fitness(self, individuals):
        """Sorts a given group of individuals based on their fitness values"""
        fitnesses = np.array([ind.fitness for ind in individuals], dtype='float64')
        sort_ind = np.argsort(fitnesses)
        return individuals[sort_ind]

    @jit(forceobj=True)
    def efficient_fitness(self, new, old=None, old_fitness=None):
        """Implements a more efficient version of fitness calculation. old, new: TSP tours. old_fitness: fitness
        value of old tour (if provided) """

        # In case we don't compare two genomes: just compute the fitness
        if new is None:
            return self.fitness(new)

        # If we compare genomes, check which version is more efficient
        else:
            # If there is too much difference between genomes, compute fitness old way
            if self.hamming_distance(old, new) > self.n // 2:
                return self.fitness(new)
            # If there is little difference between genomes, compute by comparison
            else:
                difference = self.difference_fitness(old, new)
                return old_fitness + difference

    @jit(forceobj=True)
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

    @jit(forceobj=True)
    def difference_fitness(self, old, new):
        """Computes the difference in fitness value between two genomes as efficiently as possible."""

        # Initialize difference variable:
        difference = 0

        # Get the indices where the difference in the two genomes is
        ind = np.where(old != new)[0]
        # print("Number of diff ind in diff fitness: ", len(ind))

        if len(ind) == 0:
            return 0

        # Go over these indices:
        for i in range(len(ind)):
            # If i = 0, take the first and get the connection before it
            if i == 0:
                difference -= self.distance_matrix[old[ind[0] - 1], old[ind[0]]]
                difference += self.distance_matrix[new[ind[0] - 1], new[ind[0]]]
            # For all other indices, check if:
            # (1) we simply "continue" the subtour
            elif ind[i] == ind[i - 1] + 1:
                difference -= self.distance_matrix[old[ind[i] - 1], old[ind[i]]]
                difference += self.distance_matrix[new[ind[i] - 1], new[ind[i]]]
            # (2) there was a "jump" in the indices:
            else:
                # make sure to 'close' the previous subtour
                difference -= self.distance_matrix[old[ind[i - 1]], old[ind[i - 1] + 1]]
                difference += self.distance_matrix[new[ind[i - 1]], new[ind[i - 1] + 1]]
                # and start on the new one:
                difference -= self.distance_matrix[old[ind[i] - 1], old[ind[i]]]
                difference += self.distance_matrix[new[ind[i] - 1], new[ind[i]]]

        # At the end of the for loop, we still need to close the very final subtour:
        difference -= self.distance_matrix[old[ind[-1]], old[(ind[-1] + 1) % self.n]]
        difference += self.distance_matrix[new[ind[-1]], new[(ind[-1] + 1) % self.n]]

        return difference

    #######################
    # --- HYPERPARAMS --- #
    #######################

    def get_probabilities(self, size, distr="geom", param=0.4):
        """Generates a numpy array of probabilities to select the entries of that numpy array. values: numpy array
        containing the values for which we want to get probabilities distr: string indiciating which distribution
        should be used for getting the probabilities. Default is the geometric distribution, which is a discrete
        version of the exponential distribution. """

        if distr not in self.implemented_probability_distributions:
            distr = "geom"

        # Geometric distribution:
        if distr == "geom":
            # Set the number of failures before the first success (k)
            k = np.array([i for i in range(size)])
            # Calculate the probabilities for each value of k using the geometric probability distribution
            probabilities = param * (1 - param) ** (k - 1)
            return probabilities / np.sum(probabilities)

    def interpolate_parameter(self, first_value, last_value):
        """Performs a linear interpolation between two extreme values for a parameter value"""

        if self.t > self.final_param_timer:
            return last_value
        else:
            value = (last_value - first_value) / self.final_param_timer * self.t + first_value
            return value

    def save_hyperparams(self, save_name):
        """Saves the tunable parameters into a CSV file"""

        param_names = ['lambdaa', 'mu', 'which_mutation', 'which_recombination', 'which_elimination', 'which_selection',
                       'first_tournament_size', 'last_tournament_size',
                       'tournament_size', 'first_alpha', 'last_alpha', 'alpha', 'round_robin_size',
                       'random_perm_init_fraction',
                       'greedy_road_init_fraction', 'nnb_road_init_fraction', 'which_lso',
                       'lso_init_sample_size', 'lso_init_depth', 'lso_rec_sample_size', 'lso_rec_depth',
                       'lso_mut_sample_size', 'lso_mut_depth', 'lso_elim_sample_size', 'lso_elim_depth', 'lso_cooldown',
                       'number_of_iterations', 'delta', 'no_improvement_max', 'which_metric',
                       'diversity_check_cooldown', 'diversity_threshold',
                       'tournament_size_elimination', 'final_stage_lambdaa', 'final_stage_mu', 'diversity_sample_size',
                       'crowding_sample_size',
                       'lso_elim_percentage']

        param_dict = {}
        param_vals = []
        for key in param_names:
            param_dict[key] = self.__getattribute__(key)
            param_vals.append(self.__getattribute__(key))

        print(param_dict)
        save_name = "Hyperparams/" + save_name
        if save_name[-4:] != ".csv":
            save_name = save_name + ".csv"
        array = [param_names, param_vals]
        array = np.transpose(array)
        np.savetxt(save_name, array, fmt="%s", delimiter=",")

        loaded_values = pd.read_csv(save_name, header=None)
        print(loaded_values)

    ###########################
    # --- POST PROCESSING --- #
    ###########################

    def make_diversity_plot(self, diversities, filename):
        cooldown = self.diversity_check_cooldown
        # --- Plot the diversity observed during the run
        # TODO - delete this at the end
        # start = len(diversities) // 20
        start = 0
        remainder = diversities[start:]
        xt = [start + i * cooldown for i in range(len(remainder))]
        plt.plot(xt, remainder, '--o', ms=4, color='red', label="Diversity")
        # plt.axhline(0, color='black')
        # plt.axhline(1, color='black')
        plt.axhline(self.diversity_threshold, ls='--', color='black', label="Threshold")
        plt.ylim(0, 1)
        plt.grid()
        plt.legend()
        plt.xlabel('Iteration step')
        plt.ylabel('Average Hamming distance (sampled)')
        plt.title('Diversity during algorithm for ' + str(filename))
        # plot_name = "plot_mut_" + self.which_mutation + "_rec_" + self.which_recombination
        plot_name = "plot_diversities"
        # Save the plots (as PNG and PDF)
        plt.savefig('Plots/' + plot_name + '.png', bbox_inches='tight')
        plt.savefig('Plots/' + plot_name + '.pdf', bbox_inches='tight')
        plt.close()


def make_combined_fitness_plot(mean_fit_values, best_fit_values, filename, plot_name="plot_test_run",
                      start=0, final_stage_entered=0, seconds=None):

    print("Plotting . . . ")
    # Instantiate auxiliary variables
    tour_name = filename
    heuristic_value = None

    # Get the heuristic value (in case it is one of the benchmark problems)

    for k in heuristic_dict:
        if k + ".csv" in filename:
            tour_name = k
            heuristic_value = heuristic_dict[k]
            break

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    if seconds is None:
        x = [i for i in range(len(mean_fit_values))]
    else:
        x = seconds

    # --- AX1: mean AND best:
    ax1.plot(x[start:], mean_fit_values[start:], '--o', ms=3, color='red',
                 label="Mean")
    start=0
    ax1.plot(x[start:], best_fit_values[start:], '--o', ms=3, color='blue',
             label="Best")
    ax1.axhline(heuristic_value, ls='--', lw=1.25, alpha=0.9, color='black', label="Heuristic")

    # Show when we killed diversity and went full for exploitation (in case we plot after a run)
    if final_stage_entered > 0:
        ax1.axvline(final_stage_entered, color='black')
    ax1.grid()
    ax1.legend()
    ax1.set_ylabel('Fitness')
    factor = 1.5
    if tour_name == "tour1000":
        factor = 2.5
    ax1.set_ylim(np.min(best_fit_values[-1] - heuristic_value/7), factor*heuristic_value)
    ax1.set_title('TSP for ' + tour_name)

    # --- AX2: best fitness only
    ax2.plot(x[start:], best_fit_values[start:], '--o', ms=3, color='blue',
             label="Best")
    ax2.axhline(heuristic_value, ls='--', color='black', alpha=0.9, label="Heuristic", lw=1.25)

    # Show when we killed diversity and went full for exploitation (in case we plot after a run)
    if final_stage_entered > 0:
        ax2.axvline(final_stage_entered, color='black')
    ax2.grid()
    if seconds is None:
        ax2.set_xlabel('Iteration step')
    else:
        ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Fitness')
    
    # Save the plots (as PNG and PDF)
    plt.savefig('Plots/' + plot_name + '_combined.png', bbox_inches='tight')
    plt.savefig('Plots/' + plot_name + '_combined.pdf', bbox_inches='tight')
    plt.close()


def make_fitness_plot(mean_fit_values, best_fit_values, filename, plot_mean=True, plot_name="plot_test_run",
                      start=10, final_stage_entered=0):
    """Makes plots of the mean and best fitness values."""

    print("Plotting . . . ")
    # Instantiate auxiliary variables
    tour_name = filename
    heuristic_value = None

    # We have the option either to plot the mean or not
    if plot_mean:
        plt.plot([i for i in range(start, len(mean_fit_values))], mean_fit_values[start:], '--o', ms=3, color='red',
                 label="Mean")
    plt.plot([i for i in range(start, len(best_fit_values))], best_fit_values[start:], '--o', ms=3, color='blue',
             label="Best")

    # Plot heuristic value as well:
    for k in heuristic_dict:
        if k + '.csv' in filename:
            tour_name = k
            heuristic_value = heuristic_dict[k]
            plt.axhline(heuristic_dict[k], ls='--', color='black', alpha=0.7, label="Heuristic")
            break

    # Show when we killed diversity and went full for exploitation
    if final_stage_entered > 0:
        plt.axvline(final_stage_entered, color='black', alpha=0.7)
    plt.grid()
    plt.legend()
    plt.xlabel('Iteration step')
    plt.ylabel('Fitness')
    plt.title('TSP for ' + tour_name)

    # Save the plots (as PNG and PDF)
    plt.savefig('Plots/' + plot_name + '.png', bbox_inches='tight')
    plt.savefig('Plots/' + plot_name + '.pdf', bbox_inches='tight')
    plt.close()


def load(filename):
    """Load a Reporter CSV file."""

    # Read in the CSV file
    data = np.genfromtxt(filename, delimiter=",")

    # Save as separate fields
    iterations, elapsed, mean, best = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    cycles = data[:, 4:-1]

    return iterations, elapsed, mean, best, cycles


def load_and_plot(filename, save_name="test_", seconds=None):
    """Load a Reporter CSV file and process it, plotting the fitness values."""

    # Load the values
    iterations, elapsed, mean, best, cycles = load(filename)

    # Give the tour a name, based on the size of the problem:
    name = "tour" + str(len(cycles[0])) + ".csv"

    # --- Note - we only use the combined fitness plot here, and use iterations instead of seconds
    # Plot the best and mean fitness
    # make_fitness_plot(mean, best, name, plot_name="Project plots/" + save_name + "best_mean")
    # Plot the best fitness only
    # make_fitness_plot(mean, best, name, plot_name="Project plots/" + save_name + "best", plot_mean=False)
    # Combined plot, iterations:
    make_combined_fitness_plot(mean, best, name, plot_name="Project plots/" + save_name + "iterations")
    # Combined plot, time in seconds:
    # make_combined_fitness_plot(mean, best, name, plot_name="Project plots/" + save_name + "seconds", seconds=elapsed)

    # print("Best value for this run was %0.2f" % best[-1])


def analyze_runs(which_tour="tour50"):
    """Loads in all runs performed for a tour, and analyzes them."""

    # Get the relevant Reporter CSV files
    folder_name = "./Runs/" + which_tour + "/"
    all_filenames = [(folder_name + f) for f in listdir(folder_name) if isfile(join(folder_name, f))]

    # Get the heuristic value (in case it is one of the benchmark problems)
    heuristic_dict = {"tour50": 66540, "tour100": 103436, "tour250": 405662, "tour500": 78579,
                      "tour750": 134752, "tour1000": 75446}
    for k in heuristic_dict:
        if k == which_tour:
            heuristic_value = heuristic_dict[k]
            break

    number_of_runs = len(all_filenames)
    n = int(which_tour[4:])
    print("We have %d runs of %s" % (number_of_runs, which_tour))

    # Get empty lists ready to be filled:
    all_iterations = np.zeros(len(all_filenames))
    all_elapsed    = np.zeros(len(all_filenames))
    all_best       = np.zeros(len(all_filenames))
    all_mean       = np.zeros(len(all_filenames))
    all_cycles     = np.zeros((len(all_filenames), n))

    # Load the values
    for i, filename in enumerate(all_filenames):
        iterations, elapsed, mean, best, cycles = load(filename)
        # Save at the appropriate places:
        all_iterations[i] = iterations[-1]
        all_elapsed[i]    = elapsed[-1]
        all_best[i]       = best[-1]
        all_mean[i]       = mean[-1]
        all_cycles[i]     = cycles[-1]

    # Report on the average and stdv
    avg_best = int(np.mean(all_best))
    avg_mean = int(np.mean(all_mean))
    std_best = int(np.std(all_best))
    std_mean = int(np.std(all_mean))
    print(u"The best fitness value is %d \u00B1 %d" % (avg_best, std_best))
    # print(u"The mean fitness value is %d \u00B1 %d" % (avg_mean, std_mean))

    # Check how often we beat the heuristic:
    number_of_success = np.sum(np.where(all_best < heuristic_value, 1, 0))
    percentage = (number_of_success/len(all_best))*100
    print(f"We've beaten the heuristic in {number_of_success} out of {number_of_runs} runs ({percentage} % succes rate).")

    # Report on the best of the best:
    best_index = np.argmin(all_best)
    best_best = all_best[best_index]
    best_cycle = all_cycles[best_index]
    best_file = all_filenames[best_index]
    
    print("The best fitness value observed was %0.2f, for file %s" %(best_best, best_file))
    print("This best tour was:")
    print(best_cycle)
    
    # Plot the best run as well:
    load_and_plot(best_file, save_name="best_" + which_tour + "_run_")

    # For tour 50, also make a histogram
    if which_tour == "tour50":
        # Make the histogram
        plt.hist(all_best)

        # Make pretty
        plt.grid()
        plt.xlabel("Best fitness")
        plt.title("Histogram of best fitness values for tour50")
        plt.savefig('Plots/histogram_t50_runs.png', bbox_inches='tight')
        plt.savefig('Plots/histogram_t50_runs.pdf', bbox_inches='tight')
        plt.close()
    print("Done")

def make_boxplots():
    """Create boxplots of the runs for t100, t500 and t1000 to compare the EAs performance."""
    bp_list = []
    # Create the boxplots
    colors = {"tour100": "red", "tour500": "blue", "tour1000": "black"}

    keys = ["tour100", "tour500", "tour1000"]
    for i, key in enumerate(keys):
        folder_name = "./Runs/" + key + "/"
        all_filenames = [(folder_name + f) for f in listdir(folder_name) if isfile(join(folder_name, f))]

        # Get empty lists ready to be filled:
        all_best = np.zeros(len(all_filenames))

        # Load the values
        for j, filename in enumerate(all_filenames):
            iterations, elapsed, mean, best, cycles = load(filename)
            # Save at the appropriate places:
            all_best[j] = best[-1]

        # Plot the boxplot
        w = 0.5
        bp = plt.boxplot(all_best, positions=[i], widths=w, patch_artist=True)
        # Plot the heuristic value:
        heuristic_value = heuristic_dict[key]
        plt.axhline(heuristic_value, ls='--', color=colors[key], alpha=0.9, label=key)
        bp_list.append(bp)

    # Now, make fancy:
    # plt.axhline(heuristic_value, color='black', alpha=0.75)
    # plt.title("Best values (average of 10)")
    plt.xticks([i for i in range(len(keys))], labels=keys)
    plt.xlabel("Tour size")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(axis='y')
    # Make beautiful boxplots:
    # Set the line width and color of the boxplots
    for bp in bp_list:
        for element in ["whiskers", "boxes", "caps", "medians"]:
            plt.setp(bp[element], linewidth=2)
        for patch in bp["boxes"]:
            patch.set(facecolor="white")
    # plt.show()
    plt.savefig("Plots/Project plots/boxplots_tours.png", bbox_inches='tight')
    plt.savefig("Plots/Project plots/boxplots_tours.pdf", bbox_inches='tight')
    plt.close()


def run_tour50():
    """Keeps on solving the tour 50 problem as much as possible."""
    while True:
        print("++++++++++++++++++++++++++++++")
        params_dict = {"lso_cooldown": 200}
        tsp = r0708518(params_dict)
        start = time.time()
        tsp.optimize('./tour50.csv')
        end = time.time()
        time_spent = abs(end - start)
        # Save the time spent on this problem, according to the Reporter of TSP:
        outFile = open("runtimes_t50.csv", "a")
        outFile.write(str(time_spent)+"\n")
        outFile.close()


if __name__ == "__main__":

    # --- Collect runs on tour50 for the histogram:
    # run_tour50()

    # --- Make plots for a collection of runs:
    # analyze_runs("tour1000")

    # --- Run the TSP solver on a single instance
    # mytest = r0708518()
    # mytest.optimize('./tour50.csv')

    # --- Check performance across tours
    # make_boxplots()

    # --- Plot an interesting run, to discuss in the report:
    load_and_plot("Runs/best_t1000_v1.csv", save_name="example_benefit_final_stage")

    pass
