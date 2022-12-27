# Auxiliary file to analyze stuff

import Reporter
import numpy as np
import pandas as pd
import csv
import main
import random
from random import sample
import matplotlib.pyplot as plt
# import time

plt.rcParams['figure.figsize'] = (15, 5)


def analyze_operators():
    """Estimates the performance of selected crossover and mutation operators through experiments on tour50."""

    all_recombinations = ["PMX", "SCX", "OX", "OX2", "AX"]

    for rec in all_recombinations:
        counter = 0
        # Do the optimization run 10 times
        while counter < 10:
            print("Counter: ", counter)
            params_dict = {"which_recombination": rec}
            mytest = main.r0708518(params_dict)
            bestObjective, meanObjective, iterationCounter = mytest.optimize('./tour750.csv')
            with open('Data/Analysis_mutation_recombination_v3.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                # write data (Mutation, Recombination, Best, Mean, Iterations)
                data = [rec, bestObjective, meanObjective, iterationCounter]
                writer.writerow(data)
            print("+++++++++++++++++++++++++++")
            counter += 1

def discuss_results_boxplot():
    """Shows most important features of experiments of REC operators in a boxplot."""

    heuristic_value = 134752
    df = pd.read_csv("Data/Analysis_mutation_recombination_v3.csv")

    operator_names = pd.unique(df["rec"])
    # Get the values:
    best       = df["best"]
    iterations = df["iterations"]

    # Get the average values of this:
    average_fit = []
    std_fit = []
    average_iterations = []
    std_iterations = []

    for key in operator_names:
        slice = df[df["rec"] == key]
        average_fit.append(np.mean(slice["best"]))
        average_iterations.append(np.mean(slice["iterations"]))
        # Std:
        std_fit.append(np.std(slice["best"]))
        std_iterations.append(np.std(slice["iterations"]))

    # Plot fitness values
    bp_list = []
    # Create the boxplots
    for i, key in enumerate(operator_names):
        data = df[df["rec"] == key]["best"]
        bp = plt.boxplot(data, positions=[i], widths=0.25, patch_artist=True)
        bp_list.append(bp)
    # Set the xlabels
    plt.axhline(heuristic_value, color='black', alpha=0.75)
    plt.title("Best values (average of 10)")
    plt.xticks([i for i in range(len(operator_names))], labels=operator_names)
    plt.xlabel("Recombination operator")
    plt.ylabel("Fitness")
    plt.grid(axis='y')
    # Make beautiful boxplots:
    # Set the line width and color of the boxplots
    for bp in bp_list:
        for element in ["whiskers", "boxes",  "caps", "medians"]:
            plt.setp(bp[element], linewidth=2)
        for patch in bp["boxes"]:
            patch.set(facecolor="white")
    # plt.show()
    plt.savefig("Plots/Analysis_best_value_v3_boxplot.png", bbox_inches='tight')
    plt.close()

    # Plot iteration values
    bp_list = []
    # Create the boxplots
    for i, key in enumerate(operator_names):
        data = df[df["rec"] == key]["iterations"]
        bp = plt.boxplot(data, positions=[i], widths=0.25, patch_artist=True)
        bp_list.append(bp)
    # Set the xlabels
    plt.title("Number of iterations (average of 10)")
    plt.xticks([i for i in range(len(operator_names))], labels=operator_names)
    plt.xlabel("Recombination operator")
    plt.ylabel("Number of iterations")
    plt.grid(axis='y')
    # Make beautiful boxplots:
    # Set the line width and color of the boxplots
    for bp in bp_list:
        for element in ["whiskers", "boxes", "caps", "medians"]:
            plt.setp(bp[element], linewidth=2)
        for patch in bp["boxes"]:
            patch.set(facecolor="white")
    # plt.show()
    plt.savefig("Plots/Analysis_iterations_v3_boxplot.png", bbox_inches='tight')
    plt.close()


def discuss_performance_rec_operators():
    heuristic_value = 134752
    df = pd.read_csv("Data/Analysis_mutation_recombination_v3.csv")

    operator_names = pd.unique(df["rec"])
    # Get the values:
    best = df["best"]
    iterations = df["iterations"]

    # Get the average values of this:
    average_fit = []
    std_fit = []
    average_iterations = []
    std_iterations = []

    for key in operator_names:
        slice = df[df["rec"] == key]
        average_fit.append(np.mean(slice["best"]))
        average_iterations.append(np.mean(slice["iterations"]))
        # Std:
        std_fit.append(np.std(slice["best"]))
        std_iterations.append(np.std(slice["iterations"]))

    # Plot the figure
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax1.set_title("Performance of recombination operators (average of 10)")
    bp_list = []
    # Create the boxplots for fitness values
    for i, key in enumerate(operator_names):
        data = df[df["rec"] == key]["best"]
        bp = ax1.boxplot(data, positions=[i], widths=0.25, patch_artist=True)
        bp_list.append(bp)
    # Set the xlabels
    ax1.axhline(heuristic_value, color='black', alpha=0.75)
    ax1.set_ylabel("Best fitness")
    ax1.grid(axis='y')
    # Set the line width and color of the boxplots
    for bp in bp_list:
        for element in ["whiskers", "boxes", "caps", "medians"]:
            plt.setp(bp[element], linewidth=2)
        for patch in bp["boxes"]:
            patch.set(facecolor="white")

    # Next: the number of iterations
    plt.subplot(212)
    # Plot iterations values
    x = [i for i in range(len(average_iterations))]
    plt.errorbar(x, average_iterations,  yerr=std_iterations, color='red', barsabove=True, fmt='o', markersize=6, capsize=6, linewidth=3)
    # plt.title("Best values (average of 10)")
    plt.xticks([i for i in range(len(operator_names))], labels=operator_names)
    plt.xlabel("Recombination operator")
    plt.ylabel("Number of iterations")
    plt.grid()
    # plt.show()
    plt.savefig("Plots/Analysis_best_and_iteration_v3.png", bbox_inches='tight')
    plt.close()


def discuss_results1d():
    """Shows most important features of experiments of REC operators."""

    heuristic_value = 134752
    df = pd.read_csv("Data/Analysis_mutation_recombination_v3.csv")

    operator_names = pd.unique(df["rec"])
    # Get the values:
    best       = df["best"]
    iterations = df["iterations"]

    # Get the average values of this:
    average_fit = []
    std_fit = []
    average_iterations = []
    std_iterations = []

    for key in operator_names:
        slice = df[df["rec"] == key]
        average_fit.append(np.mean(slice["best"]))
        average_iterations.append(np.mean(slice["iterations"]))
        # Std:
        std_fit.append(np.std(slice["best"]))
        std_iterations.append(np.std(slice["iterations"]))

    # Plot fitness values
    x = [i for i in range(len(average_fit))]
    plt.errorbar(x, average_fit,  yerr=std_fit, color='red', barsabove=True)
    plt.axhline(heuristic_value, color='black', alpha=0.75)
    plt.title("Best values (average of 10)")
    plt.xticks([i for i in range(len(operator_names))], labels=operator_names)
    plt.xlabel("Recombination operator")
    plt.ylabel("Fitness")
    plt.grid()
    # plt.show()
    plt.savefig("Plots/Analysis_best_value_v3.png", bbox_inches='tight')
    plt.close()

    # Plot iterations values
    x = [i for i in range(len(average_iterations))]
    plt.errorbar(x, average_iterations,  yerr=std_iterations, color='red', barsabove=True)
    plt.title("Best values (average of 10)")
    plt.xticks([i for i in range(len(operator_names))], labels=operator_names)
    plt.xlabel("Recombination operator")
    plt.ylabel("Number of iterations")
    plt.grid()
    # plt.show()
    plt.savefig("Plots/Analysis_iterations_v3.png", bbox_inches='tight')
    plt.close()

def discuss_results2d():
    """Shows most important features of experiments of REC and MUT operators."""

    df = pd.read_csv("Data/Analysis_mutation_recombination_v3.csv")
    print(df)


    # Get operator names
    all_mutations = pd.unique(df["mutation"])
    all_recombinations = pd.unique(df["recombination"])

    # Get the values
    best_values      = df["best"].values.reshape((len(all_mutations), len(all_recombinations)))
    mean_values      = df["mean"].values.reshape((len(all_mutations), len(all_recombinations)))
    iteration_values = df["iterations"].values.reshape((len(all_mutations), len(all_recombinations)))
    # times            = df["time"].values.reshape((len(all_mutations), len(all_recombinations)))

    # Plot best values
    plt.imshow(best_values)
    plt.colorbar()
    plt.title("Analysis of best values")
    plt.xticks([i for i in range(len(all_recombinations))], labels=all_recombinations)
    plt.xlabel("Recombination operator")
    plt.yticks([i for i in range(len(all_mutations))], labels=all_mutations)
    plt.ylabel("Mutation operator")
    # plt.show()
    plt.savefig("Plots/Analysis_best_value.png", bbox_inches='tight')
    plt.close()

    # Plot mean values
    plt.imshow(mean_values)
    plt.colorbar()
    plt.title("Analysis of mean values")
    plt.xticks([i for i in range(len(all_recombinations))], labels=all_recombinations)
    plt.xlabel("Recombination operator")
    plt.yticks([i for i in range(len(all_mutations))], labels=all_mutations)
    plt.ylabel("Mutation operator")
    # plt.show()
    plt.savefig("Plots/Analysis_mean_value.png", bbox_inches='tight')
    plt.close()

    # Plot iterations
    plt.imshow(iteration_values)
    plt.colorbar()
    plt.title("Analysis of iteration values")
    plt.xticks([i for i in range(len(all_recombinations))], labels=all_recombinations)
    plt.xlabel("Recombination operator")
    plt.yticks([i for i in range(len(all_mutations))], labels=all_mutations)
    plt.ylabel("Mutation operator")
    # plt.show()
    plt.savefig("Plots/Analysis_iterations.png", bbox_inches='tight')
    plt.close()

    # What are the best ones?
    print("---------------------")
    id = df["best"].argmin()
    value = df["best"].min()
    mut = df["mutation"][id]
    rec = df["recombination"][id]

    print(f"Best of best: mutation: {mut} recombination: {rec} (value: {round(value)})")

    print("---------------------")
    id = df["mean"].argmin()
    value = df["mean"].min()
    mut = df["mutation"][id]
    rec = df["recombination"][id]

    print(f"Best of mean: mutation: {mut} recombination: {rec} (value: {round(value)})")

    print("---------------------")
    id = df["iterations"].argmin()
    value = df["iterations"].min()
    mut = df["mutation"][id]
    rec = df["recombination"][id]

    print(f"Best of iters: mutation: {mut} recombination: {rec} (value: {round(value)})")


def analyze_TSP_run(filename='r0708518.csv'):
    """Loads in the values obtained from a run of a TSP problem and analyzes the result."""
    return


if __name__ == "__main__":
    # params_dict = {"which_recombination": "", "which_mutation": ""}
    # mytest = main.r0708518(params_dict)
    # mytest.optimize('./tour250.csv')

    discuss_performance_rec_operators()
    # analyze_operators()
    pass
