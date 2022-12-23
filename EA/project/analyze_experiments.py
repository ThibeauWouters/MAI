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


def discuss_results():
    """Shows most important features of experiments."""

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


if __name__ == "__main__":
    # params_dict = {"which_recombination": "", "which_mutation": ""}
    # mytest = main.r0708518(params_dict)
    # mytest.optimize('./tour250.csv')

    # discuss_results()
    analyze_operators()
    pass
