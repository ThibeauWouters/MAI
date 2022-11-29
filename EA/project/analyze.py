# Auxiliary file to analyze stuff

import Reporter
import numpy as np
import pandas as pd
import csv
import random
from random import sample
import matplotlib.pyplot as plt
import time

plt.rcParams['figure.figsize'] = (15, 5)

# if __name__=="__main__":
df = pd.read_csv("Data/Analysis_mutation_recombination.csv")
print(df)

# Get operator names
all_mutations = pd.unique(df["mutation"])
all_recombinations = pd.unique(df["recombination"])

# Get the values
best_values = df["best"].values.reshape((len(all_mutations), len(all_recombinations)))
mean_values = df["mean"].values.reshape((len(all_mutations), len(all_recombinations)))
iteration_values = df["iterations"].values.reshape((len(all_mutations), len(all_recombinations)))


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
plt.title("Analysis of number of iterations")
plt.xticks([i for i in range(len(all_recombinations))], labels=all_recombinations)
plt.xlabel("Recombination operator")
plt.yticks([i for i in range(len(all_mutations))], labels=all_mutations)
plt.ylabel("Mutation operator")
# plt.show()
plt.savefig("Plots/Analysis_iteration_value.png", bbox_inches='tight')
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
