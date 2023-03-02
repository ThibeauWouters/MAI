# Auxiliary file to analyze the given matrices

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['figure.figsize'] = (15, 5)

# Load in the CSV
tour_length = 50
filename = './tour' + str(tour_length) + '.csv'
file = open(filename)
distanceMatrix = np.loadtxt(file, delimiter=",")
file.close()

# Replace infs by -1
distanceMatrix = np.where(distanceMatrix == float('inf'), -1, distanceMatrix)
# Get max value, use it to construct a penalty term, and replace again
max_distance = np.max(distanceMatrix.flatten())
penalty_value = 2 * max_distance
distanceMatrix = np.where(distanceMatrix == -1, penalty_value, distanceMatrix)

# Check where we have the inf values (or penalty terms)
inf_indices = np.where(distanceMatrix == penalty_value, 1, 0)
number_of_illegal_roads = np.sum(inf_indices.flatten())
fraction_of_illegal_roads = number_of_illegal_roads/len(inf_indices.flatten())

print("There are %d illegal roads which is %0.2f percent of roads." % (number_of_illegal_roads, fraction_of_illegal_roads))


# Plot distance matrix
plt.imshow(distanceMatrix)
plt.colorbar()
plt.title(r"Analysis of distance matrix for n = " + str(tour_length))
skip = 10
plt.xticks([i*10 for i in range(tour_length//skip)])
plt.xlabel("x")
plt.yticks([i*10 for i in range(tour_length//skip)])
plt.ylabel("y")
# plt.show()
# Save the figure both as png and pdf
save_name = "Plots/Distance Matrices/Analysis_TSP_size_" + str(tour_length)
plt.savefig(save_name + ".png", bbox_inches='tight')
plt.savefig(save_name + ".pdf", bbox_inches='tight')
plt.close()

# TODO - somehow overlap the illegal roads with a different color?

# fig, ax = plt.subplots()
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='10%', pad=0.1)
# # Plot distance matrix
# im = ax.imshow(distanceMatrix)
# fig.colorbar(im, cax=cax, orientation='vertical')
# # Plot illegal roads
# ax.imshow(inf_indices, cmap='Reds')
# plt.title(r"Analysis of distance matrix for n = " + str(tour_length))
# skip = 10
# ax.set_xticks([i*10 for i in range(tour_length//skip)])
# plt.xlabel("x")
# ax.set_yticks([i*10 for i in range(tour_length//skip)])
# plt.ylabel("y")
# # plt.show()
# # Save the figure both as png and pdf
# save_name = "Plots/Distance Matrices/Analysis_TSP_size_" + str(tour_length)
# plt.savefig(save_name + ".png", bbox_inches='tight')
# plt.savefig(save_name + ".pdf", bbox_inches='tight')
# plt.close()
