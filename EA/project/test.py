import Reporter
import numpy as np
import pandas as pd
import csv
import random
import matplotlib.pyplot as plt
import time
import main

# Fix random seeds:
np.random.seed(0)
random.seed(0)


def test_connections_matrix(tsp):
    connections_matrix = tsp.connections_matrix
    distance_matrix = tsp.distance_matrix

    very_first = connections_matrix[0][0]
    print(very_first)
    smallest = distance_matrix[0][very_first]
    print(smallest)

    # for i in range(np.size(connections_matrix[0])):
    #     print("Row: %d" % i)
    #     print(distance_matrix[i])
    #     print(connections_matrix[i])
    #     print("Sorted? : ")
    #     print(distance_matrix[i][connections_matrix[i]])


if __name__ == "__main__":
    params_dict = {"mu": 50, "number_of_iterations": 1, "random_perm_init_number": 0, "random_road_init_number": 99,
                   "greedy_road_init_number": 0, "nnb_road_init_number": 1, "use_lso": False}
    mytest = main.r0708518(params_dict)
    mytest.optimize('./tour50.csv')

    # Test cases

    # test_connections_matrix(mytest)

    pass
