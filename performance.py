'''
For running performance tests.

qcs reserve --lattice <Aspen-?-??-?>
'''

import matrix
import vqe

import math
import statistics as stats
import time
import numpy as np
import matplotlib.pyplot as plt
from pyquil import unitary_tools, paulis
import sys

np.set_printoptions(threshold=sys.maxsize)

def quantumVsClassical(maximum, layers):
    quantum_times = []
    classical_times = []
    timesteps = range(2,maximum)
    for i in range(2,maximum):
        test_adjacency = matrix.adjacencyConstruct(i,False,0.5)
        Paulis = matrix.pauliBuilder(test_adjacency,0)
        print(Paulis)
        print(unitary_tools.lifted_pauli(Paulis, range(int(math.log(test_adjacency.shape[0],2)))))
        classical_time_init = time.time()
        print(np.linalg.eig(test_adjacency))
        classical_time = time.time()-classical_time_init
        quantum_time_init = time.time()
        print(vqe.solveVQE(Paulis,layers))
        quantum_time = time.time() - quantum_time_init
        quantum_times.append(quantum_time)
        classical_times.append(classical_time)
        print("quantum",quantum_time)
        print("classical",classical_time)
    # print(quantum_times)
    print(classical_times)
    # plt.plot(timesteps,quantum_times,'bo')
    plt.plot(timesteps,classical_times,'go')
    plt.show()

def densityComparisons(num_trials, density, size, layers):
    results = []
    for _ in range(num_trials):
        test_adjacency = matrix.adjacencyConstruct(size,False,density)
        Paulis = matrix.pauliBuilder(test_adjacency,0)
        print(Paulis)
        time_init = time.time()
        vqe.solveVQE(Paulis,layers)
        results.append(time.time() - time_init)
    return stats.mean(results)

def noiseComparisons(num_trials, density, size, noise_model):
    pass

def layeredAnsatzTimeAccuracyComparisons(num_trials, density, size, layers):
    time_results = []
    diff_results = []
    for _ in range(num_trials):
        test_adjacency = matrix.adjacencyConstruct(size,False,density)
        Paulis = paulis.simplify_pauli_sum(matrix.pauliBuilder(test_adjacency,0))
        print(Paulis)
        time_init = time.time()
        result = vqe.solveVQE(Paulis,layers)
        time_results.append(time.time() - time_init)
        correct = min(np.linalg.eig(test_adjacency)[0])
        diff = abs(result - correct)
        diff_results.append(diff)
    return stats.mean(time_results), stats.mean(diff_results)

def differentAnsatzComparisons(num_trials, density, size, layers):
    pass

# densities.append(densityComparisons(20,0.1,2,5))
# densities.append(densityComparisons(20,0.3,2,5))
# densities.append(densityComparisons(20,0.5,2,5))
# densities.append(densityComparisons(20,0.7,2,5))
# densities.append(densityComparisons(20,0.9,2,5))
# densities.append(densityComparisons(20,1,2,5))

ansatzes = []
# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(10,0.5,4,3))
# print(ansatzes)
# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(5,0.5,5,3))
# print(ansatzes)
# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(5,0.5,8,3))
# print(ansatzes)
# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(2,0.5,9,3))
# print(ansatzes)
# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(2,0.5,16,3))
# print(ansatzes)

# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(1,0.5,4,3))
# print(ansatzes)
# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(1,0.5,8,3))
# print(ansatzes)
# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(1,0.5,16,3))
# print(ansatzes)
# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(1,0.5,32,3))
# print(ansatzes)
# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(1,0.5,32,3))
# print(ansatzes)
steps = 35
for i in range(steps+1):
    density = float(i)/float(steps)

    ansatzes.append(layeredAnsatzTimeAccuracyComparisons(3,density,8,3))
    print(density)
    print(ansatzes)


# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(20,0.5,2,2))
# print(ansatzes)
# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(20,0.5,2,3))
# print(ansatzes)
# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(20,0.5,2,4))
# print(ansatzes)
# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(20,0.5,2,5))
# print(ansatzes)
# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(20,0.5,2,10))
# print(ansatzes)
# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(20,0.5,2,15))
# print(ansatzes)
# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(20,0.5,2,20))
# print(ansatzes)
# ansatzes.append(layeredAnsatzTimeAccuracyComparisons(20,0.5,2,30))
# print(ansatzes)
