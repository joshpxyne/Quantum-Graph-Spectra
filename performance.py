'''
@author: Josh Payne

Description: For running performance tests

Instructions: add print statements at the bottom to test the different parameters on QVM or QCS.

To reserve a lattice: $ qcs reserve --lattice <Aspen-?-??-?>

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
        test_adjacency = matrix.undirectedAdjacencyConstruct(i,False,0.5)
        Paulis = matrix.pauliBuilder(test_adjacency)
        print(Paulis)
        print(unitary_tools.lifted_pauli(Paulis, range(int(math.log(test_adjacency.shape[0],2)))))
        classical_time_init = time.time()
        classical_time = time.time()-classical_time_init
        quantum_time_init = time.time()
        print("estimated eigenvalue: ",vqe.solveVQE(Paulis,layers))
        print("absolute difference: ",abs(min(np.linalg.eig(test_adjacency)[0] - vqe.solveVQE(Paulis,layers))))
        quantum_time = time.time() - quantum_time_init
        quantum_times.append(quantum_time)
        classical_times.append(classical_time)
        print("quantum runtime: ",quantum_time)
        print("classical runtime: ",classical_time)
    # print(quantum_times)
    print(classical_times)
    # plt.plot(timesteps,quantum_times,'bo')
    plt.plot(timesteps,classical_times,'go')
    plt.show()

def densityComparisons(num_trials, density, size, layers):
    results = []
    for _ in range(num_trials):
        test_adjacency = matrix.undirectedAdjacencyConstruct(size,False,density)
        Paulis = matrix.pauliBuilder(test_adjacency)
        print(Paulis)
        time_init = time.time()
        vqe.solveVQE(Paulis,layers)
        results.append(time.time() - time_init)
    return stats.mean(results)

def noiseComparisons(num_trials, density, size, noise_model):
    pass


def layeredAnsatzTimeAccuracyComparisons(num_trials, density, size, layers, mat_type, eigenvalue):
    time_results = []
    diff_results = []
    for _ in range(num_trials):
        if mat_type==1: # undirected adjacency
            test_matrix = matrix.undirectedAdjacencyConstruct(size,False,density)
            Paulis = paulis.simplify_pauli_sum(matrix.pauliBuilder(test_matrix))
        if mat_type==2: # directed adjacency
            test_matrix = matrix.directedAdjacencyConstruct(size,False,density)
            print(test_matrix)
            Paulis = paulis.simplify_pauli_sum(matrix.pauliBuilder(test_matrix))
        if mat_type==3: # undirected laplacian
            test_matrix = matrix.undirectedLaplacianConstruct(size,False,density)
            Paulis = paulis.simplify_pauli_sum(matrix.laplacianPauliBuilder(test_matrix))
        if mat_type==4: # directed laplacian, outdegree
            test_matrix = matrix.directedOutDegreeLaplacianConstruct(size,False,density)
            Paulis = paulis.simplify_pauli_sum(matrix.laplacianPauliBuilder(test_matrix))
        if mat_type==5: # directed laplacian, indegree
            test_matrix = matrix.directedInDegreeLaplacianConstruct(size,False,density)
            Paulis = paulis.simplify_pauli_sum(matrix.laplacianPauliBuilder(test_matrix))
        print(Paulis)
        time_init = time.time()
        if eigenvalue=="max":
            result = -1*vqe.solveVQE(-1*Paulis,layers)
        else:
            result = vqe.solveVQE(Paulis,layers)
        time_results.append(time.time() - time_init)
        if eigenvalue=="max":
            correct = max(np.linalg.eig(test_matrix)[0])
        else:
            correct = min(np.linalg.eig(test_matrix)[0])
        print("correctEV:",correct)
        diff = abs(result - correct)
        diff_results.append(diff)

    return stats.mean(time_results), stats.mean(diff_results)

def differentAnsatzComparisons(num_trials, density, size, layers):
    pass


### Run tests here ###

ansatzes = []
steps = 30
for i in range(steps+1):
    density = float(i)/float(steps)
    ansatzes.append(layeredAnsatzTimeAccuracyComparisons(1,density,8,3,mat_type=3,eigenvalue="max"))

print(ansatzes)