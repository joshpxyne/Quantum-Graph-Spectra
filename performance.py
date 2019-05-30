import matrix
import vqe

import math
import statistics as stats
import time
import numpy as np
import matplotlib.pyplot as plt
from pyquil import unitary_tools

layers = 5

def quantumVsClassical(maximum):
    quantum_times = []
    classical_times = []
    timesteps = range(2,maximum)
    for i in range(2,maximum):
        test_adjacency = matrix.adjacencyConstruct(2**i,False,0.5)
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

def densityComparisons(num_trials, density, size):
    results = []
    for _ in range(num_trials):
        test_adjacency = matrix.adjacencyConstruct(2**size,False,density)
        Paulis = matrix.pauliBuilder(test_adjacency,0)
        time_init = time.time()
        vqe.solveVQE(Paulis,layers)
        results.append(time.time() - time_init)
    return stats.mean(results)

def noiseComparisons(num_trials, density, size, noise_model):
    pass

def ansatzComparisons(num_trials, density, size, ansatz):
    pass

print(densityComparisons(20,0.5,2))
