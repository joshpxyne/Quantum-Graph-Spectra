import numpy as np
import networkx as nx
from scipy.optimize import minimize
from pyquil import Program, get_qc, unitary_tools
from pyquil.gates import RX, RZ, CNOT
from pyquil.paulis import PauliSum
from pyquil.api import WavefunctionSimulator
import pyquil.api as api
import math
import random
import time
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

qvm = api.QVMConnection()
sim = WavefunctionSimulator()
num_layers = 5

from pyquil.paulis import sX, sZ, sI

# H = 0.2*sX(2)*sZ(1)*sX(0) + 0.9*sX(2)*sI(1)*sX(0) + 0.3*sZ(2)*sZ(1)*sZ(0)

c_1 = lambda n: 0.5*(sI(n)+sZ(n))
c_2 = lambda n: 0.5*(sX(n)+(sZ(n)*sX(n)))
c_3 = lambda n: 0.5*(sX(n)-(sZ(n)*sX(n)))
c_4 = lambda n: 0.5*(sI(n)-sZ(n))

### pyquil.unitary_tools.lifted_pauli(pauli_sum, qubits)
### simplify_pauli_sum(pauli_sum)

# test_adjacency = np.matrix('0 1 1 0; 1 0 1 1; 1 1 0 1; 0 1 1 0')

def adjacency_construct(size,show):
    test_adjacency = np.zeros((size,size)).astype(int)
    for i in range(size):
        for j in range(i,size):
            if i!=j:
                if random.uniform(0, 1) < 0.5:
                    test_adjacency[i][j] = 1
                    test_adjacency[j][i] = 1
    if (show):
        G = nx.from_numpy_matrix(test_adjacency)
        nx.draw(G, cmap = plt.get_cmap('jet'))
        plt.show()
    print(test_adjacency)
    return test_adjacency


# test_adjacency = np.matrix('0 1 1 1 1 0 1 0; \
#                             1 0 1 1 1 1 1 1; \
#                             1 1 0 1 1 1 1 0; \
#                             1 1 1 0 1 1 1 1; \
#                             1 1 1 1 0 1 1 1; \
#                             0 1 1 1 1 0 1 1; \
#                             1 1 1 1 1 1 0 1; \
#                             0 1 0 1 1 1 1 0')
# print(test_adjacency)

def adjacencyBuilder(adjacency_matrix,n) -> PauliSum:
    if adjacency_matrix.shape == (2,2):
        if str(adjacency_matrix)==str(np.matrix('0 0; 0 0')): return sI(n) - sI(n)
        if str(adjacency_matrix)==str(np.matrix('1 0; 0 1')): return sI(n)
        if str(adjacency_matrix)==str(np.matrix('0 1; 1 0')): return sX(n)
        if str(adjacency_matrix)==str(np.matrix('1 1; 1 1')): return sX(n)+sI(n)

        if str(adjacency_matrix)==str(np.matrix('1 0; 0 0')): return c_1(n)
        if str(adjacency_matrix)==str(np.matrix('0 1; 0 0')): return c_2(n)
        if str(adjacency_matrix)==str(np.matrix('0 0; 1 0')): return c_3(n)
        if str(adjacency_matrix)==str(np.matrix('0 0; 0 1')): return c_4(n)

        if str(adjacency_matrix)==str(np.matrix('1 1; 0 0')): return c_1(n) + c_2(n)
        if str(adjacency_matrix)==str(np.matrix('0 1; 0 1')): return c_2(n) + c_4(n)
        if str(adjacency_matrix)==str(np.matrix('1 0; 1 0')): return c_1(n) + c_3(n)
        if str(adjacency_matrix)==str(np.matrix('0 0; 1 1')): return c_3(n) + c_4(n)

        if str(adjacency_matrix)==str(np.matrix('1 1; 1 0')): return c_1(n) + sX(n)
        if str(adjacency_matrix)==str(np.matrix('1 1; 0 1')): return sI(n) + c_2(n)
        if str(adjacency_matrix)==str(np.matrix('1 0; 1 1')): return sI(n) + c_3(n)
        if str(adjacency_matrix)==str(np.matrix('0 1; 1 1')): return sX(n) + c_4(n)
            
        ## Logic for correct paulis

    else:
        dim = adjacency_matrix.shape[0]
        
        Adjacency_A, Adjacency_B, Adjacency_C, Adjacency_D = adjacency_matrix[:int(dim/2), :int(dim/2)], \
                                                             adjacency_matrix[:int(dim/2), int(dim/2):], \
                                                             adjacency_matrix[int(dim/2):, :int(dim/2)], \
                                                             adjacency_matrix[int(dim/2):, int(dim/2):]
        Pauli_A = c_1(n)*adjacencyBuilder(Adjacency_A,n+1)
        Pauli_B = c_2(n)*adjacencyBuilder(Adjacency_B,n+1)
        Pauli_C = c_3(n)*adjacencyBuilder(Adjacency_C,n+1)
        Pauli_D = c_4(n)*adjacencyBuilder(Adjacency_D,n+1)
        return Pauli_A + Pauli_B + Pauli_C + Pauli_D

def ansatz(params, num_layers, num_qubits):
    program = Program()
    for layer in range(num_layers):
        for qubit in range(num_qubits):
            program += RX(params[num_qubits*layer + qubit], qubit)
        for qubit in range(num_qubits):
            program += RZ(params[num_qubits*(layer+1) + qubit], qubit)
        for qubit in range(num_qubits - 1):
            program += CNOT(qubit, qubit+1)
    return program

# Function calculating expectation value
def expectation(params, num_qubits, hamiltonian):
    program = ansatz(params, num_layers, num_qubits)
    wave = sim.expectation(program,hamiltonian)
    # print(wave)
    return wave

def solveVQE(hamiltonian: PauliSum) -> float:
    num_qubits = hamiltonian.get_qubits()
    initial_params = np.random.uniform(low = 0, high = 2*np.pi, size = ((num_layers+1)*len(num_qubits),))
    minimum = minimize(expectation, initial_params, method='Nelder-Mead', args=(len(num_qubits), hamiltonian))
    return minimum.fun

def performanceTests(maximum):
    quantum_times = []
    classical_times = []
    timesteps = range(2,maximum)
    for i in range(2,maximum):
        test_adjacency = adjacency_construct(2**i,False)
        # Paulis = adjacencyBuilder(test_adjacency,0)
        # print(Paulis)
        # print(unitary_tools.lifted_pauli(Paulis, range(int(math.log(test_adjacency.shape[0],2)))))
        classical_time_init = time.time()
        print(np.linalg.eig(test_adjacency))
        classical_time = time.time()-classical_time_init
        # quantum_time_init = time.time()
        # print(solveVQE(Paulis))
        # quantum_time = time.time() - quantum_time_init
        # quantum_times.append(quantum_time)
        classical_times.append(classical_time)
        # print("quantum",quantum_time)
        # print("classical",classical_time)
    # print(quantum_times)
    print(classical_times)
    # plt.plot(timesteps,quantum_times,'bo')
    plt.plot(timesteps,classical_times,'go')
    plt.show()

# H = 1*sX(0)
# solveVQE(H)

# classical_times = [0.0006008148193359375, 0.0008120536804199219, 0.002541065216064453, 0.0011060237884521484]
# timesteps = range(2,6)
# plt.plot(timesteps,classical_times,'go')
# plt.show()
performanceTests(8)
    