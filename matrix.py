import numpy as np
import networkx as nx
from scipy.optimize import minimize
from pyquil import Program, get_qc, unitary_tools
from pyquil.gates import RX, RZ, CNOT
from pyquil.paulis import PauliSum
from pyquil.api import WavefunctionSimulator
import pyquil.api as api
import math

qvm = api.QVMConnection()

sim = WavefunctionSimulator(random_seed=1337)

num_layers = 20

from pyquil.paulis import sX, sZ, sI

H = 0.2*sX(2)*sZ(1)*sX(0) + 0.9*sX(2)*sI(1)*sX(0) + 0.3*sZ(2)*sZ(1)*sZ(0)

# c_1 = 0.5*(sI(0)+sZ(0))
# c_2 = 0.5*(sX(0)+(sZ(0)*sX(0)))
# c_3 = 0.5*(sX(0)-(sZ(0)*sX(0)))
# c_4 = 0.5*(sI(0)-sZ(0))

c_1 = lambda n: 0.5*(sI(n)+sZ(n))
c_2 = lambda n: 0.5*(sX(n)+(sZ(n)*sX(n)))
c_3 = lambda n: 0.5*(sX(n)-(sZ(n)*sX(n)))
c_4 = lambda n: 0.5*(sI(n)-sZ(n))

### pyquil.unitary_tools.lifted_pauli(pauli_sum, qubits)
### simplify_pauli_sum(pauli_sum)

# test_adjacency = np.matrix('0 1 1 0; 1 0 1 1; 1 1 0 1; 0 1 1 0')
test_adjacency = np.matrix('0 1 1 1 1 1 1 0; \
                            1 0 1 1 1 1 1 1; \
                            1 1 0 1 1 1 1 1; \
                            1 1 1 0 1 1 1 1; \
                            1 1 1 1 0 1 1 1; \
                            1 1 1 1 1 0 1 1; \
                            1 1 1 1 1 1 0 1; \
                            0 1 1 1 1 1 1 0')




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

print(adjacencyBuilder(test_adjacency,0))
print(unitary_tools.lifted_pauli(adjacencyBuilder(test_adjacency,0), range(int(math.log(test_adjacency.shape[0],2)))))

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
    print(wave)
    return wave



def solveVQE(hamiltonian: PauliSum) -> float:
    num_qubits = hamiltonian.get_qubits()
    initial_params = np.random.uniform(low = 0, high = 2*np.pi, size = ((num_layers+1)*len(num_qubits),))
    minimum = minimize(expectation, initial_params, method='Nelder-Mead', args=(len(num_qubits), hamiltonian))
    return minimum.fun
test_adjacency_vqe = np.matrix('0 1; 1 0')
print(solveVQE(adjacencyBuilder(test_adjacency,0)))