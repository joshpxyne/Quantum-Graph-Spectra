import numpy as np
import networkx as nx
from scipy.optimize import minimize
from pyquil import Program, get_qc
from pyquil.gates import RX, RZ, CNOT
from pyquil.paulis import PauliSum
from pyquil.api import WavefunctionSimulator
import pyquil.api as api

qvm = api.QVMConnection()

sim = WavefunctionSimulator(random_seed=1337)

num_layers = 20

from pyquil.paulis import sX, sZ, sI

H = 0.2*sX(2)*sZ(1)*sX(0) + 0.9*sX(2)*sI(1)*sX(0) + 0.3*sZ(2)*sZ(1)*sZ(0)

c_1 = 0.5*(sI(0)+sZ(0))
c_2 = 0.5*(sX(0)+(sZ(0)*sX(0)))
c_3 = 0.5*(sX(0)-(sZ(0)*sX(0)))
c_4 = 0.5*(sI(0)-sZ(0))

### pyquil.unitary_tools.lifted_pauli(pauli_sum, qubits)
def adjacencyBuilder(adjacency_matrix) -> PauliSum:
    if adjacency_matrix.size() == 2x2:
        ## Logic for correct paulis

    else:
        adjacency_A = tensor(c_1, adjacencyBuilder(adjacency_matrix.upper_left))
        adjacency_B = tensor(c_2, adjacencyBuilder(adjacency_matrix.upper_right))
        adjacency_C = tensor(c_3, adjacencyBuilder(adjacency_matrix.bottom_left))
        adjacency_D = tensor(c_4, adjacencyBuilder(adjacency_matrix.bottom_right))
        return adjacency_A + adjacency_B + adjacency_C + adjacency_D
print(H)
# Define ansatz
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
    return wave



def solveVQE(hamiltonian: PauliSum) -> float:
    num_qubits = hamiltonian.get_qubits()
    initial_params = np.random.uniform(low = 0, high = 2*np.pi, size = ((num_layers+1)*len(num_qubits),))
    minimum = minimize(expectation, initial_params, method='Nelder-Mead', args=(len(num_qubits), hamiltonian))
    return minimum.fun