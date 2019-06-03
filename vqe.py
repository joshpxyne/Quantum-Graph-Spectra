import numpy as np
from scipy.optimize import minimize
from pyquil import Program, get_qc
from pyquil.gates import RX, RZ, CNOT
from pyquil.api import WavefunctionSimulator
from pyquil.paulis import PauliSum
from pyquil.api._devices import list_devices, list_lattices

# qvm = QVMConnection()
sim = WavefunctionSimulator()

qc = get_qc("Aspen-4-3Q-A")
# qc = get_qc("9q-generic-qvm")

device_names = list_devices()  # Available devices are subject to change.
lattice_names = list(list_lattices().keys())

lattice_name = lattice_names[0]

print(lattice_name)

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
def expectation(params, num_qubits, hamiltonian, num_layers):
    program = ansatz(params, num_layers, num_qubits)
    wave = sim.expectation(program,hamiltonian)
    # print(wave)
    return wave

def solveVQE(hamiltonian: PauliSum, num_layers) -> float:
    num_qubits = hamiltonian.get_qubits()
    initial_params = np.random.uniform(low = 0, high = 2*np.pi, size = ((num_layers+1)*len(num_qubits),))
    minimum = minimize(expectation, initial_params, method='Nelder-Mead', args=(len(num_qubits), hamiltonian, num_layers))
    return minimum.fun