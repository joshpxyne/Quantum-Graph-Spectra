import random
import numpy as np
import networkx as nx
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pyquil.paulis import PauliSum, sX, sZ, sI
from networkx.drawing.nx_agraph import graphviz_layout


# H = 0.2*sX(2)*sZ(1)*sX(0) + 0.9*sX(2)*sI(1)*sX(0) + 0.3*sZ(2)*sZ(1)*sZ(0)

c_1 = lambda n: 0.5*(sI(n)+sZ(n))
c_2 = lambda n: 0.5*(sX(n)+(sZ(n)*sX(n)))
c_3 = lambda n: 0.5*(sX(n)-(sZ(n)*sX(n)))
c_4 = lambda n: 0.5*(sI(n)-sZ(n))

def adjacencyConstruct(size,show,density):
    test_adjacency = np.zeros((size,size)).astype(int)
    for i in range(size):
        for j in range(i,size):
            if i!=j:
                if random.uniform(0, 1) < density:
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

def pauliBuilder(adjacency_matrix,n) -> PauliSum:
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
        Pauli_A = c_1(n)*pauliBuilder(Adjacency_A,n+1)
        Pauli_B = c_2(n)*pauliBuilder(Adjacency_B,n+1)
        Pauli_C = c_3(n)*pauliBuilder(Adjacency_C,n+1)
        Pauli_D = c_4(n)*pauliBuilder(Adjacency_D,n+1)
        return Pauli_A + Pauli_B + Pauli_C + Pauli_D

# H = 1*sX(0)
# solveVQE(H)

# classical_times = [0.0006008148193359375, 0.0008120536804199219, 0.002541065216064453, 0.0011060237884521484]
# timesteps = range(2,6)
# plt.plot(timesteps,classical_times,'go')
# plt.show()


    