#!/usr/bin/env python
# coding: utf-8

# In[94]:


import random
import numpy as np
import networkx as nx
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pyquil.paulis import PauliSum, sX, sZ, sI
from networkx.drawing.nx_agraph import graphviz_layout


# In[95]:


c_1 = lambda n: 0.5*(sI(n)+sZ(n))
c_2 = lambda n: 0.5*(sX(n)+(sZ(n)*sX(n)))
c_3 = lambda n: 0.5*(sX(n)-(sZ(n)*sX(n)))
c_4 = lambda n: 0.5*(sI(n)-sZ(n))


# In[96]:


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


# In[97]:


def pauliBuilder(adjacency_matrix,n) -> PauliSum:
    #padding to n = next power of 2
    dim = adjacency_matrix.shape[0]
    if ((dim & (dim - 1)) != 0):
        new_dim = int (np.power(2, np.ceil(np.log(dim)/np.log(2))))
        adjacency_matrix = np.pad (adjacency_matrix, ((0, new_dim - dim), (0, new_dim - dim)), 'constant', constant_values = (0))
        #print (adjacency_matrix)
        n = int(np.log2 (adjacency_matrix.shape[0]) - 1)
        dim = new_dim
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
        
        Adjacency_A, Adjacency_B, Adjacency_C, Adjacency_D = adjacency_matrix[:int(dim/2), :int(dim/2)],                                                              adjacency_matrix[:int(dim/2), int(dim/2):],                                                              adjacency_matrix[int(dim/2):, :int(dim/2)],                                                              adjacency_matrix[int(dim/2):, int(dim/2):]
        Pauli_A = c_1(n)*pauliBuilder(Adjacency_A,n+1)
        Pauli_B = c_2(n)*pauliBuilder(Adjacency_B,n+1)
        Pauli_C = c_3(n)*pauliBuilder(Adjacency_C,n+1)
        Pauli_D = c_4(n)*pauliBuilder(Adjacency_D,n+1)
        return Pauli_A + Pauli_B + Pauli_C + Pauli_D


# In[161]:


#My code starts here 
import itertools
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy import optimize


# In[99]:


#Generate 1000 0/1 matrices of size (n, n)
def genAdjacencyMatrices (n):
    return [np.reshape(np.array(i), (n, n)) for i in itertools.product([0, 1], repeat = n*n)]


# In[100]:


#Generate random sampled adjacency matrices
def sampleAdjacencyMatrices (num_matrices, size, density):
    sample = []
    for k in range (num_matrices):
        if (k%50 == 0):
            print (k)
        test_adjacency = np.zeros((size,size)).astype(int)
        for i in range(size):
            for j in range(size):
                if random.uniform(0, 1) < density:
                    test_adjacency[i][j] = 1
        sample += [test_adjacency]
    return np.array(sample)


# In[185]:


def sampleAdjacencyMatricesUndirected (num_matrices, size, density):
    sample = []
    for k in range (num_matrices):
        if (k%50 == 0):
            print (k)
        test_adjacency = np.zeros((size,size)).astype(int)
        for i in range(size):
            for j in range(i,size):
                if i!=j:
                    if random.uniform(0, 1) < density:
                        test_adjacency[i][j] = 1
                        test_adjacency[j][i] = 1
        sample += [test_adjacency]
    return np.array(sample)


# In[101]:


#Count gates in a pauli string representation
def count_gates (pauli_string):
    count = 0
    for i in (pauli_string):
        if i == '*':
            count = count + 1
    return count


# In[102]:


#create a pauli string for an adjacency matrix
def generatePauliString (adjacency_matrix):
    n = int(np.log2 (adjacency_matrix.shape[0]) - 1)
    adjacency_matrix_pauli = pauliBuilder (adjacency_matrix, n)
    pauli_string = str (adjacency_matrix_pauli)
    return pauli_string


# In[215]:


#returns maximum number of gates for a given input size and a map of adjacency matrices to number of gates
def maxNumberGates (n):
    #sampling 250 each of different densities
    #adjacency_matrices = np.concatenate ((sampleAdjacencyMatricesUndirected (20, n, .1), sampleAdjacencyMatricesUndirected (20, n, .25), sampleAdjacencyMatricesUndirected (20, n, .33), 
                                          #sampleAdjacencyMatricesUndirected (20, n, .5), sampleAdjacencyMatricesUndirected (20, n, .75)))
    #I used this for n = 16-128:
    #adjacency_matrices = sampleAdjacencyMatricesUndirected (10, n, .75)
    #this for n = 256
    adjacency_matrices = sampleAdjacencyMatricesUndirected (5, n, .75)
    max_gates = 0
    gates_map = []
    print_val = 100
    i = 0
    for adjacency_matrix in adjacency_matrices:
        i = i + 1
        print (i)
        if (i%print_val == 0):
            print (i)
        pauliString = generatePauliString (adjacency_matrix)
        num_gates = count_gates (pauliString)
        gates_map.append (tuple([adjacency_matrix, num_gates]))
        if (num_gates > max_gates):
            max_gates = num_gates
    gates_map = sorted(gates_map,key=itemgetter(1))
    gates_map.reverse()
    return (max_gates, gates_map)


# In[ ]:


(num, gates) = maxNumberGates (4)


# In[ ]:


#find maxNumberGates and gates mapes for n = range (2:16)
X = []
Y = []
gates_maps = []
for n in range (2, 16):
    print (n)
    X.append (n)
    (max_gates, gates_map) = maxNumberGates (n)
    Y.append (max_gates)
    gates_maps.append (gates_map)


# In[ ]:


#adding higher n values
X.append (16)
(max_gates, gates_map) = maxNumberGates (16)
Y.append (max_gates)
gates_maps.append (gates_map)


# In[ ]:



X.append (32)
(max_gates, gates_map) = maxNumberGates (32)
Y.append (max_gates)
gates_maps.append (gates_map)


# In[ ]:



X.append (64)
(max_gates, gates_map) = maxNumberGates (64)
Y.append (max_gates)
gates_maps.append (gates_map)


# In[ ]:


X.append (128)
(max_gates, gates_map) = maxNumberGates (128)
Y.append (max_gates)
gates_maps.append (gates_map)


# In[ ]:


X.append (256)
(max_gates, gates_map) = maxNumberGates (256)
Y.append (max_gates)
gates_maps.append (gates_map)


# In[235]:


plt.xlabel ('# vertices')
plt.ylabel ('# gates')
plt.scatter (X, Y)
plt.plot(X, Y_fitted, label='fitted curve: 6.4X^2 - 189.9X + 1575.0' , color = "red")
plt.legend(loc='best')
plt.show()

plt.xlabel ('# vertices')
plt.ylabel ('# gates')
plt.scatter (X, Y)
plt.plot(range(251), Y_fitted_2, label='fitted curve: 1.05^X + 2832.8' , color = "red")
plt.legend(loc='best')
plt.show()


# In[220]:


def guess_func(x, a, b, c):
    return a*x**2 + b*x + c 


# In[221]:


params, params_covariance = optimize.curve_fit(guess_func, X, Y)


# In[223]:


def fitted_curve(X, a, b, c):
    return [a*x_val**2 + b*x_val + c for x_val in X]
Y_fitted = fitted_curve (X, params[0], params[1], params[2])
plt.plot(X, Y_fitted, label='fitted curve: 6.40065423X^2 -189.80870783X + 1574.97013821' , color = "red")
plt.show()


# In[231]:


def guess_func_2(x, a, b):
    return a**x + b
params_2, params_covariance_2 = optimize.curve_fit(guess_func_2, X, Y)
def fitted_curve_2(X, a, b):
    return [a**x_val + b for x_val in X]
Y_fitted_2 = fitted_curve_2 (range(251), params_2[0], params_2[1])
plt.plot(range(251), Y_fitted_2, label='fitted curve' , color = "red")
plt.show()


# In[ ]:




