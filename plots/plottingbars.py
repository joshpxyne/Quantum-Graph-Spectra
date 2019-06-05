import matplotlib.pyplot as plt
import numpy as np

x = np.arange(5)
lst = [(26.18771629333496, 0.0005009889443475579), (42.352642345428464, 0.22465062206232664), (28.165834760665895, 0.002412561561502535), (44.75053324699402, 0.2904245382095795), (46.56294770240784, 0.5033196008884878)]


times = [i[0] for i in lst]
accuracies = [i[1] for i in lst]

fig, ax = plt.subplots()
plt.bar(x, times)
plt.xticks(x, ('Undirected Adjacency', 'Directed Adjacency', 'Undirected Laplacian', 'Directed Laplacian - outdegree', 'Directed Laplacian - indegree'))
plt.xticks(rotation=10)
plt.ylabel("Mean Runtime (s)")
plt.show()

fig, ax = plt.subplots()
plt.bar(x, accuracies)
plt.xticks(x, ('Undirected Adjacency', 'Directed Adjacency', 'Undirected Laplacian', 'Directed Laplacian - outdegree', 'Directed Laplacian - indegree'))
plt.xticks(rotation=10)
plt.ylabel("Mean Error (s)")
plt.show()