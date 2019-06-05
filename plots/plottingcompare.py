

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import numpy as np

x = [4, 5, 8, 9, 16, 32, 64]
lst =  [(2.585895228385925, 4.9364027687737175e-08), (13.683554983139038, 0.035110609181115214), (16.250244092941283, 0.17863540370194406), (63.24338710308075, 0.2964172356868715), (81.63710451126099, 0.4288713187054065), (457.574907541275, 1.5497983761583067), (2580.9958889484406, 3.939294261935955)]

# otherlist = [0.0023797988891601563, 1.6597139596939088, 1.7730239033699036, 2.4004372358322144, 2.2994803905487062, 1.8459036707878114, 1.3680771589279175]

times = [i[0] for i in lst]
accuracies = [i[1] for i in lst]

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

offset = 0
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis["right"] = new_fixed_axis(loc="right",
                                    axes=par2,
                                    offset=(offset, 0))
par2.axis["right"].toggle(all=True)

host.set_xlabel("Number of Vertices")
host.set_ylabel("Mean Runtime (s)")
par1.set_ylabel("Mean Error")

p1, = host.plot(x, times, label="Mean Runtime (s)")
p2, = par1.plot(x, accuracies, label="Mean Error")

par1.set_ylim(-0.4, 3.99)
par2.set_ylim(0.4, 3.939)

host.legend()

par1.axis["left"].label.set_color(p1.get_color())
par2.axis["right"].label.set_color(p2.get_color())


plt.draw()
plt.show()