#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

N = 101
x = np.linspace(0, 4, N) + np.random.rand(N) - .5
y = 0.0 + 0.75 * x + np.random.rand(N) - .5
#print(x)
#print(y)

xa = np.c_[np.zeros(N), x]
a0, a1 = np.linalg.pinv(xa) @ y
print(a0, a1)
#t = np.linalg.inv(xa.T @ xa) @ xa.T @ y
#print(t)
#a = (1. / (x.T @ x) * x.T) @ y
#print(a)

x2 = np.array([0, 4.])
y2 = a1 * x2
fig, axes = plt.subplots(1, 1)
fig.set_size_inches(8, 8)

axes.scatter(x, y)
axes.set_aspect('equal')
axes.set_xlim([0, 4.])
axes.set_ylim([0, 4.])
axes.plot(x2, y2)

plt.show()



