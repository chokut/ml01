#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time
import random

import numpy as np
import torch

import matplotlib.pyplot as plt



#def activate(x):
#    return 1 / (1 + np.exp(-x))
#
#def activate_dash(x):
#    return (1. - activate(x)) * activate(x)

def activate(x):
    y = x.copy()
    y[y < 0.] = 0.
    return y

def activate_dash(x):
    y = x.copy()
    y[y < 0.] = 0.
    y[y >= 0.] = 1.
    return y

lr = 0.05
wy = np.random.randn(32, 4)
wz = np.random.randn(2, 32)


for i in range(1000):
    for j in range(4):
        x = np.eye(4)[j]
        x = x.reshape(-1, x.shape[0])
        if j == 0:
            t = np.array([0., 0.])
        elif j == 1:
            t = np.array([0., 1.])
        elif j == 2:
            t = np.array([1., 0.])
        else:
            t = np.array([1., 1.])
        t = t.reshape(-1, t.shape[0])

        iy = x @ wy.T
        y = activate(iy)
        iz = y @ wz.T
        z = activate(iz)
        
        dLdz = (z - t)
        dLdwz = (dLdz * activate_dash(iz)).T @ y
        dLdy = (dLdz * activate_dash(iz)) @ wz
        dLdwy = (dLdy * activate_dash(iy)).T @ x
            
        wy -= dLdwy * lr
        wz -= dLdwz * lr

for j in range(4):
    x = np.eye(4)[j]
    x = x.reshape(-1, x.shape[0])
    iy = x @ wy.T
    y = activate(iy)
    iz = y @ wz.T
    z = activate(iz)
    print(j, z)

