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

#a = np.random.beta(0.1, 0.1, 10000)
##print(a)
#print(np.std(a))

#x = np.arange(100)
#plt.hist(a)
#plt.show()
#noise_std = 4
##th = 0.25
th = 0.25
M, N = 5, 5
noise_std = 1.5

l = []
#for i in range(10000):
#    a = torch.rand(M, N) - 0.5
#    a[a >= th] = 1
#    a[a < -th] = -1
#    a[(-th <= a) & (a < th)] = 0
#    #a *= np.sqrt(2)
#    a *= np.sqrt(1./(th*2))
#    a *= noise_std
#    l += [torch.std(a)]

a = torch.rand(M, N) - 0.5
a[a >= th] = 1
a[a < -th] = -1
a[(-th <= a) & (a < th)] = 0
a *= np.sqrt(1./(th*2))
a *= noise_std
print(a)

#print(sum(l) / len(l))
#a *= 2
#a *= (1 / (th * 2))
#a *= noise_std
#print(a)
#print(torch.std(a))
#print(torch.count_nonzero(a))
#
#b = torch.randn(M, N) * noise_std
#print(b)
#print(torch.std(b))
 

