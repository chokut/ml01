#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time

import numpy as np
#import torch

np.set_printoptions(precision=3)
a = np.random.rand(3, 3) * 2. - 1.
print("# a")
print(a)
b = np.random.rand(*a.shape)
print("# b")
print(b)
print("# y = a @ b")
y = a @ b
print(y)
print()

def quantize(x, bits=8, range_=1.):
    scale = (2 ** bits - 1) / range_
    q = np.round(x * scale).astype(int)
    return q

def dequantize(q, bits=8, range_=1.):
    scale = (2 ** bits - 1) / range_
    x = q / scale
    return x

for bits in range(3, 10):
    #bits = 8
    qa = quantize(a, bits)
    #print(qa)
    qb = quantize(b, bits)
    #print(qb)
    qy = qa @ qb
    d1 = dequantize(dequantize(qy, bits), bits)
    #print(d1)
    
    mae = float(np.mean(np.abs(y - d1)))
    print(bits, mae)

