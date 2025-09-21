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

#N = 5
N = 16

x = np.random.rand(N)
print("# x")
print(x)

w1 = np.random.rand(N, N) * 2. - 1.
#print("# w1")
#print(w1)

z1 = w1.T @ x
z1[z1 < 0.] = 0
print("# z1")
print(z1)

w2 = np.random.rand(N, N) * 2. - 1.
#print("# w2")
#print(w2)

y = w2.T @ z1
print("# y")
print(y)

def quantize(x, bits=8, range_=1.):
    scale = (2 ** bits - 1) / range_
    q = np.round(x * scale).astype('int64')
    return q

def dequantize(q, bits=8, range_=1.):
    scale = (2 ** bits - 1) / range_
    x = q / scale
    return x

bits_ = 8
range_ = 5.

qx = quantize(x, bits_, range_)
print("# qx")
print(qx)
qw1 = quantize(w1, bits_, range_)
qz1 = qw1.T @ qx
qz1[qz1 < 0.] = 0
#qz1 = dequantize(qz1, bits_, range_)

qw2 = quantize(w2, bits_, range_)
qy = qw2.T @ qz1
print("# qy")
print(qy)

yy = dequantize(dequantize(dequantize(qy, bits_, range_), bits_, range_), bits_, range_)
#yy = dequantize(dequantize(qy, bits_, range_), bits_, range_)
print("# yy")
print(yy)

mae = float(np.mean(np.abs(y - yy)))
print("# mae")
print(mae)

#b = np.random.rand(*a.shape)
#print("# b")
#print(b)
#print("# y = a @ b")
#y = a @ b
#print(y)
#print()
#
#for bits in range(3, 10):
#    #bits = 8
#    qa = quantize(a, bits)
#    #print(qa)
#    qb = quantize(b, bits)
#    #print(qb)
#    qy = qa @ qb
#    d1 = dequantize(dequantize(qy, bits), bits)
#    #print(d1)
#    
#    mae = float(np.mean(np.abs(y - d1)))
#    print(bits, mae)

