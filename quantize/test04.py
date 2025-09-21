#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time

import numpy as np
import torch

torch.set_printoptions(precision=3, linewidth=120)

def quantize(x, bits=8, range_=1.):
    scale = (2 ** bits - 1) / range_
    q = torch.round(x * scale).to(torch.int64)
    return q

def dequantize(q, bits=8, range_=1.):
    scale = (2 ** bits - 1) / range_
    x = q / scale
    return x

def join_w_b(w, b):
    wb = torch.cat([w, b.unsqueeze(1)], dim=1)
    r = torch.cat([torch.zeros(wb.shape[1] - 1), torch.tensor([1.])]).unsqueeze(0)
    wb_ex = torch.cat([wb, wb2], dim=0)
    return wb_ex

def add_one(x):
    x1 = torch.cat([x, torch.tensor([1])])
    return x1

bits_ = 16
range_ = 1.

N = 3
w = torch.rand(N, N)
b = torch.rand(N)
x = torch.rand(N)
y = w @ x + b
print(y[:10])

#wb = join_w_b(w, b)
#x1 = add_one(x)
#y = wb @ x1
#print(y[:10])
#print(y.shape)
#
#qwb = quantize(wb, bits_, range_)
#qx1 = quantize(x1, bits_, range_)
#qy = qwb @ qx1
#yy = dequantize(dequantize(qy, bits_, range_), bits_, range_)
#print(yy[:10])
#print(w)
#print(b)
#
#print(w.shape)
#print(b.shape)
#
wb = torch.cat([w, b.unsqueeze(1)], dim=1)
#print(wb)
wb2 = torch.cat([torch.zeros(wb.shape[1] - 1), torch.tensor([1.0])]).unsqueeze(0)
#print(wb2)
print(torch.cat([wb, wb2], dim=0))
print(join_w_b(w, b))
#print(wb.shape)
#
#x = torch.rand(3)
#x1 = torch.cat([x, torch.tensor([1])])
#print(x1)
#y = wb @ x1
#print(y)
#print(y.shape)

