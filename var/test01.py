#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import math
import re
import time
import random

import numpy as np
import torch

#torch.set_printoptions(precision=4, threshold=10000, linewidth=200)
torch.set_printoptions(precision=4)

def make_mat(var=1.):
    return torch.randn([50, 50]) / math.sqrt(1. / var)

a = make_mat(.5)
print(a.var(unbiased=False))
b = make_mat(.2)
print(b.var(unbiased=False))
y1 = a * b
print(y1.var(unbiased=False))
y2 = a @ b
print(y2.var(unbiased=False))

#t = torch.rand(20, 10, device="cuda")
#print(t)
#var_rows = t.var(dim=1, unbiased=False)
#print(var_rows)

#r = 0.3
#
#t = torch.randn([500, 10]) / math.sqrt(1. / r)
#print(t)
#v = t.var(unbiased=False)
#print(v)

#for i in range(100):
#    a = random.randint(0, 10)
#    print(a, end=" ")
#
