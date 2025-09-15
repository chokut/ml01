#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time

SCALE = 256  # 小数点以下4桁
SCALE = 16  # 小数点以下4桁

def to_fixed(x):
    return int(x * SCALE)

def from_fixed(a):
    #return a / SCALE
    return a / SCALE / SCALE

def fixed_mul(a, b):
    #return (a * b) // SCALE
    return (a * b)

A = 3.14
B = 2.21
A = 0.51
B = 0.7
A = 0.8
B = 0.7
# 例
a = to_fixed(A)
b = to_fixed(B)
print(a, b)
c = fixed_mul(a, b)
print(c)
print(from_fixed(c))  # ≈ 8.539

