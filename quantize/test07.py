#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time
import random

import numpy as np
#import torch

from utils import quantize, dequantize, add_one, join_w_b

def fix_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

fix_seeds()

np.set_printoptions(precision=3, linewidth=120)

#WORLD_W = 12
#WORLD_H = 9
#
#a = np.ones([4, 4])
##print(a)
##print(np.pad(a, ((1, 2), (3, 4))))
#PAYLOAD_R0, PAYLOAD_C0 = 2, 2
#PAYLOAD_H, PAYLOAD_W = a.shape
#
#print(a)
#PAD0, PAD1 = PAYLOAD_R0, WORLD_H - PAYLOAD_H - PAYLOAD_R0
#PAD2, PAD3 = PAYLOAD_C0, WORLD_W - PAYLOAD_W - PAYLOAD_C0
#print(np.pad(a, ((PAD0, PAD1), (PAD2, PAD3))))
##print(np.pad(a, ((PAYLOAD_R0, WORLD_H - PAYLOAD_H - PAYLOAD_R0), (PAYLOAD_C0, WORLD_W - PAYLOAD_W - PAYLOAD_C0))))

def add_pad(a, world_h, world_w, r0, c0):
    WORLD_W = world_w
    WORLD_H = world_h
    PAYLOAD_R0, PAYLOAD_C0 = r0, c0
    PAYLOAD_H, PAYLOAD_W = a.shape
    PAD0, PAD1 = PAYLOAD_R0, WORLD_H - PAYLOAD_H - PAYLOAD_R0
    PAD2, PAD3 = PAYLOAD_C0, WORLD_W - PAYLOAD_W - PAYLOAD_C0
    return np.pad(a, ((PAD0, PAD1), (PAD2, PAD3)))

w = np.random.randint(0, 10, [5, 5])
print(w)
w = add_pad(w, 11, 9, 1, 2)
print(w)

#w = np.random.randint(0, 10, [9, 9])
#print(w)
print()

for i, x in enumerate(w.flatten()):
    if i % 8 == 0:
        print(f"0x{i:04x},", end="")
    print(f"0x{x:04x}", end="")
    if i % 8 == 7:
        print()
    else:
        print("", end=",")

