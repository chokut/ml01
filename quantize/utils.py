#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time

import numpy as np
import torch

#def add_noise(x, rate):
#    im = x.detach()
#    ns = torch.rand(*im.size())
#    ns[ns <= (1. - rate)] = 0.
#    ns[ns > (1. - rate)] = 1.
#    im += ns
#    im = torch.clamp(im, 0., 1.)
#    return im

def add_noise(x, rate):
    im = x.detach()
    sel = torch.rand(*im.size())
    sel[sel <= (1. - rate)] = 0.
    sel[sel > (1. - rate)] = 1.
    ns = torch.rand(*im.size()) * 2.
    ns -= 1.
    ns *= sel
    im += ns
    im = torch.clamp(im, 0., 1.)
    return im


def add_weight_noise(w_, rate=0., amp=2.):
    w = w_.detach().numpy()

    rng = np.random.default_rng()
    ns = rng.uniform(-amp, amp, w.shape)

    sel = np.random.rand(*w.shape)
    sel[sel < (1. - rate)] = 0.
    sel[sel >= (1. - rate)] = 1.
    ns = np.exp(ns * sel)
    w = w * ns
    w = np.clip(w, -5., 5.)
    return torch.from_numpy(w.astype(np.float32)).clone()

BITS = 8
RANGE = 4.
#RANGE = 16.

#def set_q_opts(bits, range_):
#    global BITS, RANGE
#    BITS = bits
#    RANGE = range_

def quantize(x, bits=BITS, range_=RANGE):
    scale = (2 ** bits - 1) / (2. * range_)
    q = torch.round(x * scale).to(torch.int64)
    q = torch.clip(q, -2 ** (bits - 1), 2 ** (bits - 1) - 1)
    return q

def dequantize(q, bits=BITS, range_=RANGE):
    scale = (2 ** bits - 1) / (2. * range_)
    q = torch.clip(q, -2 ** (bits - 1), 2 ** (bits - 1) - 1)
    x = q / scale
    return x

def join_w_b(w, b):
    wb = torch.cat([w, b.unsqueeze(1)], dim=1)
    r = torch.cat([torch.zeros(wb.shape[1] - 1), torch.tensor([1.])]).unsqueeze(0)
    wb_ex = torch.cat([wb, r], dim=0)
    return wb_ex

def add_one(x):
    x1 = torch.cat([x, torch.tensor([1])])
    return x1

