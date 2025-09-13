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
    return torch.from_numpy(w.astype(np.float32)).clone()

