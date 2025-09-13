#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time

import numpy as np
import torch

import utils

np.set_printoptions(precision=2)
torch.set_printoptions(precision=2, sci_mode=False)

a = torch.rand(4, 4) * 2
a -= 1.
print(a)
y = utils.add_weight_noise(a, 0.25, 2)
print(y / a)
y = utils.add_weight_noise(a, 0.5, 2)
print(y / a)
y = utils.add_weight_noise(a, 1., 2)
print(y / a)

