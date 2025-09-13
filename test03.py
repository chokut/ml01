#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time
import argparse

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import Net
import config as cf

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="model_vanilla.pth")
args = parser.parse_args()

device = "cpu"

model = Net(cf.image_size, cf.hidden_size, 10).to(device)
model.load_state_dict(torch.load(args.model, weights_only=True))
model.eval()  # モデルを評価モードにする

fig, axes = plt.subplots(3, 10)
fig.set_size_inches(10, 4)

w = model.fc1.weight.data
w = w[:128,:256]

np.set_printoptions(precision=2)
torch.set_printoptions(precision=2)

for i, rate in enumerate(np.linspace(0.1, 1., 10)):
    for j, amp in enumerate([1., 2., 4.]):
        #print(i, j, w, amp)
        #print(f"{torch.min(w)} {torch.max(w)}")
        axes[j, i].imshow(utils.add_weight_noise(w, rate, amp), cmap="gray", vmin=-.2, vmax=.2)

plt.show()



