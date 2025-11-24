#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=64, output_size=10, N=2000):
        super(Net, self).__init__()

        self.N = N
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        gauss = torch.randn(N, hidden_size)
        self.register_buffer("gauss", gauss)

        th = 0.25
        pepper = torch.rand(N, hidden_size) - 0.5
        pepper[pepper >= th] = 1
        pepper[pepper < -th] = -1
        pepper[(-th <= pepper) & (pepper < th)] = 0
        pepper *= math.sqrt(1./(th*2))
        self.register_buffer("pepper", pepper)

    def forward(self, x, noise_std=0, noise_type="gauss"):
        indices = torch.randperm(self.N)[:x.shape[0]]

        x = self.fc1(x)
        if noise_std > 0:
            if noise_type == "gauss":
                x = x + self.gauss[indices,:] * noise_std
            elif noise_type == "pepper":
                x = x + self.pepper[indices,:] * noise_std
            else:
                raise Exception("invalid noise type")
        x1 = x.detach()
        x = F.relu(x)
        x = self.fc2(x)
        return x, x1

#class Pepper(nn.Module):
#    def __init__(self, input_size=28*28, hidden_size=64, output_size=10, N=2000):
#        super(Net, self).__init__()
#
#        self.N = N
#        self.hidden_size = hidden_size
#
#        self.fc1 = nn.Linear(input_size, hidden_size)
#        self.fc2 = nn.Linear(hidden_size, output_size)
#
#        gauss = torch.randn(N, hidden_size)
#        self.register_buffer("gauss", gauss)
#
#        th = 0.25
#        pepper = torch.rand(N, hidden_size) - 0.5
#        pepper[pepper >= th] = 1
#        pepper[pepper < -th] = -1
#        pepper[(-th <= pepper) & (pepper < th)] = 0
#        pepper *= np.sqrt(1./(th*2))
#
#    def forward(self, x, noise_std=0):
#        indices = torch.randperm(self.N)[:x.shape[0]]
#
#        x = self.fc1(x)
#        if noise_std > 0:
#            x = x + self.pepper[indices,:] * noise_std
#        x = F.relu(x)
#        x = self.fc2(x)
#        return x


# ==== CNNモデル定義 ====
#class CNN(nn.Module):
#    def __init__(self):
#        super(CNN, self).__init__()
#        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.fc1 = nn.Linear(64 * 7 * 7, 128)
#        self.fc2 = nn.Linear(128, 10)
#        self.dropout = nn.Dropout(0.25)
#
#    def forward(self, x):
#        x = self.pool(F.relu(self.conv1(x)))  # (1,28,28) -> (32,14,14)
#        x = self.pool(F.relu(self.conv2(x)))  # (32,14,14) -> (64,7,7)
#        x = x.view(-1, 64 * 7 * 7)
#        x = F.relu(self.fc1(x))
#        x = self.dropout(x)
#        x = self.fc2(x)
#        return x

#class CNN(nn.Module):
#    def __init__(self):
#        super(CNN, self).__init__()
#        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#        #self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#        self.pool = nn.MaxPool2d(2, 2)
#        #self.fc1 = nn.Linear(32 * 7 * 7, 128)
#        self.fc1 = nn.Linear(32 * 14 * 14, 128)
#        self.fc2 = nn.Linear(128, 10)
#        self.dropout = nn.Dropout(0.25)
#
#    def forward(self, x):
#        x = self.pool(F.relu(self.conv1(x)))  # (1,28,28) -> (32,14,14)
#        #x = self.pool(F.relu(self.conv2(x)))  # (32,14,14) -> (64,7,7)
#        #x = x.view(-1, 64 * 7 * 7)
#        x = x.view(-1, 32 * 14 * 14)
#        x = F.relu(self.fc1(x))
#        x = self.dropout(x)
#        x = self.fc2(x)
#        return x

