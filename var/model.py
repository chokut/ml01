#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

#----------------------------------------------------------
# ニューラルネットワークモデルの定義
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, N=500):
        super(Net, self).__init__()

        self.N = N

        # 各クラスのインスタンス（入出力サイズなどの設定）
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        noise1 = torch.randn(N, hidden_size)
        self.register_buffer("noise1", noise1)

        noise2 = torch.randn(N, output_size)
        self.register_buffer("noise2", noise2)

    def forward(self, x, noise_var1=0., noise_var2=0.):
        # 順伝播の設定（インスタンスしたクラスの特殊メソッド(__call__)を実行）
        index = random.randint(0, self.N - 1)

        x = self.fc1(x)
        if noise_var1 > 0:
            x = x + self.noise1[index] / math.sqrt(1. / noise_var1)

        x = F.relu(x)

        x = self.fc2(x)
        if noise_var2 > 0:
            x = x + self.noise2[index] / math.sqrt(1. / noise_var2)
        return x

