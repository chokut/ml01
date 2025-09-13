#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

#----------------------------------------------------------
# ハイパーパラメータなどの設定値
learning_rate = 0.01   # 学習率
image_size = 28*28      # 画像の画素数(幅x高さ)
hidden_size = 256

