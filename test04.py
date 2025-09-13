#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import Net
import config as cf

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_epochs = 10         # 学習を繰り返す回数
num_batch = 100         # 一度に処理する画像の枚数

#----------------------------------------------------------
# 学習用／評価用のデータセットの作成

def image_resize(x):
    #print(x.shape)
    x = x[0,2:25,3:25]
    return x
# 変換方法の指定
transform = transforms.Compose([
    transforms.ToTensor(),
    image_resize
    ])

# MNISTデータの取得
# https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
# 学習用
train_dataset = datasets.MNIST(
    './data',               # データの保存先
    train = True,           # 学習用データを取得する
    download = True,        # データが無い時にダウンロードする
    transform = transform   # テンソルへの変換など
    )
# 評価用
test_dataset = datasets.MNIST(
    './data', 
    train = False,
    transform = transform
    )

# データローダー
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = num_batch,
    shuffle = True)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,     
    batch_size = num_batch,
    shuffle = True)

data = next(iter(test_dataloader))
images, labels = data

fig, axes = plt.subplots(10, 10)
fig.set_size_inches(8, 8)

for i, im in enumerate(images):
    #print(im.shape)
    im = im.view(23, 22)
    #im = im.view(28, 28)
    #im = im[2:25,3:25]
    #print(im.shape)
    ax = axes.flatten()[i]
    ax.imshow(im, cmap="gray", vmin=0., vmax=1.)

plt.show()

