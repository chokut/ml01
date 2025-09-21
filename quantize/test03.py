#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import Net

import utils

torch.set_printoptions(precision=3, linewidth=120)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="model_vanilla.pt")
parser.add_argument('--ds', type=str, default="MNIST")
args = parser.parse_args()

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_epochs = 10         # 学習を繰り返す回数
num_batch = 100         # 一度に処理する画像の枚数

DS = getattr(datasets, args.ds)

# 変換方法の指定
transform = transforms.Compose([
    transforms.ToTensor()
    ])

# MNISTデータの取得
# https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
# 学習用
#train_dataset = datasets.MNIST(
train_dataset = DS(
    '../ds',               # データの保存先
    train = True,           # 学習用データを取得する
    download = True,        # データが無い時にダウンロードする
    transform = transform   # テンソルへの変換など
    )

# 評価用
#test_dataset = datasets.MNIST(
test_dataset = DS(
    '../ds', 
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

#----------------------------------------------------------
# ニューラルネットワークの生成
checkpoint = torch.load(args.model, weights_only=True)
params = checkpoint["params"]
model = Net(**params)
model.load_state_dict(checkpoint["model_state_dict"])
#model = model.to(device)

input_size = params["input_size"]

#----------------------------------------------------------
# 評価
model.eval()  # モデルを評価モードにする



w1 = model.fc1.weight.data.clone()
b1 = model.fc1.bias.data.clone()
w2 = model.fc2.weight.data.clone()
b2 = model.fc2.bias.data.clone()

#print(b1)
#print(b2)

def quantize(x, bits=8, range_=1.):
    scale = (2 ** bits - 1) / range_
    q = torch.round(x * scale).to(torch.int64)
    return q

def dequantize(q, bits=8, range_=1.):
    scale = (2 ** bits - 1) / range_
    x = q / scale
    return x

def join_w_b(w, b):
    #wb = torch.cat([w, b.unsqueeze(1)], dim=1)
    #return wb
    wb = torch.cat([w, b.unsqueeze(1)], dim=1)
    r = torch.cat([torch.zeros(wb.shape[1] - 1), torch.tensor([1.])]).unsqueeze(0)
    wb_ex = torch.cat([wb, r], dim=0)
    return wb_ex

def add_one(x):
    x1 = torch.cat([x, torch.tensor([1])])
    return x1

bits_ = 8
range_ = 16.

qw1 = quantize(w1, bits_, range_)
qw2 = quantize(w2, bits_, range_)

wb1 = join_w_b(w1, b1)
wb2 = join_w_b(w2, b2)
qwb1 = quantize(wb1, bits_, range_)
qwb2 = quantize(wb2, bits_, range_)

count = 0
maes = []
for input_, label in test_dataset:
    #print(input_.shape, label) 

    x = input_.view(28*28)
    #y_ = model(x.view(1, -1))
    #print(y_.detach())
    #a_ = torch.argmax(y_).item()
    ##print(label, a_)

    z1 = w1 @ x + b1
    #print(z1[:20])
    #print(torch.min(z1), torch.max(z1))
    z1[z1 < 0] = 0
    y = w2 @ z1 + b2
    print(y)
    #print(label, torch.argmax(y).item())
    #a = torch.argmax(y).item()

    qx = quantize(add_one(x), bits_, range_)
    qz1 = qwb1 @ qx
    #print(qz1.shape)
    #print(qwb1.shape)
    qz1[qz1 < 0] = 0
    #qz1 = qz1 >> 2
    #print(qz1[:20])
    #qz1[qz1 > 127] = 127
    qz1[qz1 > 255] = 255
    #qz1[qz1 > 511] = 511
    #qz1[qz1 > 1023] = 1023
    #qz1 = (qz1 >> 2) & 255
    #qz1 = qz1 & 255
    qy = qwb2 @ qz1
    #print(qy)
    yy = dequantize(dequantize(dequantize(qy, bits_, range_), bits_, range_), bits_, range_)
    print(yy[:-1])
    #print(label, torch.argmax(y).item(), torch.argmax(yy).item())

    #qx = quantize(x, bits_, range_)
    #qz1 = qw1 @ qx
    #qz1[qz1 < 0.] = 0
    #qz1[qz1 > 127] = 127
    ##qz1[qz1 > 255] = 255
    ##print(qz1[:20])
    #qy = qw2 @ qz1
    ##print(qy)
    #yy = dequantize(dequantize(dequantize(qy, bits_, range_), bits_, range_), bits_, range_)
    ##print(yy)
    #aa = torch.argmax(yy).item()

    mae = float(torch.mean(torch.abs(y - yy[:-1])))
    maes += [mae]
    #print(y)
    #print(yy)
    #print(label, a, aa, mae)

    count += 1
    if count >= 10:
        break

if maes:
    print("mae ave", sum(maes) / len(maes))
