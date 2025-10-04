#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import Net

from utils import quantize, dequantize, add_one, join_w_b

def fix_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

fix_seeds()
torch.set_printoptions(precision=3, linewidth=120, sci_mode=False)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="model_vanilla.pt")
args = parser.parse_args()

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_epochs = 10         # 学習を繰り返す回数
num_batch = 10         # 一度に処理する画像の枚数

# 変換方法の指定
transform = transforms.Compose([
    transforms.ToTensor()
    ])

# MNISTデータの取得
# https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
# 学習用
train_dataset = datasets.MNIST(
    '../ds',               # データの保存先
    train = True,           # 学習用データを取得する
    download = True,        # データが無い時にダウンロードする
    transform = transform   # テンソルへの変換など
    )
# 評価用
test_dataset = datasets.MNIST(
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
model = model.to(device)

input_size = params["input_size"]

#----------------------------------------------------------
# 評価
model.eval()  # モデルを評価モードにする

model = model.to("cpu")
state_dict = model.state_dict()
w1 = state_dict["fc1.weight"]
b1 = state_dict["fc1.bias"]
w2 = state_dict["fc2.weight"]
b2 = state_dict["fc2.bias"]

wb1 = join_w_b(w1, b1)
wb2 = join_w_b(w2, b2)

qwb1 = quantize(wb1)
qwb2 = quantize(wb2)
print(qwb1)
sys.exit(0)
loss_sum = 0
correct = 1

with torch.no_grad():
    for inputs, labels in test_dataloader:

        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to("cpu")
        labels = labels.to("cpu")

        # ニューラルネットワークの処理を行う
        inputs = inputs.view(-1, input_size) # 画像データ部分を一次元へ並び変える

        #xs = torch.cat([inputs, torch.ones([num_batch, 1])], dim=1)
        #u1 = xs @ wb1.T
        #z1 = F.relu(u1)
        #ys = z1 @ wb2.T
        #ys = ys[:,:-1]
        #outputs = ys

        #xs = torch.cat([inputs, torch.ones([num_batch, 1])], dim=1)
        #qxs = quantize(xs)
        #qu1 = qxs @ qwb1.T
        #qz1 = F.relu(qu1)
        #qys = qz1 @ qwb2.T
        #outputs = dequantize(dequantize(dequantize(qys)))
        #outputs = outputs[:,:-1]
        ##print(outputs)
        
        xs = torch.cat([inputs, torch.ones([num_batch, 1])], dim=1)
        qxs = quantize(xs)
        qxs = qxs >> 4
        qu1 = qxs @ qwb1.T
        qu1 = qu1 >> 4
        qz1 = F.relu(qu1)
        qys = qz1 @ qwb2.T
        qys = qys >> 4
        outputs = dequantize(dequantize(dequantize(qys)))
        outputs = outputs[:,:-1]
        #print(outputs)
        
        #outputs = model(inputs)
        #print(ys)
        #print(outputs)
        #break

        # 正解の値を取得
        pred = outputs.argmax(1)
        # 正解数をカウント
        correct += pred.eq(labels.view_as(pred)).sum().item()

print(f"Accuracy: {100*correct/len(test_dataset)}% ({correct}/{len(test_dataset)})")

