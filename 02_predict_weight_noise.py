#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import Net
import config as cf

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="model_weights.pth")
args = parser.parse_args()

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_epochs = 10         # 学習を繰り返す回数
num_batch = 100         # 一度に処理する画像の枚数

# 変換方法の指定
transform = transforms.Compose([
    transforms.ToTensor()
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

#----------------------------------------------------------
# ニューラルネットワークの生成
model = Net(cf.image_size, cf.hidden_size, 10).to(device)

model.load_state_dict(torch.load(args.model, weights_only=True))

#----------------------------------------------------------
# 評価
model.eval()  # モデルを評価モードにする

fc1_bk = model.fc1.weight.data.detach()

#rates = np.linspace(0, 1., 11)
#for i, rate in enumerate(rates):
for j, amp in enumerate([1., 2., 4.]):
    print(f"#amp={amp}")
    print("rate,acc")
    for i, rate in enumerate(np.linspace(0.1, 1., 10)):
        loss_sum = 0
        correct = 1
        
        with torch.no_grad():
            for inputs, labels in test_dataloader:
        
                # GPUが使えるならGPUにデータを送る
                inputs = inputs.to(device)
                labels = labels.to(device)
        
                model.fc1.weight.data = utils.add_weight_noise(fc1_bk.to("cpu"), rate, amp=amp).to(device)

                # ニューラルネットワークの処理を行う
                inputs = inputs.view(-1, cf.image_size) # 画像データ部分を一次元へ並び変える
                outputs = model(inputs)
        
                # 正解の値を取得
                pred = outputs.argmax(1)
                # 正解数をカウント
                correct += pred.eq(labels.view_as(pred)).sum().item()
        
        #print(f"Accuracy: {100*correct/len(test_dataset)}% ({correct}/{len(test_dataset)})")
        print(f"{rate:.2f},{100*correct/len(test_dataset)}%")

