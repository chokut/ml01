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

from model import *

#import utils

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="model_vanilla.pt")
parser.add_argument('--input_size', type=int, default=768)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--output_size', type=int, default=10)
parser.add_argument('--learnrate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--noiserate', type=float, default=0.)
parser.add_argument('--droprate', type=float, default=0.)
args = parser.parse_args()

print(f"# model file: {args.model}")
print(f"# input_size: {args.input_size}")
print(f"# hidden_size: {args.hidden_size}")
print(f"# output_size: {args.output_size}")
print(f"# learning rate: {args.learnrate}")
print(f"# noise rate: {args.noiserate}")
print(f"# drop rate: {args.droprate}")

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_dir = "../ds"

#----------------------------------------------------------
# 学習用／評価用のデータセットの作成

#normalize = transforms.Normalize(
#    mean=[0.4914, 0.4822, 0.4465],
#    std=[0.2023, 0.1994, 0.2010],
#)
#
# 変換方法の指定
#transform = transforms.Compose([
#    transforms.Resize((227,227)),
#    transforms.ToTensor(),
#    normalize,
#])

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

# MNISTデータの取得
# https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
# 学習用
train_dataset = datasets.CIFAR100(
    data_dir,               # データの保存先
    train = True,           # 学習用データを取得する
    download = True,        # データが無い時にダウンロードする
    transform = transform   # テンソルへの変換など
    )
# 評価用
test_dataset = datasets.CIFAR100(
    data_dir, 
    train = False,
    transform = transform
    )

# データローダー
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = args.batch_size,
    shuffle = True)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,     
    batch_size = args.batch_size,
    shuffle = True)

#----------------------------------------------------------
# ニューラルネットワークの生成
#model = VGG16(100).to(device)
model = VGG16_CIFAR(100).to(device)

#x = torch.rand(5, 3, 32, 32).to(device)
#y = model(x)
#print(y.shape)
#----------------------------------------------------------
# 損失関数の設定
criterion = nn.CrossEntropyLoss() 

#----------------------------------------------------------
# 最適化手法の設定
optimizer = torch.optim.Adam(model.parameters(), lr = args.learnrate) 

#----------------------------------------------------------
# 学習
model.train()  # モデルを訓練モードにする

for epoch in range(args.epochs): # 学習を繰り返し行う
    loss_sum = 0

    for inputs, labels in train_dataloader:

        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # optimizerを初期化
        optimizer.zero_grad()

        # ニューラルネットワークの処理を行う
        #inputs = inputs.view(-1, args.input_size) # 画像データ部分を一次元へ並び変える
        outputs = model(inputs)

        # 損失(出力とラベルとの誤差)の計算
        loss = criterion(outputs, labels)
        loss_sum += loss

        # 勾配の計算
        loss.backward()

        # 重みの更新
        optimizer.step()

    # 学習状況の表示
    print(f"Epoch: {epoch+1}/{args.epochs}, Loss: {loss_sum.item() / len(train_dataloader)}")

# モデルの重みの保存
checkpoint = {
    "model_state_dict": model.state_dict(),
    "params": {
            "input_size": args.input_size,
            "hidden_size": args.hidden_size,
            "output_size": args.output_size,
    }
}
torch.save(checkpoint, args.model)

#----------------------------------------------------------
# 評価
model.eval()  # モデルを評価モードにする

loss_sum = 0
correct = 1

with torch.no_grad():
    for inputs, labels in test_dataloader:

        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # ニューラルネットワークの処理を行う
        outputs = model(inputs)

        # 損失(出力とラベルとの誤差)の計算
        loss_sum += criterion(outputs, labels)

        # 正解の値を取得
        pred = outputs.argmax(1)
        # 正解数をカウント
        correct += pred.eq(labels.view_as(pred)).sum().item()

print(f"Loss: {loss_sum.item() / len(test_dataloader)}, Accuracy: {100*correct/len(test_dataset)}% ({correct}/{len(test_dataset)})")

