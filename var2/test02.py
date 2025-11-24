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

import matplotlib.pyplot as plt

import arch

#import utils

parser = argparse.ArgumentParser()
parser.add_argument('--train', action="store_true", default=False)
parser.add_argument('--model', type=str, default="models/model_vanilla.pt")
parser.add_argument('--arch', type=str, default="Net")
parser.add_argument('--ds', type=str, default="MNIST")
parser.add_argument('--input_size', type=int, default=28*28)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--output_size', type=int, default=10)
parser.add_argument('--learnrate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--train_noise_std', type=float, default=0.)
parser.add_argument('--train_noise_type', type=str, default="gauss")
parser.add_argument('--infer_noise_std', type=float, default=0.)
parser.add_argument('--infer_noise_type', type=str, default="gauss")
#parser.add_argument('--droprate', type=float, default=0.)
args = parser.parse_args()

#print(f"# model file: {args.model}")
if not args.train:
#    print(f"# learning rate: {args.learnrate}")
    print(f"# train noise type: {args.train_noise_type}")
    print(f"# train noise std: {args.train_noise_std}")
#    print(f"# drop rate: {args.droprate}")

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_dir = "../ds"
cp_list = set([5, 10, 20, 40, 80, 100, 200, 300, 400, 500, 1000])

#----------------------------------------------------------
# 学習用／評価用のデータセットの作成

dataset = getattr(datasets, args.ds)

transform = transforms.Compose([
    transforms.ToTensor(),
])

# 学習用
train_dataset = dataset(
    data_dir,               # データの保存先
    train = True,           # 学習用データを取得する
    download = True,        # データが無い時にダウンロードする
    transform = transform   # テンソルへの変換など
    )

# 評価用
test_dataset = dataset(
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
model = getattr(arch, args.arch)(args.input_size, args.hidden_size, args.output_size)
model = model.to(device)

#----------------------------------------------------------
# 損失関数の設定
criterion = nn.CrossEntropyLoss() 

def fs2str(s):
    return f"{s:.1f}".replace(".", "p")

def do_eval():
    model.eval()
    
    loss_sum = 0
    correct = 1
    
    with torch.no_grad():
        for inputs, labels in test_dataloader:
    
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            inputs = inputs.view(-1, args.input_size) # 画像データ部分を一次元へ並び変える
            outputs, x1 = model(inputs, args.infer_noise_std, args.infer_noise_type)
    
            loss_sum += criterion(outputs, labels)
    
            pred = outputs.argmax(1)
            correct += pred.eq(labels.view_as(pred)).sum().item()
    
    #print(f"Loss: {loss_sum.item() / len(test_dataloader)}, Accuracy: {100*correct/len(test_dataset)}% ({correct}/{len(test_dataset)})")
    acc = 100 * correct / len(test_dataset)
    loss = loss_sum.item() / len(test_dataset)
    return loss, acc, correct, len(test_dataset)

os.makedirs(os.path.dirname(args.model), exist_ok=True)
model_prefix, model_ext = os.path.splitext(args.model)

if args.train:
    #do_train()
#def do_train():
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learnrate) 
    
    model.train()
    
    train_loss_trace = []
    test_loss_trace = []

    for epoch in range(args.epochs):
    
        loss_sum = 0
        for inputs, labels in train_dataloader:
    
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
    
            inputs = inputs.view(-1, args.input_size)
            outputs, x1 = model(inputs, args.train_noise_std, args.train_noise_type)
    
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
    
            loss.backward()
            optimizer.step()
    
        print(f"epoch: {epoch+1}/{args.epochs}, train loss: {loss_sum / len(train_dataloader):.4f}, ", end="")
        train_loss_trace += [loss_sum]
        loss_, acc_, correct_, total = do_eval()
        test_loss_trace += [loss_]
        print(f"test loss: {loss_:.4f}, acc: {acc_:.2f}% ({correct_}/{total})", end="")
        print()
    
        if ((epoch + 1) in cp_list) or (epoch + 1 == args.epochs):
            model_path = f"{model_prefix}_{args.train_noise_type}{fs2str(args.train_noise_std)}_cp{epoch+1}{model_ext}"
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "params": {
                },
                "info": {
                    "trace": {
                        "train_loss": train_loss_trace,
                        "test_loss": test_loss_trace,
                    },
                }
            }
            torch.save(checkpoint, model_path)
            print(f"{model_path} saved")
else:
    checkpoint = torch.load(args.model, weights_only=True)
    params = checkpoint["params"]
    model = getattr(arch, args.arch)(**params)
    model.load_state_dict(checkpoint["model_state_dict"])
    #model = model.to(device)

    x = torch.ones(100, 28*28)
    y, x1 = model(x)
    print(x1)
    y, x2 = model(x, 20, "pepper")
    print(x2)

    t = x2 - x1
    print(t)
    ax = plt.gca()
    ax.hist(t.flatten())
    plt.show()

#loss_, acc_, correct_, total = do_eval()
#print(f"test loss: {loss_:.4f}, acc: {acc_:.2f}% ({correct_}/{total})")

