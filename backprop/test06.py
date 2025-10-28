#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time
import random

import numpy as np
import torch

import matplotlib.pyplot as plt

import pickle
import gzip

data_dir = "../ds/MNIST/raw"

key_file ={
    'x_train': os.path.join(data_dir, 'train-images-idx3-ubyte.gz'),
    't_train': os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'),
    'x_test':  os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'),
    't_test':  os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'),
}

def load_label(file_name):
    file_path = file_name
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
            # 最初の８バイト分はデータ本体ではないので飛ばす
    one_hot_labels = np.zeros((labels.shape[0], 10))
    for i in range(labels.shape[0]):
        one_hot_labels[i, labels[i]] = 1
    return one_hot_labels

def load_image(file_name):
    file_path = file_name
    with gzip.open(file_path, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
        # 画像本体の方は16バイト分飛ばす必要がある
    return images

def convert_into_numpy(key_file):
    dataset = {}

    dataset['x_train'] = load_image(key_file['x_train'])
    dataset['t_train'] = load_label(key_file['t_train'])
    dataset['x_test']  = load_image(key_file['x_test'])
    dataset['t_test']  = load_label(key_file['t_test'])

    return dataset

def load_mnist():
    # mnistを読み込みNumPy配列として出力する
    dataset = convert_into_numpy(key_file)
    dataset['x_train'] = dataset['x_train'].astype(np.float32) # データ型を`float32`型に指定しておく
    dataset['x_test'] = dataset['x_test'].astype(np.float32)
    dataset['x_train'] /= 255.0
    dataset['x_test'] /= 255.0 # 簡単な標準化
    dataset['x_train'] = dataset['x_train'].reshape(-1, 28*28)
    dataset['x_test']  = dataset['x_test'].reshape(-1, 28*28)
    return dataset

ds = load_mnist()
#print(ds)
train_inputs = ds["x_train"]
train_labels = ds["t_train"]
test_inputs = ds["x_test"]
test_labels = ds["t_test"]
#print(train_inputs)
#print(train_inputs.shape)

train_inputs = torch.from_numpy(train_inputs)
train_labels = torch.from_numpy(train_labels)
test_inputs = torch.from_numpy(test_inputs)
test_labels = torch.from_numpy(test_labels)

train_inputs = train_inputs.to(torch.float32)
train_labels = train_labels.to(torch.float32)
test_inputs = test_inputs.to(torch.float32)
test_labels = test_labels.to(torch.float32)
#print(train_inputs)
#print(train_labels)
#print(test_inputs)
#print(test_labels)

device = "cuda"

train_inputs = train_inputs.to(device)
train_labels = train_labels.to(device)
test_inputs = test_inputs.to(device)
test_labels = test_labels.to(device)

def activate(x):
    y = x.clone()
    y[y < 0.] = 0.
    #y[y > 10.] = 0.
    return y

def activate_dash(x):
    y = x.clone()
    y[y < 0.] = 0.
    y[y >= 0.] = 1.
    return y

lr = 0.0001
wy = torch.randn(256, 784).to(device)
wz = torch.randn(10, 256).to(device)

batch = 3

dLdwy = torch.zeros(wy.shape).to(device)
dLdwz = torch.zeros(wz.shape).to(device)
    
#a = torch.randn(2, 3, 2)
#b = torch.randn(2, 2)
#y = a @ b
#print(y)
for epoch in range(10):
    loss = 0
    for i in range(200):
        choice = random.sample(range(train_inputs.shape[0]), batch)
        #print(choice)

        dLdwy.zero_()
        dLdwz.zero_()
    
        x = train_inputs[choice,:]
        t = train_labels[choice]

        x = x.reshape(batch, 1, -1)
        t = t.reshape(batch, 1, -1)
        #print(x)
        #print(x.shape)
        #print(t)
        #sys.exit(0)

        iy = x @ wy.T
        y = activate(iy)
        iz = y @ wz.T
        #z = activate(iz)
        z = iz
        #print(z)
        #sys.exit(0)
        
        #loss += ((z - t) ** 2).sum().item()
        #print(loss)
        #sys.exit(0)
        dLdz = (z - t)
        ###dLdwz += (dLdz * activate_dash(iz)).T @ y
        ###dLdy = (dLdz * activate_dash(iz)) @ wz
        #print(dLdz.shape, y.shape)
        #sys.exit(0)

        print(dLdz.transpose(1, 2) @ y)
        print((dLdz.transpose(1, 2) @ y).shape)
        dLdwz = dLdz.transpose(1, 2) @ y
        dLdy = dLdz @ wz
        dLdwy = (dLdy * activate_dash(iy)).transpose(1, 2) @ x

        ##dLdwz = dLdz.T @ y
        ##dLdy = dLdz @ wz
        ##dLdwy = (dLdy * activate_dash(iy)).T @ x
        print(dLdwz)
        print(dLdwz.shape)
        #print(dLdwz.sum(0))
        #print(dLdwz.sum(0).shape)
        #sys.exit(0)

        #dLdwy = dLdwy.sum(0)
        #dLdwz = dLdwz.sum(0)
        #print(dLdwy)
        sys.exit(0)

        #for j in choice:
        #    x = train_inputs[j,:]
        #    x = x.reshape(-1, x.shape[0])
    
        #    t = train_labels[j]
        #    t = t.reshape(-1, t.shape[0])
    
        #    iy = x @ wy.T
        #    y = activate(iy)
        #    iz = y @ wz.T
        #    #z = activate(iz)
        #    z = iz
    
        #    loss += ((z - t) @ (z - t).T).item()
        #    
        #    dLdz = (z - t)
        #    ##dLdwz += (dLdz * activate_dash(iz)).T @ y
        #    ##dLdy = (dLdz * activate_dash(iz)) @ wz
        #    dLdwz = dLdz.T @ y
        #    dLdy = dLdz @ wz
        #    dLdwy += (dLdy * activate_dash(iy)).T @ x
                
        wy -= (dLdwy / batch) * lr
        wz -= (dLdwz / batch) * lr

    print(f"loss: {loss:.3f}")

    #break


#correct = 0
#total = 0
#for j in range(1000):
#    x = train_inputs[j,:]
#    x = x.reshape(-1, x.shape[0])
#
#    iy = x @ wy.T
#    y = activate(iy)
#    iz = y @ wz.T
#    #z = activate(iz)
#    z = iz
#
#    t = train_labels[j]
#    t = t.reshape(-1, t.shape[0])
#    if np.argmax(z) == np.argmax(t):
#        correct += 1
#    total += 1
#
#print(f"{correct}/{total}")

