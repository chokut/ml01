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

#def activate(x):
#    return 1 / (1 + np.exp(-x))
#
#def activate_dash(x):
#    return (1. - activate(x)) * activate(x)

def activate(x):
    y = x.copy()
    y[y < 0.] = 0.
    #y[y > 10.] = 0.
    return y

def activate_dash(x):
    y = x.copy()
    y[y < 0.] = 0.
    y[y >= 0.] = 1.
    return y

lr = 0.001
wy = np.random.randn(256, 784)
wz = np.random.randn(10, 256)

batch = 20
for i in range(1000):
    choice = random.sample(range(train_inputs.shape[0]), batch)
    #print(choice)
    loss = 0
    for j in choice:
        x = train_inputs[j,:]
        x = x.reshape(-1, x.shape[0])

        t = train_labels[j]
        t = t.reshape(-1, t.shape[0])

        iy = x @ wy.T
        y = activate(iy)
        iz = y @ wz.T
        z = activate(iz)
        #z = iz
        
        loss += (z - t) @ (z - t).T
        dLdz = (z - t)
        dLdwz = (dLdz * activate_dash(iz)).T @ y
        dLdy = (dLdz * activate_dash(iz)) @ wz
        #dLdwz = dLdz.T @ y
        #dLdy = dLdz @ wz
        dLdwy = (dLdy * activate_dash(iy)).T @ x
            
        #print(np.max(dLdwy), np.max(dLdwy), np.max(dLdwz), np.max(dLdwz))
        wy -= dLdwy * lr
        wz -= dLdwz * lr
        #print(np.max(wy), np.max(wz))

    print(loss)
    #break


correct = 0
total = 0
for j in range(1000):
    x = train_inputs[j,:]
    x = x.reshape(-1, x.shape[0])

    iy = x @ wy.T
    y = activate(iy)
    iz = y @ wz.T
    z = activate(iz)
    #z = iz

    t = train_labels[j]
    t = t.reshape(-1, t.shape[0])
    if np.argmax(z) == np.argmax(t):
        correct += 1
    total += 1

print(f"{correct}/{total}")
