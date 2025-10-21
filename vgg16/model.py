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
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16_CIFAR(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG16_CIFAR, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 → 16

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 → 8

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 → 4

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4 → 2

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2 → 1
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

##----------------------------------------------------------
#class VGG16(nn.Module):
#    def __init__(self, num_classes=10):
#        super(VGG16, self).__init__()
#        self.layer1 = nn.Sequential(
#            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(64),
#            nn.ReLU())
#        self.layer2 = nn.Sequential(
#            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(64),
#            nn.ReLU(), 
#            nn.MaxPool2d(kernel_size = 2, stride = 2))
#        self.layer3 = nn.Sequential(
#            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(128),
#            nn.ReLU())
#        self.layer4 = nn.Sequential(
#            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(128),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size = 2, stride = 2))
#        self.layer5 = nn.Sequential(
#            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(256),
#            nn.ReLU())
#        self.layer6 = nn.Sequential(
#            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(256),
#            nn.ReLU())
#        self.layer7 = nn.Sequential(
#            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(256),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size = 2, stride = 2))
#        self.layer8 = nn.Sequential(
#            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(512),
#            nn.ReLU())
#        self.layer9 = nn.Sequential(
#            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(512),
#            nn.ReLU())
#        self.layer10 = nn.Sequential(
#            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(512),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size = 2, stride = 2))
#        self.layer11 = nn.Sequential(
#            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(512),
#            nn.ReLU())
#        self.layer12 = nn.Sequential(
#            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(512),
#            nn.ReLU())
#        self.layer13 = nn.Sequential(
#            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(512),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size = 2, stride = 2))
#        self.fc = nn.Sequential(
#            nn.Dropout(0.5),
#            nn.Linear(7*7*512, 4096),
#            nn.ReLU())
#        self.fc1 = nn.Sequential(
#            nn.Dropout(0.5),
#            nn.Linear(4096, 4096),
#            nn.ReLU())
#        self.fc2= nn.Sequential(
#            nn.Linear(4096, num_classes))
#        
#    def forward(self, x):
#        out = self.layer1(x)
#        out = self.layer2(out)
#        out = self.layer3(out)
#        out = self.layer4(out)
#        out = self.layer5(out)
#        out = self.layer6(out)
#        out = self.layer7(out)
#        out = self.layer8(out)
#        out = self.layer9(out)
#        out = self.layer10(out)
#        out = self.layer11(out)
#        out = self.layer12(out)
#        out = self.layer13(out)
#        out = out.reshape(out.size(0), -1)
#        out = self.fc(out)
#        out = self.fc1(out)
#        out = self.fc2(out)
#        return out

