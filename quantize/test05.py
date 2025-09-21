#
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 
#

import os
import sys
import re
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- PyTorch NN 定義 ---
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


# --- scikit-learn 互換ラッパー ---
class TorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dim=10, lr=0.01, batch_size=16, max_epochs=50):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.model_ = None
        self.scaler_ = None  # データ正規化用

    def fit(self, X, y):
        # スケーリング（標準化）
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        input_dim = X.shape[1]
        output_dim = len(torch.unique(y))
        self.model_ = SimpleNN(input_dim, self.hidden_dim, output_dim)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.max_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                outputs = self.model_(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        X = self.scaler_.transform(X)  # 学習時のスケーラーを使用
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model_(X)
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


# --- データセット準備 ---
iris = load_iris()
X, y = iris.data, iris.target

# --- GridSearchCV ---
param_grid = {
    "hidden_dim": [5, 10, 20],
    "lr": [0.01, 0.001],
    "batch_size": [8, 16],
    "max_epochs": [50, 100],
}

grid = GridSearchCV(TorchClassifier(), param_grid, cv=3, n_jobs=1)
grid.fit(X, y)

print("Best params:", grid.best_params_)
print("Best score:", grid.best_score_)

