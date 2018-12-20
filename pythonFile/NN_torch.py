# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 22:25:18 2018

@author: kumac
"""

import torch
from torch import nn
from torch import optim

import pandas as pd

net = nn.Sequential(
        nn.Linear(225, 512),
        nn.ReLU(),
        nn.Linear(512, 124),
        nn.ReLU(),
        nn.Linear(124, 20)
)

# データの読み込み
train_df = pd.read_csv("db_list.csv", index_col=0)
test_df = pd.read_csv("query_list.csv", index_col=0)

X_train = torch.tensor(train_df.drop(["target"], axis=1).values, dtype=torch.float32)
y_train = torch.tensor(train_df["target"], dtype=torch.int64)

X_test = torch.tensor(test_df.drop(["target"], axis=1).values, dtype=torch.float32)
y_test = torch.tensor(test_df["target"], dtype=torch.int64)
# めんどいので外れ値を21に変更
#y_test[56] = 21
#y_test[57] = 21

# 損失関数
loss_fn = nn.CrossEntropyLoss()

# adam
optimizer = optim.Adam(net.parameters())

# 損失ログ
losses_train = []
losses_test = []
accuracy = []

# 20エポック回す
for epoc in range(100):
    optimizer.zero_grad()
    
    y_pred = net(X_train)
    
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    
    optimizer.step()
    
    losses_train.append(loss.item())
    
    y_test_pred = net(X_test)
    _, predicted = torch.max(y_test_pred, 1)
    corrects = 0
    for i in range(len(predicted)):
        if(predicted[i]==y_test[i]):
            corrects += 1
    accuracy.append(corrects/len(y_test))
    
    
    print("-"*8+"epoch{}".format(epoc)+"-"*8)
    print("pred:{}".format(predicted))
    print("corr:{}".format(y_test))
    print("accuracy:{:.3}".format(accuracy[-1]))
    print("-"*20)