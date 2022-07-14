from pickletools import optimize
from turtle import forward
from transformer import Encoder, adaptive_conv1d

import torch
import torch.nn as nn
from torch.nn import functional as F

import os, json, re, random
import pandas as pd
import numpy as np


class adaptive_transformer(nn.Module):
    
    def __init__(self):
        super(adaptive_transformer, self).__init__()

        self.conv1 = adaptive_conv1d(out_channel=16, kernel_size=3, padding="same")
        # self.position = nn.Parameter(torch.zeros(1, 32, 32)) # [1, n_channels, n_dims], graph embedded features
        self.position = nn.Parameter(torch.zeros(1, 16, 47)) # raw features
        self.dropout  = nn.Dropout(0.1)
        self.encoder  = Encoder(47, 6, 6) # n_dims, n_heads, n_layers
        self.conv2 = adaptive_conv1d(out_channel=1, kernel_size=1, padding="same")
        # self.cla = nn.Linear(in_features=32, out_features=6, bias=True) # graph embedded features
        self.cla = nn.Sequential(nn.Linear(in_features=47, out_features=16, bias=True),
                                nn.Sigmoid(),
                                nn.Linear(in_features=16, out_features=6, bias=True)
                            ) # raw features
    
    def forward(self, x):
        x = self.conv1(x)
        x = x + self.position
        x = self.dropout(x)
        x, _ = self.encoder(x)
        x = self.conv2(x)
        x = x.flatten(1)
        pred = self.cla(x)
        return pred


class adaptive_conv(nn.Module):

    def __init__(self):
        super(adaptive_conv, self).__init__()

        self.layer1 = nn.Sequential(
            adaptive_conv1d(out_channel=16, kernel_size=3, padding="valid"),
            nn.BatchNorm1d(num_features=16),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding="valid"),
            nn.BatchNorm1d(num_features=32),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding="valid"),
            nn.ReLU()
        )

        self.cla = nn.Linear(in_features=512, out_features=6)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.flatten(1)
        x = self.cla(x)

        return x


# read champion roles and transfer into labels
# embs = np.load("all_champ_abilities.npy")
# embs = np.load("embedding/ability_emb_hope_32d.npy") # 48 in 159, adam lr=0.009
# embs = np.load("embedding/ability_emb_lle_32d.npy") # baddest
# embs = np.load("embedding/ability_emb_lle_32d.npy") # 88 in 159, adam lr=0.003

if __name__ == "__main__":

    torch.random.manual_seed(5)

    with open("new_roles.json") as fin:
        roles = json.load(fin)
    all_roles = set()
    root      = "champ/en/"
    end_line  = 0

    X = []
    Y = []

    for f in os.listdir(root):
        champ_name = f[:-4]
        champ_file = pd.read_csv(root + f)
        # nrows = len(champ_file.index)

        # x = embs[end_line: end_line + nrows]
        x = champ_file.values[:, 2:].astype(float) # 
        y = roles[champ_name][0]
        # end_line += nrows

        X.append(x)
        Y.append(y)
        all_roles.add(y)

    # print(embs.shape, end_line)

    all_roles = ["Mage", "Tank", "Marksman", "Support", "Fighter", "Assassin"]
    Y = [all_roles.index(y) for y in Y]

    data = list(zip(X, Y))
    print(len(data))
    random.shuffle(data)
    wanted = [int(35 * 100 / 159), int(20 * 100 / 159), int(27 * 100 / 159), int(16 * 100 / 159), int(44 * 100 / 159), int(17 * 100 / 159)]
    count = [0, 0, 0, 0, 0, 0]

    train = []
    evalu = []
    for it in data:
        role = it[1]
        if count[role] <= wanted[role]:
            train.append(it)
            count[role] += 1
        else:
            evalu.append(it)


    train_N = 100
    evalu_N = 20
    batch_size = 20
    EPOCH_N = 30

    # model = adaptive_transformer()
    model = adaptive_conv()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    loss_fn = nn.CrossEntropyLoss()


    for epoch in range(EPOCH_N):

        model.train()
        optimizer.zero_grad()
        record = []
        for it in train:
            x, y = it

            x = torch.Tensor([x])
            y = torch.Tensor([y]).long()

            pred = model(x)

            loss = loss_fn(pred, y)
            record.append(loss.item())
            loss.backward()

            # if it % batch_size == batch_size - 1:
        optimizer.step()
        # optimizer.zero_grad()
        
        model.eval()
        total   = 0
        correct = 0
        for it in evalu:
            x, y = it

            x = torch.Tensor([x])
            y = torch.Tensor([y]).long()

            pred = model(x)

            _, pred = torch.max(pred.data, 1)
            total += 1
            correct += (y == pred).sum().item()
        record = sum(record) / len(record)
        print("In epoch %d, loss: %f, eval acc: %d in %d" % (epoch, record, correct, total))

    # testing
    model.eval()
    total   = 0
    correct = 0
    record  = []
    L = len(data)
    for it in range(L):
        x, y = data[it]

        x = torch.Tensor([x])
        y = torch.Tensor([y]).long()

        pred = model(x)

        _, pred = torch.max(pred.data, 1)
        total += 1
        correct += (y == pred).sum().item()
        record.append((y[0].item(), pred[0].item()))
    print("Test acc: %d in %d" % (correct, total))

    rubbish = {key: [0, 0, 0] for key in all_roles}
    for it in range(L):
        y, pred = record[it]
        rubbish[all_roles[y]][1] += 1
        rubbish[all_roles[pred]][2] += 1
        if y == pred:
            rubbish[all_roles[y]][0] += 1
    print(rubbish)


