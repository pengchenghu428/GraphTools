#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> gcn
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/12/9 17:11
@Desc   ：GCN 模型实验
=================================================='''

import config.read_gcn_config as config
import torch as th
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dgl.nn.pytorch import GraphConv, SumPooling, MaxPooling
from utils.pytorch import *
from layers.pytorch import *

os.chdir('../../')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU


# GCN 分类器
class GCNClassifier(nn.Module):
    def __init__(self,
                 in_feats,
                 n_output=2,
                 n_hidden=[256, 256, 256],
                 activation=nn.ReLU,
                 dropout=0.5,
                 pooling_type='h'):
        super(GCNClassifier, self).__init__()
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.activation = activation

        self.gcn_layers = nn.Module
        self.gcn_layers.append(GraphConv(in_feats, n_hidden[0]))
        for i in range(len(n_hidden)-1):
            self.gcn_layers.append(GraphConv(n_hidden[i], n_hidden[i+1]))

        self.dense_1 = nn.Linear(2*n_hidden[-1], 128)
        self.dense_2 = nn.Linear(128, n_output)

    def forward(self, graph):
        readouts = list()

        # GCN
        feat = graph.ndata['h'].float()
        for gcn_layer in self.gcn_layers:
            feat = gcn_layer(graph, feat)
            readouts.append(SumMaxPooling()(graph, feat))

        if self.pooling_type == 'h':
            merged = th.stack(readouts)
            merged = th.sum(merged, dim=0)
        else:
            merged = readouts[-1]

        # MLP
        merged = merged.view(-1, self.get_flatten_size(merged))
        dropout1 = nn.Dropout(self.dropout)(merged)
        dense1 = self.activation(self.dense_1(dropout1))
        dropout2 = nn.Dropout(self.dropout)(dense1)
        dense2 = self.dense_2(dropout2)
        out = th.sigmoid(dense2)
        return out

    def get_flatten_size(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


