#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> gcn
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/12/9 17:11
@Desc   ：GCN 模型实验
=================================================='''
import os
import pickle
import torch as th
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
import config.read_gcn_config as config

from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from dgl.nn.pytorch import GraphConv, SumPooling, MaxPooling
from dgl.data import Subset
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
                 activation=F.leaky_relu,
                 dropout=0.5,
                 pooling_type='h'):
        super(GCNClassifier, self).__init__()
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.activation = activation

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GraphConv(in_feats, n_hidden[0], activation=activation))
        for i in range(len(n_hidden)-1):
            self.gcn_layers.append(GraphConv(n_hidden[i], n_hidden[i+1], activation=activation))

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
        dense1 = self.dense_1(dropout1)
        dropout2 = nn.Dropout(self.dropout)(dense1)
        dense2 = self.dense_2(dropout2)
        out = th.sigmoid(dense2)
        return out, graph

    def get_flatten_size(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# 模型构建
def build_model(in_feats, n_output=2,n_hidden=[256, 256, 256],
                activation=nn.LeakyReLU, dropout=0.5, pooling_type='h'):
    model = GCNClassifier(in_feats=in_feats, n_output=n_output,
                          n_hidden=n_hidden, activation=activation,
                          dropout=dropout, pooling_type=pooling_type).to(DEVICE)
    optimizer = opt.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(reduction='sum')
    return model, optimizer, criterion


# 数据加载
def load_data(root_path, name):
    return GraphDataset(root_path, name, type='')


# 实验
def main():
    # 加载数据
    graph_dataset = load_data(config.dataset_dir, config.dataset_name)
    model_name = "{}ep_{}es_{:.5f}lr_{}hi_{:.2f}dp_{}pt_{}fd".format(config.epoch, config.es_patience,
                                                                     config.lr, config.n_hidden,
                                                                     config.dropout, config.pooling_type,
                                                                     config.n_fold)

    # 构建参数模型
    def build_gcn_model():
        return build_model(graph_dataset.get_feature_size(), n_output=config.n_output,
                           n_hidden=config.n_hidden, activation=F.leaky_relu,
                           dropout=config.dropout, pooling_type=config.pooling_type)

    # 多随机种子10-折训练
    metrics_results = list()
    for random_seed in config.random_seeds:
        metrics_result = k_fold_train(dataset=graph_dataset, model_fn=build_gcn_model, n_fold=config.n_fold,
                                      save_path=config.save_dir, model_name=model_name, device=DEVICE,
                                      batch_size=config.batch_size, random_seed=random_seed,
                                      epochs=config.epoch, es_patience=config.es_patience)
        metrics_results.append(metrics_result)
    result = pd.concat(metrics_results, axis=0, ignore_index=True)
    result.to_csv("{}/{}/result.csv".format(config.save_dir, model_name))


if __name__ == "__main__":
    # execute only if run as a script
    print("GCN_Classifier_Experiments")
    main()
