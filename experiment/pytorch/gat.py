#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> gat.py
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/12/10 17:28
@Desc   ：
=================================================='''

import os
import pickle
import warnings
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
import config.read_gat_config as config

from dgl.nn.pytorch import GATConv
from utils.pytorch import *
from layers.pytorch import *

warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=UserWarning)
os.chdir('../../')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU


# GCN 分类器
class GATClassifier(nn.Module):
    def __init__(self,
                 in_feats,
                 n_output=2,
                 n_hidden=[256, 256, 256],
                 n_head=8,
                 activation=F.leaky_relu,
                 dropout=0.5,
                 pooling_type='h'):
        super(GATClassifier, self).__init__()
        self.n_hidden = n_hidden
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.activation = activation

        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(in_feats, n_hidden[0], num_heads=n_head,
                               feat_drop=dropout, attn_drop=dropout, activation=activation))
        for i in range(len(n_hidden)-1):
            self.gat_layers.append(GATConv(n_hidden[i], n_hidden[i+1], num_heads=n_head,
                                           feat_drop=dropout, attn_drop=dropout, activation=activation))

        if pooling_type == 'h':
            self.dense_1 = nn.Linear(2*np.sum(n_hidden), 128)
            # self.dense_1 = nn.Linear(2 * n_hidden[-1], 128)
        else:
            self.dense_1 = nn.Linear(2*n_hidden[-1], 128)
        self.dense_2 = nn.Linear(128, n_output)

    def forward(self, graph):
        readouts = list()

        # GAT
        feat = graph.ndata['h'].float()
        for idx, gat_layer in enumerate(self.gat_layers):
            feat = gat_layer(graph, feat)
            feat = torch.mean(feat, dim=1).view(-1, self.n_hidden[idx])
            readouts.append(SumMaxPooling()(graph, feat))

        if self.pooling_type == 'h':
            merged = th.cat(readouts, dim=1)
        else:
            merged = readouts[-1]

        # MLP
        merged = merged.view(-1, self.get_flatten_size(merged))
        dropout1 = nn.Dropout(self.dropout)(merged)
        dense1 = self.dense_1(dropout1)
        dense1 = self.activation(dense1)
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
def build_model(in_feats, n_output=2,n_hidden=[256, 256, 256], n_head=9,
                activation=nn.LeakyReLU, dropout=0.5, pooling_type='h'):
    model = GATClassifier(in_feats=in_feats, n_output=n_output,
                          n_hidden=n_hidden, n_head=n_head, activation=activation,
                          dropout=dropout, pooling_type=pooling_type).to(DEVICE)
    optimizer = opt.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    return model, optimizer, criterion


# 数据加载
def load_data(root_path, name, type='', node_attr_type=0, device=None):
    return GraphDataset(root_path, name, type=type, node_attr_type=node_attr_type, device=device)


# 实验
def main():
    # 加载数据
    graph_dataset = load_data(config.dataset_dir, config.dataset_name, node_attr_type=config.node_attr_type, device=DEVICE)
    model_name = "{}bs_{}ep_{}es_{:.5f}lr_{}hi_{:.2f}dp_{}pt_{}fd_{}nh_{}nat".format(config.batch_size, config.epoch,
                                                                                     config.es_patience,
                                                                                     config.lr, config.n_hidden,
                                                                                     config.dropout, config.pooling_type,
                                                                                     config.n_fold, config.n_head,
                                                                                     config.node_attr_type)

    # 构建参数模型
    def build_gcn_model():
        return build_model(graph_dataset.get_feature_size(), n_output=config.n_output,
                           n_hidden=config.n_hidden, n_head=config.n_head, activation=F.leaky_relu,
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
    result.to_csv("{}/{}/result.csv".format(config.save_dir, model_name), index=False)
    result.describe().to_csv("{}/{}/result_describe.csv".format(config.save_dir, model_name), index=False)


if __name__ == "__main__":
    # execute only if run as a script
    print("GAT_Classifier_Experiments")
    main()
