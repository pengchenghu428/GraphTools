#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> process
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/25 9:39
@Desc   ：数据处理
=================================================='''

import dgl
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def collate(samples, device=torch.device('cuda')):
    '''
    将多张图打包，并送入gpu
    :param samples: 对(graph, label)的列表
    :param device: 训练位置
    :return:
    '''
    if not device is None:
        for idx in range(len(samples)):  # 送进GPU中
            samples[idx][0].to(device)

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(labels),


def get_dataset_info(dirpath="data/", dataset="FRANKENSTEIN"):
    '''
    :param dirpath: 数据集地址
    :param dataset: 数据集名称
    :return:
    '''
    GRAPH_IDX_PATH = "{}/{}/{}.graph_idx".format(dirpath, dataset, dataset)
    GRAPH_LABELS_PATH = "{}/{}/{}.graph_labels".format(dirpath, dataset, dataset)
    all_idx = np.unique(pd.read_csv(GRAPH_IDX_PATH, header=None).values.flatten())  # 所有图的索引号
    all_label = pd.read_csv(GRAPH_LABELS_PATH, header=None).values.flatten()  # 所有图的标签
    return all_idx, all_label


def split_datasets(X, y, ratio=0.2):
    '''
    划分数据集
    :param X:
    :param y: 分层采样
    :return:
    '''
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=42)
    train_split_idx, val_split_idx = next(iter(stratified_split.split(X, y)))
    train_graphs_idx, val_graphs_idx = X[train_split_idx], X[val_split_idx]
    train_target, val_target = y[train_split_idx], y[val_split_idx]
    return train_graphs_idx, val_graphs_idx, train_target, val_target

