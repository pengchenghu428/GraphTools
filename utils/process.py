#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> utils -> process
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/11 20:06
@Desc   ：处理数据
=================================================='''

import numpy as np
from collections import Counter


def normalize_adj(adj, symmetric=True):
    '''
    归一化邻接矩阵
    :param adj: 方阵
    :param symmetric:是否对称处理
    :return:归一化邻接矩阵
    '''
    if symmetric:
        d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    '''
    邻接矩阵自循环 A = A + I
    :param adj:邻接矩阵
    :param symmetric:是否对称处理
    :return: 归一化自循环的邻接矩阵
    '''
    adj = adj + np.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def preprocess_features(features):
    '''
    归一化特征
    :param features:
    :return:
    '''
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def extend_graph(X, A, N):
    '''
    扩展图到指定的尺寸
    :param X: 特征矩阵
    :param A: 邻接矩阵
    :param N: 保留节点数目
    :return: X_resize, A_resize
    '''
    print("Resize Graph Size")
    assert X.shape[0] <= N, "原始Graph的节点数目大于重定义节点数目"

    origin_nodes, feature_dims = X.shape
    X_new = np.r_[X, np.zeros((N-origin_nodes, feature_dims))]
    A_new = np.zeros((N, N))
    A_new[:origin_nodes, :origin_nodes] += A
    return X_new, A_new


def split_data(Xs, As, y, ratio=0.2, stratified=False):
    '''
    按一定比例划分数据
    :param Xs: 特征矩阵集合
    :param As: 邻接矩阵集合
    :param y: 图标签
    :param ratio: 划分比例
    :param stratified: 是否分层采样
    :return:
    '''

    sample_number = round(len(y) * ratio)  # 采样数目
    if not stratified:
        shuffle_idx = np.random.permutation(np.arange(len(y)))
        train_idx, test_idx = shuffle_idx[sample_number:], shuffle_idx[:sample_number]
        return Xs[train_idx], As[train_idx], y[train_idx], Xs[test_idx], As[test_idx], y[test_idx]
    else:
        class_counter = Counter(y)  # 统计各个类别的数量

        train_idx, test_idx = list(), list()
        for idx, key in enumerate(class_counter.keys()):
            ids = np.where(y == key)  # class 下标
            split_n = round(ids * ratio)  # 采样数目
            shuffle_idx = np.random.permutation(ids)  # 打乱列表
            class_train_idx, class_test_idx = shuffle_idx[split_n:], shuffle_idx[:split_n]  # 采样下标
            train_idx.extend(class_train_idx)
            test_idx.extend(class_test_idx)
        train_idx = np.random.permutation(train_idx)  # 重新打乱数据
        test_idx = np.random.permutation(test_idx)

        return Xs[train_idx], As[train_idx], y[train_idx], Xs[test_idx], As[test_idx], y[test_idx]
