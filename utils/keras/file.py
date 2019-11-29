#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> utils -> file
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/11 20:06
@Desc   ：作用读取/保存数据文件
=================================================='''

import os
import numpy as np
import pandas as pd


def checkd_directory(path):
    '''
    保证保存文件时，文件夹存在
    :param path: 文件或文件夹位置
    :return:
    '''
    if not os.path.exists(path):  # 判断该文件/文件夹是否存在
        dirs = path.split('/')
        if '.' in dirs[-1]:  # 如果最后是文件，则弹出最后一个元素
            dirs.pop()
        dir_path = '/'.join(dirs)  # 文件夹路径
        if not os.path.exists(dir_path):  # 文件夹不存在，则创建
            os.makedirs(dir_path)


def load_data(path="data/", dataset="FRANKENSTEIN"):
    '''
    读取包含多个图的数据集
    :param path: 数据集位置
    :param dataset: 数据集名称
    :return: [[X, A], [X, A],...[X, A]], y_graph
    '''
    print('Loading {} dataset...'.format(dataset))

    # 文件路径
    graph_idx_path = "{}/{}/{}.graph_idx".format(path, dataset, dataset)  # graph_idx
    edges_path = "{}/{}/{}.edges".format(path, dataset, dataset)  # edges
    node_attrs_path = "{}/{}/{}.node_attrs".format(path, dataset, dataset)  # node_attrs
    graph_labels_path = "{}/{}/{}.graph_labels".format(path, dataset, dataset)  # graph_labels

    # 读取文件  pd.read_csv 的 速度比 np.genfrontxt 快很多
    graph_idx = pd.read_csv(graph_idx_path, header=None).values.flatten().astype('int')
    edges = pd.read_csv(edges_path, header=None).values.astype('int')
    graph_labels = pd.read_csv(graph_labels_path, header=None).values.flatten().astype('int')
    if os.path.exists(node_attrs_path):  # 是否存在节点属性文件
        node_attrs = pd.read_csv(node_attrs_path, header=None).values
    else:  # 节点无属性，则用单位矩阵替换
        nodes_attrs = None

    # 切分多图
    Xs, As = list(), list()
    node_idx = np.arange(1, graph_idx.shape[0]+1)
    for gidx in np.unique(graph_idx):
        node = node_idx[graph_idx == gidx]  # 切分出节点编号
        node_length = node.shape[0]  # 节点数量
        mask = np.isin(edges, node).all(axis=1)  # 切分出对应的连接
        edge = edges[mask]
        edge_sub_min = edge - np.min(edge)  # 从0开始编号
        A = np.zeros((node_length, node_length))  # 邻接矩阵
        A[edge_sub_min[:, 0], edge_sub_min[:, 1]] = 1
        X = np.eye(node_length, dtype=int) if node_attrs is None else node_attrs[node-1] # 减去1是因为默认节点从1开始编号，切片使用
        Xs.append(X)
        As.append(A.astype('int'))
    y = np.where(graph_labels == -1, 1, 0)
    return Xs, As, y


if __name__ == "__main__":
    # execute only if run as a script
    os.chdir('../')
    load_data()
