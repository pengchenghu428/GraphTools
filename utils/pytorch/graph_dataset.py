#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> graph_dataset
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/24 14:19
@Desc   ：Torch 方式加载图数据
=================================================='''

import pandas as pd
import numpy as np
import torch
import dgl
import os


class GraphDataset(object):
    """
    加载图数据
    """
    def __init__(self,
                 srcpath,
                 name="FRANKENSTEIN",
                 available_graph_idx=None,
                 type="Train"):
        '''
        实现初始化方法，在初始化的时候将数据载入
        :param srcpath: 数据集文件夹位置
        :param name: 数据集名称
        '''
        print('Loading {} {} dataset...'.format(name, type))

        # 从磁盘读取数据
        graph_idx, edges, graph_labels, node_attrs = self.read_from_disk(srcpath, name)

        # 划分多张图数据
        data, target = self.split_multi_graph(graph_idx, edges, graph_labels,
                                                        node_attrs, available_graph_idx)
        self.data, self.target = data, target

    def __len__(self):
        '''
        返回数据集长度
        :return:
        '''
        return len(self.target)

    def __getitem__(self, idx):
        '''
        根据idx 返回一行数据
        :param item:
        :return:
        '''
        return self.data[idx], self.target[idx]

    def read_from_disk(self, srcpath, name):
        '''
        从磁盘读取原始图数据
        :param srcpath:
        :param name:
        :return:
        '''
        graph_idx_path = "{}/{}/{}.graph_idx".format(srcpath, name, name)  # graph_idx
        edges_path = "{}/{}/{}.edges".format(srcpath, name, name)  # edges
        node_attrs_path = "{}/{}/{}.node_attrs".format(srcpath, name, name)  # node_attrs
        graph_labels_path = "{}/{}/{}.graph_labels".format(srcpath, name, name)  # graph_labels

        # 读取文件  pd.read_csv 的 速度比 np.genfrontxt 快很多
        graph_idx = pd.read_csv(graph_idx_path, header=None).values.flatten().astype('int')
        edges = pd.read_csv(edges_path, header=None).values.astype('int')
        graph_labels = pd.read_csv(graph_labels_path, header=None).values.flatten().astype('int')
        if os.path.exists(node_attrs_path):  # 是否存在节点属性文件
            node_attrs = pd.read_csv(node_attrs_path, header=None).values
        else:  # 节点无属性，则用单位矩阵替换
            nodes_attrs = None

        return graph_idx, edges, graph_labels, node_attrs

    def split_multi_graph(self, graph_idx, edges, graph_labels, node_attrs, available_graph_idx):
        '''
        将原始大图划分成各个子图
        :param graph_idx: graph 的索引
        :param edges: 连接边
        :param graph_labels: 图的label
        :param node_attrs:
        :return:
        '''
        data = list()
        node_idx = np.arange(1, graph_idx.shape[0] + 1)
        available_graph_idx = np.unique(graph_idx) if available_graph_idx is None else available_graph_idx
        for gidx in available_graph_idx:
            node = node_idx[graph_idx == gidx]  # 切分出节点编号
            node_length = node.shape[0]  # 节点数量
            mask = np.isin(edges, node).all(axis=1)  # 切分出对应的连接
            edge = edges[mask]
            edge_sub_min = edge - np.min(edge)  # 从0开始编号
            g = dgl.DGLGraph()
            g.add_nodes(node_length)
            g.add_edges(edge_sub_min[:, 0], edge_sub_min[:, 1])
            g.ndata['h'] = np.eye(node_length, dtype=int) if node_attrs is None else node_attrs[node - 1]
            data.append(g)
        y = np.where(graph_labels == -1, 1, 0)[available_graph_idx-1].tolist()
        return data, y

if __name__ == "__main__":
    # execute only if run as a script
    os.chdir('../../')
    graph_dataset = GraphDataset('data/', 'FRANKENSTEIN')
    print(graph_dataset)
