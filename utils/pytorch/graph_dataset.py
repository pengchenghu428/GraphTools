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
from tqdm import tqdm
import torch as th
import dgl
import os

from sklearn.preprocessing import OneHotEncoder


class GraphDataset(object):
    """
    加载图数据
    """
    def __init__(self,
                 srcpath,
                 name="FRANKENSTEIN",
                 available_graph_idx=None,
                 type="Train",
                 node_attr_type=0,
                 device=None):
        '''
        实现初始化方法，在初始化的时候将数据载入
        :param srcpath: 数据集文件夹位置
        :param name: 数据集名称
        :param available_graph_idx: 选择部分图
        :param type: 数据集类型名称
        :param node_attr_type: 如果 .node_attrs存在，则使用node_attrs特征
                                如果不存在，则1代表使用单位矩阵，0代表使用Node_label || node_degree
        '''
        print('Loading {} {} dataset...'.format(name, type))
        self.node_attr_type = node_attr_type
        self.device = device

        # 从磁盘读取数据
        graph_idx, edges, graph_labels, node_attrs, node_labels = self.read_from_disk(srcpath, name)

        # 划分多张图数据
        data, target = self.split_multi_graph(graph_idx, edges, graph_labels,
                                              node_attrs, node_labels, available_graph_idx)
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

    # 从磁盘读取数据文件
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
        node_labels_path = "{}/{}/{}.node_labels".format(srcpath, name, name)  # node_labels
        graph_labels_path = "{}/{}/{}.graph_labels".format(srcpath, name, name)  # graph_labels

        # 读取文件  pd.read_csv 的 速度比 np.genfrontxt 快很多
        graph_idx = pd.read_csv(graph_idx_path, header=None).values.flatten().astype('int')
        edges = pd.read_csv(edges_path, header=None).values.astype('int')
        graph_labels = pd.read_csv(graph_labels_path, header=None).values.flatten().astype('int')

        node_labels = pd.read_csv(node_labels_path, header=None).values[:, -1] if os.path.exists(node_labels_path) else None
        node_attrs = pd.read_csv(node_attrs_path, header=None).values if os.path.exists(node_attrs_path) else None

        return graph_idx, edges, graph_labels, node_attrs, node_labels

    # 拆分多张图
    def split_multi_graph(self, graph_idx, edges, graph_labels,
                          node_attrs, node_labels, available_graph_idx):
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
        max_node_length = pd.Series(graph_idx).value_counts().iloc[0]  # Graph 最大节点数目

        # One-Hot node_label
        if not node_labels is None:
            onehot_encoder = OneHotEncoder(categories='auto').fit(node_labels.reshape(-1, 1))
            node_labels = onehot_encoder.transform(node_labels.reshape(-1, 1)).toarray() if not node_labels is None else None

        # 连接node_label和node_attrs
        if not node_attrs is None and not node_labels is None:
            node_attrs = np.concatenate([node_attrs, node_labels], axis=1)

        for gidx in tqdm(available_graph_idx):
            node = node_idx[graph_idx == gidx]  # 切分出节点编号
            node_length = node.shape[0]  # 节点数量
            mask = np.isin(edges, node).all(axis=1)  # 切分出对应的连接
            edge = edges[mask]
            edge_sub_min = edge - np.min(edge) if not len(edge)==0 else edge
            # 生成 DGLGraph
            g = dgl.DGLGraph()
            g.add_nodes(node_length)
            g.add_edges(edge_sub_min[:, 0], edge_sub_min[:, 1])
            g = dgl.add_self_loop(g)  # 添加节点自循环
            # 节点特征
            if node_attrs is None:  # 无节点特征信息
                if self.node_attr_type == 1:  # 采用单位矩阵
                    g.ndata['h'] = np.eye(N=node_length, M=max_node_length, dtype=int)  # 节点特征矩阵
                else: # 采用度矩阵和节点特征
                    degree = g.in_degrees().numpy().reshape(-1, 1)
                    node_label_one_hot = node_labels[node-1]
                    g.ndata['h'] = th.from_numpy(np.concatenate((node_label_one_hot, degree), axis=1))
            else:  # 存在节点特征信息
                # g.ndata['h'] = node_attrs[node - 1]
                degree = g.in_degrees().numpy().reshape(-1, 1)
                node_attr = node_attrs[node - 1]
                g.ndata['h'] = th.from_numpy(np.concatenate((node_attr, degree), axis=1))
            if self.device:
                g = g.to(self.device)
            data.append(g)
        y = graph_labels[available_graph_idx-1]
        y[y == -1] = 0
        y = y - np.min(y)  # 类别归一化到0 二分类（0，1） 多分类（0，1，2，...）
        return data, y.tolist()

    # 获取节点特征矩阵维度
    def get_feature_size(self):
        return self.data[0].ndata['h'].size()[-1]


if __name__ == "__main__":
    # execute only if run as a script
    os.chdir('../../')
    graph_dataset = GraphDataset('data/', 'NCI1', node_attr_type=0)
    print(graph_dataset)
