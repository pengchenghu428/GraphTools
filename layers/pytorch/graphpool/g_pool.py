#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> g_pool
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/12/11 10:17
@Desc   ：g_pool
=================================================='''

import dgl
import torch as th
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GPooling(nn.Module):
    def __init__(self, k, in_feats, activation=F.sigmoid):
        super(GPooling, self).__init__()
        self.k = k  # 保留节点数目
        self._in_feats = in_feats  # 特征维度
        self.activation = activation

        self.weight = Parameter(th.Tensor(in_feats, 1))

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = init.calculate_gain('leaky_relu')
        init.xavier_normal_(self.weight, gain=gain)

    def forward(self, graph, feat):
        with graph.local_scope():
            graph.ndata['h'] = feat

            gate = feat.mm(self.weight)
            gate = gate / th.sqrt(self.weight.t().mm(self.weight))
            graph.ndata['gate'] = gate  # 节点得分

            graphs = dgl.unbatch(graph)

            def get_sub_graph(graph):
                valid_idx_single = dgl.topk_nodes(graph, 'gate', self.k)[1].view(-1)[0:min(self.k, graph.number_of_nodes())]
                sub_graph = graph.subgraph(valid_idx_single)
                sub_graph.copy_from_parent()
                return sub_graph

            sub_graphs = [get_sub_graph(graph) for idx, graph in enumerate(graphs)]  # 列表推导式生成子图
            sub_graph = dgl.batch(sub_graphs)
            feat = sub_graph.ndata['h'] * self.activation((sub_graph.ndata['gate']))

            return sub_graph, feat
