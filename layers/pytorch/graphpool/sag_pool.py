#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> self_attention_pooling
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/27 10:21
@Desc   ：
=================================================='''

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter


class SelfAttentionPooling(nn.Module):

    def __init__(self, k, in_feats, activation=F.sigmoid):
        super(SelfAttentionPooling, self).__init__()
        self.k = k
        self.in_feats = in_feats  # 特征维度
        self.activation = activation

        self.weight = Parameter(th.Tensor(in_feats, 1))

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = init.calculate_gain('leaky_relu')
        init.xavier_normal_(self.weight, gain=gain)

    def forward(self, graph, feat):
        with graph.local_scope():
            # 考虑结构信息
            in_degree = graph.in_degrees() + th.eye(feat.size()[0]).to(device=feat.device)
            norm = th.pow(in_degree.float().clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp).to(feat.device)
            adj_loop = graph.adjacency_matrix(ctx=feat.device) + th.eye(norm.size()[0]).to(device=feat.device)
            adj_loop = adj_loop.view(-1).long()
            feat = norm.mm(adj_loop).mm(adj_loop)
            graph.ndata['h'] = feat

            # 注意力机制
            gate = feat.mm(self.weight)
            gate = self.activation(gate)
            graph.ndata['gate'] = gate  # 节点得分

            graphs = dgl.unbatch(graph)

            # 获取子图
            def get_sub_graph(graph):
                valid_idx_single = dgl.topk_nodes(graph, 'gate', self.k)[1].view(-1)[
                                   0:min(self.k, graph.number_of_nodes())]
                sub_graph = graph.subgraph(valid_idx_single)
                sub_graph.copy_from_parent()
                return sub_graph

            # 子图重新组合
            sub_graphs = [get_sub_graph(graph) for idx, graph in enumerate(graphs)]  # 列表推导式生成子图
            sub_graph = dgl.batch(sub_graphs)
            feat = sub_graph.ndata['h'] * sub_graph.ndata['gate']

            return sub_graph, feat

