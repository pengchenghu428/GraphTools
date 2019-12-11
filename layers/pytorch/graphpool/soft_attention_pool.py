#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> soft_attention_pool
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/25 20:09
@Desc   ： Soft Attention Pooling
=================================================='''

import torch.nn as nn
import dgl
import torch


class SoftAttentionPooling(nn.Module):
    def __init__(self, k, in_feats, feat_nn=None):
        super(SoftAttentionPooling, self).__init__()
        self.k = k  # 转换后的节点数目
        self.in_feats = in_feats  # 输入节点特征维度
        self.feat_nn = feat_nn

        self.layers = nn.ModuleList()
        for i in range(k):
            self.layers.append(nn.Linear(in_feats, 1))

    def forward(self, graph, feat):
        r"""Compute sort pooling.

        Parameters
        ----------
        graph : DGLGraph or BatchedDGLGraph
            The graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(1, n_heads,  D)` (if
            input graph is a BatchedDGLGraph, the result shape
            would be :math:`(B, n_heads, D)`.
        """
        readouts = list()
        with graph.local_scope():
            for i in range(self.k):
                gate = self.layers[i](feat)
                assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
                tmp_feat = self.feat_nn(feat) if self.feat_nn else feat

                graph.ndata['gate'] = gate
                gate = dgl.softmax_nodes(graph, 'gate')
                graph.ndata.pop('gate')

                graph.ndata['r'] = tmp_feat * gate
                readout = dgl.sum_nodes(graph, 'r')
                graph.ndata.pop('r')
                readouts.append(readout)

            out = torch.stack(readouts, dim=1)
            out = out.view(-1, self.k, self.in_feats)
            return out


