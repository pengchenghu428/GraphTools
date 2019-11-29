#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> attrntion_sum_ppnp_conv
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/29 17:00
@Desc   ：Graph Isomorphsim Attention Propagation of Neural Predictions Network
         图同构注意力神经预测个性化传播网络
=================================================='''

import torch as th
import torch.nn as nn
import dgl.function as fn
import dgl


class GIAttPNP(nn.Module):
    r"""Attention Sum Approximate Personalized Propagation of Neural Predictions

        Parameters
        ----------
        k : int
            Number of iterations :math:`K`.
        in_feats: int
            Number of feat_dim
        alpha : float
            The teleport probability :math:`\alpha`.
        edge_drop : float, optional
            Dropout rate on edges that controls the
            messages received by each node. Default: ``0``.
        """

    def __init__(self,
                 k,
                 in_feats,
                 init_alpha,
                 aggregator_type='sum',
                 edge_drop=0.,
                 learn_alpha=False):
        super(GIAttPNP, self).__init__()

        self.k = k  # 迭代次数
        self.in_feats = in_feats  # 输入特征维度
        self.edge_drop = edge_drop  #

        self.layers = nn.ModuleList()
        for i in range(k):
            self.layers.append(nn.Linear(in_feats, 1))

        # 聚合类型
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))

        # 确定参数 alpha 是否可学习
        if learn_alpha:
            self.alpha = th.nn.Parameter(th.FloatTensor([init_alpha]))
        else:
            self.register_buffer('alpha', th.FloatTensor([init_alpha]))

    def forward(self, graph, feat):
        graph = graph.local_var()
        graph.ndata['h'] = feat
        feat_0 = feat
        feats = list()
        for idx in range(self.k):
            gate = self.layers[idx](feat)
            assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
            graph.ndata['gate'] = gate
            gate = dgl.softmax_nodes(graph, 'gate')
            graph.ndata.pop('gate')
            graph.ndata['r'] = feat * gate
            graph.update_all(fn.copy_u('r', 'm'), self._reducer('m', 'neigh'))
            feat = (1-self.alpha) * graph.ndata['neigh'] + self.alpha * feat_0
            feats.append(feat)
        return th.stack(feats, dim=0)

