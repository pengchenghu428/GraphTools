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
from dgl.nn.pytorch.utils import Identity


class GIAttPNP(nn.Module):

    def __init__(self,
                 in_feats,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(GIAttPNP, self).__init__()

        self._in_feats = in_feats

        self.attn_l = nn.Linear(in_feats, 1)
        self.attn_r = nn.Linear(in_feats, 1)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if residual:
            self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        self.activation = activation

    def forward(self, graph, feat):
        graph = graph.local_var()
        feat = self.feat_drop(feat)

        el = self.attn_l(feat)
        er = self.attn_r(feat)

        graph.ndata.update({'ft': feat, 'el': el, 'er': er})
        # compute edge attention
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        # graph.edata['a'] = self.attn_drop(dgl.nn.pytorch.edge_softmax(graph, e))
        graph.edata['a'] = self.attn_drop(e)
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.ndata['ft']

        # residual
        if self.res_fc is not None:
            resval = self.res_fc(feat)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)

        return rst
