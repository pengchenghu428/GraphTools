#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> diff_pool
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/12/11 10:16
@Desc   ：diff_pool
=================================================='''

import torch as th
import torch.nn as nn
import dgl
from dgl import BatchedDGLGraph


class DiffPooling(nn.Module):
    def __init__(self, k, in_feats):
        super(DiffPooling, self).__init__()
        self.k = k
        self._in_feats = in_feats

        self.weight = nn.Linear(in_feats, k)

    def forward(self, graph, feat):
        graph = graph.local_var()
        norm = th.pow(graph.in_degrees().float().clamp(min=1), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = th.reshape(norm, shp).to(feat.device)
        feat = feat * norm

        if self._in_feats > self.k:
            # mult W first to reduce the feature size for aggregation.
            feat = th.matmul(feat, self.weight)
            graph.ndata['h'] = feat
            graph.update_all(dgl.copy_src(src='h', out='m'),
                             dgl.sum(msg='m', out='h'))
            rst = graph.ndata['h']
        else:
            # aggregate first then mult W
            graph.ndata['h'] = feat
            graph.update_all(dgl.copy_src(src='h', out='m'),
                             dgl.sum(msg='m', out='h'))
            rst = graph.ndata['h']
            rst = th.matmul(rst, self.weight)

        rst = rst * norm
