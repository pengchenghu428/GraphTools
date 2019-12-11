#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> self_attention_pooling
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/27 10:21
@Desc   ：
=================================================='''

import torch.nn as nn
import dgl

from dgl import BatchedDGLGraph


class SelfAttentionPooling(nn.Module):

    def __init__(self, k, in_feats, feat_nn=None):
        super(SelfAttentionPooling, self).__init__()
        self.k = k

        self.att = nn.Linear(in_feats, 1)

    def forward(self, graph, feat):
        with graph.local_scope():
            #
            gate = self.att(feat)
            assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
            feat = self.feat_nn(feat) if self.feat_nn else feat

            graph.ndata['gate'] = gate
            gate = dgl.softmax_nodes(graph, 'gate')  # 节点得分
            graph.ndata['gate'] = gate

            graph.ndata['h'] = feat
            # Sort nodes according to their gate score
            ret = dgl.topk_nodes(graph, 'gate', self.k)[0].view(
                -1, self.k, feat.shape[-1])
            graph.ndata.pop('gate')

            if isinstance(graph, BatchedDGLGraph):
                return ret
            else:
                return ret.squeeze(0)
