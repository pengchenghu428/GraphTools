#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> wavenet
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/12/2 9:56
@Desc   ：实现 WaveNet 模式的Graph Neural Network
=================================================='''

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from dgl.nn.pytorch.utils import Identity


class WaveGIN(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 init_eps=0,
                 learn_eps=False,
                 residual=False):
        super(WaveGIN, self).__init__()

        # 学习参数
        self.w_coff = nn.Linear(in_feats, out_feats)
        self.w_message = nn.Linear(in_feats, out_feats)

        # 节点汇聚形式
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))

        # 确定eps是否需要训练
        if learn_eps:
            self.eps = th.nn.Parameter(th.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', th.FloatTensor([init_eps]))

        # 是否以残差形式训练
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, graph, feat):
        # 汇聚周围节点信息
        graph = graph.local_var()
        graph.ndata['h'] = feat
        graph.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'neigh'))
        feat = (1 + self.eps) * feat + graph.ndata['neigh']  # 结构信息

        coff = self.w_coff(feat)  # 系数：保留多少信息
        coff = F.sigmoid(coff)
        message = self.w_message(feat)  # 信息：新的特征信息
        message = F.tanh(message)

        rst = coff.mul(message)

        # 残差
        if self.res_fc is not None:
            resval = self.res_fc(feat)
            rst = rst + resval

        return rst