#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> sum_max_pooling
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/12/9 17:44
@Desc   ：
=================================================='''

import torch as th
import torch.nn as nn
from dgl.nn.pytorch import SumPooling, MaxPooling


class SumMaxPooling(nn.Module):

    def __init__(self):
        super(SumMaxPooling, self).__init__()

    def forward(self, graph, feat):
        with graph.local_scope():
            sum_pool = SumPooling()(graph, feat)
            max_pool = MaxPooling()(graph, feat)
            pool = th.cat([sum_pool, max_pool], dim=-1)
            return pool
