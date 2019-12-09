#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> loss
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/12/2 13:24
@Desc   ：实现自定义损失函数
=================================================='''

import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# 结构损失误差
class ConstructLoss(nn.Module):

    def __init__(self):
        super(ConstructLoss, self).__init__()

    def forward(self, graph):
        graphs = dgl.unbatch(graph)

        loss = 0
        for graph in graphs:
            # 预测值
            z = graph.ndata['h']
            z = th.sigmoid(z.mm(z.t()))
            z_line = z.view(-1, 1)  # 预测成1的概率
            z_line_ = th.ones_like(z_line, device=z.device) - z_line  # 预测成0的概率
            pred = th.cat([z_line_, z_line], dim=1).view(-1, 2)  # 展开成1维
            # 真实值
            adj = graph.adjacency_matrix(ctx=z.device)
            loop = th.eye(z.size()[0]).to(device=z.device)
            adj_loop = loop + adj
            target = adj_loop.view(-1).long()
            # 节点权重
            pos_weight = (z.size()[0]*z.size()[0] - th.sum(target).float()) / th.sum(target).float()
            weight = th.from_numpy(np.array((1, pos_weight.item()))).to(adj.device).float()
            # 图重构损失
            loss += F.cross_entropy(pred, target, weight=weight).item()
        # 图平均重构损失
        loss /= len(graphs)

        return loss


# 结构误差+分类误差
class CustomizedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomizedLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true, graphs):
        classify_loss = F.cross_entropy(y_pred, y_true)
        construct_loss = [ConstructLoss()(graph) for graph in graphs]
        construct_loss = np.sum(construct_loss)
        loss = classify_loss + self.alpha*construct_loss
        return loss, classify_loss, construct_loss
