#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> train
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/25 9:35
@Desc   ：训练中的函数调用
=================================================='''

import torch
import numpy as np
import torch.nn as nn


# 封装训练过程
def do_train(model, device, train_loader, optimizer, criterion):
    '''
    模型训练
    :param model: 模型
    :param device: 设备位置
    :param train_loader: 训练集数据
    :param train_length: 训练集数据
    :param optimizer: 优化器
    :param criterion: 损失函数/评价指标
    :param epoch: 当前迭代轮次
    :return:
    '''
    model.train()
    loss = 0
    train_size = len(train_loader.dataset)  # 训练大小
    trained_size = 0  # 已训练大小
    correct, batch_correct = 0, 0
    progress_bar = "\r\t{}/{} [{}{}] - loss: {:.4f} - acc: {:.4f}"
    for batch_idx, (data, target) in enumerate(train_loader):
        train_loader.dataset.target = []

        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        batch_loss = criterion(output, target)  # batch 损失
        loss += batch_loss  # 总体损失
        batch_loss.backward()  # 梯度下降
        optimizer.step()  # 参数更新

        # 训练进度条显示
        trained_size += len(target)
        trained_bar = round(trained_size/train_size * 20)
        arrow, dot = ">"*trained_bar, "."*(20-trained_bar)
        batch_loss /= len(target)
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        batch_correct = pred.eq(target.view_as(pred)).sum().item()
        correct += batch_correct
        print(progress_bar.format(trained_size, train_size, arrow, dot,
                                  batch_loss.item(), batch_correct/len(target)), end="")
    # 整体的loss显示
    loss /= train_size
    loss = loss.item()
    acc = correct / trained_size
    print(progress_bar.format(trained_size, train_size, arrow, dot, loss, acc), end="")
    return loss, acc


# 封装测试过程
def do_test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target)  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_loss = test_loss.item()
    test_acc = correct/len(test_loader.dataset)
    print(' - val_loss: {:.4f} - val_acc: {:.4f}'.format(test_loss, test_acc))
    return test_loss, test_acc

