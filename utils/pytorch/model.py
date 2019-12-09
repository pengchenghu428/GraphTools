#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> model
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/25 9:33
@Desc   ：torch 模型保存与加载
=================================================='''
import os
import pickle
import torch
from utils.pytorch.process import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 封装预测
def do_predict(model, device, test_dataset, test_loader=None, batch_size=256):
    model.eval()
    if test_loader is None:
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=collate)
    y_pred = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # target = target.to(device)
            output, graph = model(data)
            # pred = output.max(1, keepdim=True)[1].view(-1)  # 找到概率最大的下标
            proba = output[:, 1].view(-1)
            y_pred.append(proba)
    return torch.cat(y_pred, dim=-1).view(-1).cpu().numpy()


def plot_train_process(dir_path, name):
    '''
    绘制训练过程的图
    :param dir_path:
    :param name:
    :return:
    '''
    history_path = "{}/{}/{}_history.pkl".format(dir_path, name, name)
    if not os.path.exists(history_path):
        print("Plot_train_process: 训练数据不存在")

    history = pickle.load(open(history_path, 'rb'))

    for item in['loss', 'acc']:
        plt.plot(history[item])
        plt.plot(history['val_{}'.format(item)])

        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'])

        fig = plt.gcf()
        fig_path = history_path.replace('.pkl', '_{}.png'.format(item))
        fig.savefig(fig_path, dpi=100)

        plt.cla()
