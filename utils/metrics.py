#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> test
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/11 20:06
@Desc   ：
=================================================='''

import numpy as np


def precise_score(y, y_pred):
    '''
    精确率
    :param y:
    :param y_pred:
    :return:
    '''
    assert len(np.unique(y)) == 2, "不是二分类，无法计算precise_score"
    TP = ([y == 1] & [y_pred == 1]).astype('int').sum()
    TP_FP = ([y_pred == 1]).astype('int').sum()
    return TP / TP_FP


def recall_score(y, y_pred):
    '''
    召回率
    :param y:
    :param y_pred:
    :return:
    '''

    assert len(np.unique(y)) == 2, "不是二分类，无法计算precise_score"
    TP = ([y == 1] & [y_pred == 1]).astype('int').sum()
    TP_FP = ([y == 1]).astype('int').sum()
    return TP / TP_FP


def accuracy_score(y, y_pred):
    '''
    准确率
    :param y:
    :param y_pred:
    :return:
    '''
    true = ([y == y_pred]).astype('int').sum()
    return true / len(y)

def Fl_score(y, y_pred, alpha=1):
    '''
    F1 值
    :param y:
    :param y_pred:
    :return:
    '''
    assert len(np.unique(y)) == 2, "不是二分类，无法计算F1_score"
    p= precise_score(y, y_pred)
    r = recall_score(y, y_pred)
    return (alpha*alpha + 1)*(p*r) / (p+r)


def auc_roc_score(y, y_pred):
    '''
    roc 值
    :param y:
    :param y_pred:
    :return:
    '''
    print()


def mean_square_error(y, y_pred):
    '''
    均方误差
    :param y:
    :param y_pred:
    :return:
    '''
    assert len(y)==len(y_pred), "长度不同，无法计算"
    return np.mean(np.square(y-y_pred))


def root_mean_square_error(y, y_pred):
    '''
    均方根误差
    :param y_y_pred:
    :return:
    '''
    assert len(y) == len(y_pred), "长度不同，无法计算"
    return np.sqrt(mean_square_error(y, y_pred))

