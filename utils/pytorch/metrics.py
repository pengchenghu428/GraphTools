#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> metrics
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/25 10:31
@Desc   ：评价指标
=================================================='''

import numpy as np
from sklearn.metrics import  precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, confusion_matrix


def evaluate_binary_classification(y, y_pred):
    '''
    评估二分类模型
    :param y: 标签值
    :param y_pred: 预测值
    :return:dict
    '''
    res = dict()
    y_pred_label = np.where(y_pred > 0.5, 1, 0)
    res['precise_score'] = precision_score(y, y_pred_label)
    res['recall_score'] = recall_score(y, y_pred_label)
    res['accuracy_score'] = accuracy_score(y, y_pred_label)
    res['F1_score'] = f1_score(y, y_pred_label)
    res['roc_auc_score'] = roc_auc_score(y, y_pred)
    res['confusion_matrix'] = confusion_matrix(y, y_pred_label)
    return res