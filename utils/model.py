#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> model
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/13 16:39
@Desc   ：模型保存/加载
=================================================='''

import os
import pickle
from keras.models import model_from_json

from layers import *
from utils import *


def load_model(dir, name):
    '''
    加载现存的模型
    :param name:
    :param name:
    :return:
    '''
    architecture_path = "{}/{}/{}_model_architecture.json".format(dir, name, name)
    weight_path = "{}/{}/{}_model_weights.best.h5".format(dir, name, name)
    model = model_from_json(open(architecture_path).read(),
                            custom_objects={'GraphAttention': GraphAttention,
                                            'GraphConvolution': GraphConvolution,
                                            'DiffPooling': DiffPooling,
                                            'SAGraphPooling': SAGraphPooling,
                                            'SortPooling': SortPooling,
                                            'GlobalMeanMaxPooling': GlobalMeanMaxPooling,})
    model.load_weights(weight_path)
    return model


def save_model(dir, name, model, history=None):
    '''
    保存模型
    :param dir:路径
    :param name:模型名称
    :param model:训练好的模型
    :return:
    '''
    checkd_directory(dir)  # 检查保存文件夹是否存在

    architecture_path = "{}/{}/{}_model_architecture.json".format(dir, name, name)  # 模型结构位置
    weight_path = "{}/{}/{}_model_weights.best.h5".format(dir, name, name)  # 权重位置

    json_string = model.to_json()  # 保存模型结构
    open(architecture_path, 'w').write(json_string)

    if not os.path.exists(weight_path):  # 如果权重文件不存在，则保存权重
        model.save_weights(weight_path)

    if not history is None:  # 保存训练过程
        history_path = "{}/{}/{}_history.pkl".format(dir, name, name)
        pickle.dump(history.history, open(history_path, 'wb'))