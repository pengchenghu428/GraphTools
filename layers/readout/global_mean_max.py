#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> global_mean_max
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/12 15:50
@Desc   ：Keras 实现 Mean || Max Readout 层
=================================================='''

from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
import tensorflow as tf


class GlobalMeanMaxPooling(Layer):

    def __init__(self,
                 type='mean|max',
                 **kwargs):
        super(GlobalMeanMaxPooling).__init__(kwargs)

    def build(self, input_shape):
        # input = [X, scoring] or input = X

        self.build = True

    def call(self, inputs):
        assert len(inputs) >= 2  # 检查输入维度 input=[X, scoring], 如果没有scoring， 用None填充
        X, scoring = inputs

        x_mean = K.mean(X, axis=0)  # 论文中的 1/N * (sum(x_i))

        if scoring is None:  # 如果没有提供scoring，默认是排序后的X， 最大取第一行
            x_max = K.gather(X, 0)  # 一维
        else:
            max_indices = tf.nn.top_k(scoring, 1).indices
            x_max = K.gather(X, max_indices)  # 二维
            x_max = K.flatten(x_max)  # 展开成一维

        if self.type == 'mean':
            output = x_mean  # mean
        elif self.type == 'max':
            output = x_max  # max
        else:
            output = K.concatenate([x_mean, x_mean])  # mean || max
        return output

    def compute_output_shape(self, input_shape):
        X_dims = input_shape[0][-1]
        if self.type == 'mean':
            output_dim = X_dims
        elif self.type == 'max':
            output_dim = X_dims
        else:
            output_dim = 2 * X_dims
        return (output_dim, )   # 返回的是一维数组
