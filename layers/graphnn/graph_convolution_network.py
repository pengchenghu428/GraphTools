#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> layers -> graphnn -> graph_convolution_network
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/11 20:06
@Desc   ：keras 实现Graph Convolution Network 2017
=================================================='''

from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
import tensorflow as tf


class GraphConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, units, support=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)

        self.units = units  # 神经元个数
        self.activation = activations.get(activation)  # 激活函数
        self.use_bias = use_bias  # 是否使用偏置项
        self.kernel_initializer = initializers.get(kernel_initializer)  # 权值初始化方法
        self.bias_initializer = initializers.get(bias_initializer)  # 偏置初始化方法
        self.kernel_regularizer = regularizers.get(kernel_regularizer)  # 施加在权重上的正则项
        self.bias_regularizer = regularizers.get(bias_regularizer)  # 施加在偏置向量上的正则项
        self.activity_regularizer = regularizers.get(activity_regularizer)  # 施加在输出上的正则项
        self.kernel_constraint = constraints.get(kernel_constraint)  # 施加在权重上的约束项
        self.bias_constraint = constraints.get(bias_constraint)  # 施加在偏置上的约束项

        self.support = support

        assert support >= 1

    # 初始化weight
    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 2  # 检查X的输入维度
        input_dim = features_shape[1]  # 特征维度

        # 权重
        self.kernel = self.add_weight(shape=(input_dim * self.support,  #
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # 偏置
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.build = True  # 必须将 self.built 设置为True, 以保证该 Layer 已经成功 build

    # 用来执行 Layer 的职能, 即当前 Layer 所有的计算过程均在该函数中完成
    def call(self, inputs, mask=None):
        features, basis = inputs[0], inputs[1]  # X, A [(None, N, F), (None, N, N)]

        supports = list()

        for i in range(self.support):
            supports.append(tf.matmul(basis[i], features))  # A * X (None, N, F)
        supports = K.concatenate(supports, axis=2)  # 在给定轴上将一个列表中的张量串联为一个张量  (None, N, k*F)
        output = K.dot(supports, self.kernel)  # A * X * W  (None, N, k*F_)

        if self.use_bias:  # 偏置
            output += self.bias
        return self.activation(output)  # sigma(A * X * W)

    # 计算输出shape
    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0]  # input = [X, A] [(None, N, F), (None, N, N)]
        batch_size = input_shape[0][0]
        output_shape = (batch_size, features_shape[1], self.units)  # (None, N, F_)
        return output_shape

    # 输出当前网络配置
    def get_config(self):
        config = {'units': self.units,
                  'support': self.support,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

