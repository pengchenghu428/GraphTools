#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> layers -> graphnn -> graph_attention_network
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/11 20:06
@Desc   ：Keras 实现注意力机制图卷积 2018
=================================================='''

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU
import tensorflow as tf


class GraphAttention(Layer):

    def __init__(self,
                 units,
                 attn_heads=1,
                 attn_heads_reduction='concat',
                 dropout_rate=0.5,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphAttention, self).__init__(**kwargs)

        self.units = units  # 输出特征的数量
        self.attn_heads = attn_heads  # 独立计算的attention数量
        self.attn_heads_reduction = attn_heads_reduction  # 论文中Eq.5 和 公式6： concat 连接 average 平均
        self.dropout_rate = dropout_rate  # 丢失率
        self.activation = activations.get(activation)  # 激活函数
        self.use_bias = use_bias  # 是否使用偏置项

        self.kernel_initializer = initializers.get(kernel_initializer)  # 权值初始化方法
        self.bias_initializer = initializers.get(bias_initializer)  # 偏置初始化方法
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)  # 注意力初始化方法

        self.kernel_regularizer = regularizers.get(kernel_regularizer)  # 施加在权重上的正则项
        self.bias_regularizer = regularizers.get(bias_regularizer)  # 施加在偏置向量上的正则项
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)  # 施加在注意力上的正则项
        self.activity_regularizer = regularizers.get(activity_regularizer)  # 施加在输出上的正则项

        self.kernel_constraint = constraints.get(kernel_constraint)  # 施加在权重上的约束项
        self.bias_constraint = constraints.get(bias_constraint)  # 施加在偏置上的约束项
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)  # 施加在注意力上的约束项
        self.supports_masking = False

        # build 初始化时使用
        self.kernels = []   # 注意力权重集合
        self.biases = []  # 注意力偏置集合
        self.attn_kernels = []  # 注意力得分集合

        if attn_heads_reduction == 'concat':
            # 输出维度 (..., K * units)
            self.output_dim = self.units * self.attn_heads
        else:
            # 输出维度 (..., units)
            self.output_dim = self.units

    # 用来初始化定义
    def build(self, input_shape):
        assert len(input_shape) >= 2  # 检查输入维度 input=[X, A]
        F = input_shape[0][-1]  # 输入特征的维度

        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.units),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.units, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.units, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(self.units, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True  # 必须将 self.built 设置为True, 以保证该 Layer 已经成功 build

    def call(self, inputs):
        Xs = inputs[0]  # 节点特征 (batch X N X F)
        As = inputs[1]  # 邻接矩阵 (batch X N X N)

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W 维度 （F， units）
            attention_kernel = self.attn_kernels[head]  # 注意力核 维度（2*units, 1）

            # 计算注意力网络的输入
            features = K.dot(Xs, kernel)  # Xs * W (batch X N x units)

            # 计算特征联合
            # [[a_1], [a_2]]^T [[Wh_i], [Wh_j]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attention_kernel[0])  # (batch, N, units), (units, 1) -> (batch, N, 1)
            attn_for_neighs = K.dot(features, attention_kernel[1])  # (batch, N, units), (units, 1) -> (batch, N, 1)

            # 注意力机制 a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            # dense = attn_for_self + K.transpose(attn_for_neighs)  # K.transpose (batch, N, units)->(units, N, batch)
            dense = attn_for_self + tf.transpose(attn_for_neighs, perm=[0, 2, 1])  # 第一维度不变，二、三维度转置
            # print("dense0.shape:{}".format(K.int_shape(dense)))

            # 添加非线性变换
            dense = LeakyReLU(alpha=0.2)(dense)
            # print("dense1.shape:{}".format(K.int_shape(dense)))

            # 激活前的掩码值
            mask = -10e9 * (1.0 - As)
            # print("mask.shape:{}".format(K.int_shape(mask)))
            dense = dense + mask  # bug

            # softmax 获得注意力分数
            dense = K.softmax(dense)

            # 对特征和注意力分数应用dropout
            dropout_attn = Dropout(self.dropout_rate)(dense)  # (batch, N, N)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (batch, N, units)

            # 注意力权重组合特征值
            # node_features = K.dot(dropout_attn, dropout_feat)  # K.dot 会提前resize矩阵，导致计算不成功
            node_features = tf.matmul(dropout_attn, dropout_feat)

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # 将输出添加到最终的输出
            outputs.append(node_features)

        # 根据不同的维度约见方法聚集各个注意力的输出
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (batch X N x Kunits)
        else:
            output = K.mean(K.stack(outputs), axis=0)  # (batch X N x F')

        output = self.activation(output)  # 激活函数
        return output

    # 计算输出维度
    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], input_shape[0][1], self.output_dim
        return output_shape

    # 输出当前网络配置
    def get_config(self):

        config = {'units': self.units,
                  'attn_heads': self.attn_heads,
                  'attn_heads_reduction': self.attn_heads_reduction,
                  'dropout_rate': self.dropout_rate,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'attn_kernel_initializer': initializers.serialize(self.attn_kernel_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'attn_kernel_regularizer': regularizers.serialize(self.attn_kernel_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'attn_kernel_constraint': constraints.serialize(self.attn_kernel_constraint)}

        base_config = super(GraphAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


