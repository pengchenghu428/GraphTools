#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> layers -> graphpool -> sag_pooling
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/11 20:06
@Desc   ：keras 实现Self-Attention Graph Pooling  2019
=================================================='''

from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
import tensorflow as tf


class SAGraphPooling(Layer):

    def __init__(self,
                 rate=0.5,
                 attn_heads=1,
                 attn_heads_reduction='average',
                 activation='softmax',
                 attn_initializer='glorot_uniform',
                 attn_regularizer=None,
                 activity_regularizer=None,
                 attn_constraint=None,
                 **kwargs):
        super(SAGraphPooling, self).__init__(**kwargs)

        self.rate = rate  # 神经元个数
        self.attn_heads = attn_heads
        self.attn_heads_reduction=attn_heads_reduction
        self.activation = activations.get(activation)  # 激活函数
        self.attn_initializer = initializers.get(attn_initializer)  # 注意力分数初始化方法
        self.attn_regularizer = regularizers.get(attn_regularizer)  # 施加在注意力分数上的正则项
        self.activity_regularizer = regularizers.get(activity_regularizer)  # 施加在输出上的正则项
        self.attn_constraint = constraints.get(attn_constraint)  # 施加在注意力分数上的约束项

        self.attn_kernels = []  # 注意力得分集合

    # 初始化weight
    def build(self, input_shapes):
        assert len(input_shapes) >= 2  # 检查输入维度 input=[X, A]
        X_shape, A_shape = input_shapes  # [(None, N, F), (None, N, N)]

        for head in range(self.attn_heads):
            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(X_shape[-1], 1),
                                           initializer=self.attn_initializer,
                                           regularizer=self.attn_regularizer,
                                           constraint=self.activity_regularizer,
                                           name='attn_kernel_self_{}'.format(head), )  # (F, 1)
            self.attn_kernels.append((attn_kernel_self))

        # self.batch_size = X_shape[0]
        self.origin_nodes = X_shape[1]  # 原节点数目
        self.keep_nodes = max(int(self.rate * self.origin_nodes), 1)  # 保存节点的数目
        self.features_dim = X_shape[-1]
        self.build = True  # 必须将 self.built 设置为True, 以保证该 Layer 已经成功 build

    # 用来执行 Layer 的职能, 即当前 Layer 所有的计算过程均在该函数中完成
    def call(self, inputs, mask=None):
        Xs = inputs[0]  # 节点特征 (None, N, F)
        As = inputs[1]  # 邻接矩阵 (None, N, N)

        scores = []

        for head in range(self.attn_heads):
            attention_kernel = self.attn_kernels[head]  # 注意力核 维度（F, 1）

            # 计算注意力网络的输入
            support = tf.matmul(As, Xs)  # A * X (None, N, F)
            support = K.dot(support, attention_kernel)  # A * X * attn (None, N, 1)  # K.dot 和 tf.matmul 区别
            score = self.activation(support)
            scores.append(score)

        if self.attn_heads_reduction == 'average':
            scoring = K.mean(K.stack(scores), axis=0)  # (None, N, 1)
        else:  # 可以有max/sum...
            scoring = K.mean(K.stack(scores), axis=0)  # (None, N, 1)
        # scoring = K.reshape(scoring, (self.batch_size, self.origin_nodes))  # reshape 不能处理None
        scoring = K.squeeze(scoring, axis=2)  # K.squeeze 缩减一维，要求被缩减的维度的长度为1

        # 按照输入的最后一个维度排序，选取top keep_nodes的值
        keep_values = tf.nn.top_k(scoring, k=self.keep_nodes).values
        keep_indices = tf.nn.top_k(scoring, k=self.keep_nodes).indices

        Xs_new, As_new = [], []

        for idx in range(self.batch_size):
            X, A = K.gather(Xs, idx), K.gather(As, idx)  # 拿出对应位置的X, A
            keep_indice = K.gather(keep_indices, idx)  # 取出对应的重要节点索引
            X_out = K.gather(X, keep_indice)  # 特征矩阵行选择
            A_out = tf.gather(A, keep_indice)  # 邻接矩阵行选择
            A_out = tf.gather(A_out, keep_indice, axis=1)  # 邻接矩阵列选择  K.gather 没有axis参数
            Xs_new.append(X_out)  # 保存结果
            As_new.append(A_out)

        Xs_out = K.stack(Xs_new)  # 堆叠输出结果， (None, kN, F)
        As_out = K.stack(As_new)  # (None, kN, kN)

        return [Xs_out, As_out, keep_values]

    # 计算输出shape
    def compute_output_shape(self, input_shape):
        X_shape = (self.batch_size, self.keep_nodes, self.features_dim)  # (None, kN, F)
        A_shape = (self.batch_size, self.keep_nodes, self.keep_nodes)  # (None, kN, kN)
        scoring_shape = (self.batch_size, self.keep_nodes)  # (None, kN)
        output_shape = [X_shape, A_shape, scoring_shape]
        return output_shape



