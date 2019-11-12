#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> layers -> graphpool -> sag_pooling
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/11 20:06
@Desc   ：keras 实现Self-Attention Graph Pooling
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
                 attn_heads_reduction='mean',
                 activation='softmax',
                 attn_initializer='glorot_uniform',
                 attn_regularizer=None,
                 activity_regularizer=None,
                 attn_constraint=None,
                 **kwargs):
        super(SAGraphPooling, self).__init__(**kwargs)

        self.rate = rate  # 神经元个数
        self.attn_heads=attn_heads,
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
        X_shape, A_shape = input_shapes

        for head in range(self.attn_heads):
            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(X_shape[-1], 1),
                                           initializer=self.attn_kernel_initializer,
                                           regularizer=self.attn_kernel_regularizer,
                                           constraint=self.attn_kernel_constraint,
                                           name='attn_kernel_self_{}'.format(head), )  # (F, 1)
            self.attn_kernels.append((attn_kernel_self))

        self.origin_nodes = X_shape[0]
        self.build = True  # 必须将 self.built 设置为True, 以保证该 Layer 已经成功 build

    # 用来执行 Layer 的职能, 即当前 Layer 所有的计算过程均在该函数中完成
    def call(self, inputs, mask=None):
        X = inputs[0]  # 节点特征 (N X F)
        A = inputs[1]  # 邻接矩阵 (N X N)

        scores = []

        for head in range(self.attn_heads):
            attention_kernel = self.attn_kernels[head]  # 注意力核 维度（F, 1）

            # 计算注意力网络的输入
            support = K.dot(A, X)  # A * X
            support = K.dot(support, attention_kernel)  # A * X * attn (NXN, NXF, FX1)->(N, 1)
            score = self.activation(support)
            scores.append(score)

        if self.attn_heads_reduction == 'mean':
            scoring = K.mean(K.stack(scores), axis=0)  # (N x 1)
        else:
            scoring = K.mean(K.stack(scores), axis=0)  # (N x 1)

        keep_nodes = max(int(self.rate*self.origin_nodes), 1)  # 保存节点的数目

        # 按照输入的最后一个维度排序，选取top keep_nodes的值
        keep_indices = scoring
        X_out = K.gather(X, keep_indices)
        A_out = A[keep_indices, keep_indices]
        return X_out, A_out

    # 计算输出shape
    def compute_output_shape(self, input_shape):

        return None


