#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> layers -> graphpool -> sort_pooling
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/11 20:06
@Desc   ：Keras 实现SortPooling
=================================================='''

from keras.engine import Layer
import keras.backend as K
import tensorflow as tf


class SortPooling(Layer):
    def __init__(self,
                 keep_nodes=20,
                 **kwargs):
        super(SortPooling, self).__init__(**kwargs)

        self.keep_nodes = keep_nodes  # 保留节点数

    def build(self, input_shape):
        self.built = True  # 必须将 self.built 设置为True, 以保证该 Layer 已经成功 build

    def call(self, inputs):
        features = inputs  # X
        origin_nodes, features_dim = features.shape
        # 按照输入的最后一个维度排序，选取top keep_nodes的值
        reordered = K.gather(features, tf.nn.top_k(features[:, -1], k=self.keep_nodes).indices)
        if reordered.shape[1] < self.keep_nodes:  # 扩充0元素
            extend_rows = K.zeros((self.keep_nodes - origin_nodes, features_dim))
            reordered = K.concatenate([reordered, extend_rows], axis=0)
        return reordered

    def compute_output_shape(self, input_shape):
        output_shape = (self.keep_nodes, input_shape[-1])
        return output_shape
