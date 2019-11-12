#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> layers -> graphnn -> gated_graph_nn
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/11 20:06
@Desc   ：Keras 实现门控图神经网络 2016
=================================================='''

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout

class GatedGraphNN(Layer):

    def __init__(self,
                 **kwargs):
        super(GatedGraphNN, self).__init__(**kwargs)

    def build(self, input_shape):

        self.built = True  # 必须将 self.built 设置为True, 以保证该 Layer 已经成功 build

    def call(self, inputs):
        output = None
        return output

    def compute_output_shape(self, input_shape):
        output_shape = None
        return output_shape
