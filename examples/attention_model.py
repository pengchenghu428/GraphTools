#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> attention_model
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/13 13:17
@Desc   ：创建attention 模型
=================================================='''

import os
from layers import *
from utils import *
from keras.layers import Input, Dense, Dropout, LSTM
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2

dataset_path = 'data/'
dataset_name = 'FRANKENSTEIN'
model_save_path = 'output/models/'
model_name = 'graph_att'

F_ = 8                        # gat 层数输出尺寸
n_attn_heads = 8              # 注意力机制数量
dropout_rate = 0.6            # 丢失率
l2_reg = 5e-4/2               # l2正则化因子
learning_rate = 5e-3          # Adam 学习率
epochs = 10000                # 迭代次数
batch_size = 256              # batch 尺寸
es_patience = 100             # 提前停止轮数


def data_prepared(Xs, As):
    print()


def create_attention_model(Xs, As):
    X_in = Input(shape=(Xs.shape[1], Xs.shape[-1]))  # (n_batch, nodes, f_dims)
    A_in = Input(shape=(As.shape[1], As.shape[-1]))  # (n_batch, nodes, nodes)

    graph_attention_1 = GraphAttention(units=32,
                                       attn_heads=n_attn_heads,
                                       attn_heads_reduction='concat',
                                       dropout_rate=dropout_rate,
                                       activation='elu',
                                       kernel_regularizer=l2(l2_reg),
                                       attn_kernel_regularizer=l2(l2_reg)
                                       )([X_in, A_in])
    sag_pooling_X_1, sag_pooling_A_1, sag_pooling_keep_values_1 = SAGraphPooling(rate=0.5,
                                                                                 attn_heads=8,
                                                                                 attn_heads_reduction='mean',
                                                                                 activation='softmax',
                                                                                 attn_initializer='glorot_uniform')([graph_attention_1, A_in])
    read_out_1 = GlobalMeanMaxPooling(type='mean|max')([sag_pooling_X_1, sag_pooling_keep_values_1])

    dense_1 = Dense(units=128, activation='relu')(read_out_1)
    dropout_1 = Dropout(rate=0.5)(dense_1)
    output = Dense(units=1, activation='sigmoid')(dropout_1)

    # Build examples
    model = Model(inputs=[X_in, A_in], outputs=output)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  weighted_metrics=['acc'])
    return model


def train(Xs, As, y, path, name):
    '''

    :param Xs: 特征矩阵集合
    :param As: 邻接矩阵集合
    :param y: 训练标签
    :param path: 模型保存位置
    :param name: 模型名称
    :return:
    '''
    print("Train attention examples")

    model = create_attention_model()

    weight_path = "{}/{}/{}_model_weights.best.h5".format(dir, name, name)
    checkpoint = ModelCheckpoint(filepath=weight_path, monitor='val_acc', verbose=0,
                                save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_acc', patience=es_patience)
    callback_list = [checkpoint, early_stopping]
    history = model.fit([Xs_train] + [As_train], y_train,
              epochs=epochs, batch_size=batch_size, shuffle=True,
              validation_split=0.1, callbacks=callback_list, verbose=1)
    model = load_model(path, name)  # 加载最优模型
    save_model(path, name, model, history=history)   # 保存模型和训练参数


if __name__ == "__main__":
    # execute only if run as a script
    os.chdir('../')
    Xs, As, y = load_data(dataset_path, dataset_name)  # 从磁盘加载数据
    Xs_train, As_train, y_train, Xs_test, As_test, y_test = split_data(Xs, As, y)  # 划分数据集
    train(Xs_train, As_train, y_train, model_save_path, model_name)
    model = load_model(model_save_path, model_name)  # 加载最优模型


