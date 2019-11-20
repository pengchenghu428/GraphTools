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
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dataset_path = 'data/'
dataset_name = 'FRANKENSTEIN'
model_save_path = 'output/models/'
model_name = 'graph_att'

n_attn_heads = 8              # 注意力机制数量
dropout_rate = 0.6            # 丢失率
l2_reg = 5e-4/2               # l2正则化因子
learning_rate = 5e-3          # Adam 学习率
epochs = 10000                # 迭代次数
batch_size = 32              # batch 尺寸
es_patience = 10             # 提前停止轮数


def data_prepared(Xs, As):
    '''
    数据准备
    :param Xs:
    :param As:
    :return:
    '''
    # 统一尺寸
    Xs, As = resized_graph(Xs, As)
    As = np.array([add_self_adj(A) for A in As])
    return Xs, As


def create_attention_model(Xs, As):
    X_in = Input(shape=(Xs.shape[1], Xs.shape[-1]))  # (n_batch, nodes, f_dims)
    A_in = Input(shape=(As.shape[1], As.shape[-1]))  # (n_batch, nodes, nodes)

    graph_attention_1 = GraphAttention(units=256,
                                       attn_heads=n_attn_heads,
                                       attn_heads_reduction='average',
                                       dropout_rate=dropout_rate,
                                       activation='elu',
                                       kernel_regularizer=l2(l2_reg),
                                       attn_kernel_regularizer=l2(l2_reg)
                                       )([X_in, A_in])  # gat
    read_out_1 = GlobalMeanPooling()(graph_attention_1)  # read_out_1
    graph_attention_2 = GraphAttention(units=128,
                                      attn_heads=n_attn_heads,
                                      attn_heads_reduction='average',
                                      dropout_rate=dropout_rate,
                                      activation='elu',
                                      kernel_regularizer=l2(l2_reg),
                                      attn_kernel_regularizer=l2(l2_reg)
                                      )([graph_attention_1, A_in])  # gat
    read_out_2 = GlobalMeanPooling()(graph_attention_2)  # read_out_2
    merged_layer = Concatenate(axis=-1)([read_out_1, read_out_2])  # [read_out1, read_out2, read_out_3]
    dense_1 = Dense(units=256, activation='relu')(merged_layer)  # dense
    dropout_1 = Dropout(rate=0.5)(dense_1)  # dropout
    output = Dense(units=1, activation='sigmoid')(dropout_1)  # output

    # Build examples
    model = Model(inputs=[X_in, A_in], outputs=output)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def train(Xs, As, y, path, name):
    '''
    注意力模型训练
    :param Xs: 特征矩阵集合
    :param As: 邻接矩阵集合
    :param y: 训练标签
    :param path: 模型保存位置
    :param name: 模型名称
    :return:
    '''
    print("Train attention examples")

    model = create_attention_model(Xs, As)

    weight_path = "{}/{}/{}_model_weights.best.h5".format(path, name, name)
    checkd_directory(weight_path)
    checkpoint = ModelCheckpoint(filepath=weight_path, monitor='val_acc', verbose=0,
                                save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_acc', patience=es_patience)
    callback_list = [checkpoint, early_stopping]
    history = model.fit([Xs, As], y,
              epochs=epochs, batch_size=batch_size, shuffle=True,
              validation_split=0.2, callbacks=callback_list, verbose=1)
    model = load_model_weight(path, name, model)  # 加载最优模型
    save_model(path, name, model, history=history)   # 保存模型和训练参数


def predict(model, Xs, As):
    '''
    预测结果
    :param model: 调用模型
    :param Xs:特征矩阵集合
    :param As:邻接矩阵集合
    :return:预测值
    '''
    y_pred = model.predict([Xs, As])
    return y_pred


def evaluate(y, y_pred, path=None, name=None):
    result = evaluate_binary_classification(y, y_pred)  # 评价指标
    print_binary_evaluation_dict(result)  # 控制台输出

    if not ((path is None) and (name is None)):
        metrics_path = "{}/{}/{}_metrics.txt".format(path, name, name)  # 保存指标
        write_metrics_to_file(result, metrics_path)

        result_path = "{}/{}/{}_result.txt".format(path, name, name)  # 保存结果
        write_result_to_file(y, y_pred, result_path)


if __name__ == "__main__":
    # execute only if run as a script
    os.chdir('../')
    Xs, As, y = load_data(dataset_path, dataset_name)  # 从磁盘加载数据
    Xs, As = data_prepared(Xs, As)
    Xs_train, As_train, y_train, Xs_test, As_test, y_test = split_data(Xs, As, y)  # 划分数据集
    train(Xs_train, As_train, y_train, model_save_path, model_name)
    model = load_model(model_save_path, model_name)  # 加载最优模型
    y_pred = predict(model,  Xs_test, As_test)  # 预测结果
    evaluate(y_test, y_pred, model_save_path, model_name)  # 评估模型
