#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> spectral_spatial_gnn
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/18 15:57
@Desc   ：时域和频域双重卷积的图分类模型
=================================================='''
from __future__ import absolute_import

import os
from layers.keras import *
from utils.keras import *
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

dataset_path = 'data/'
dataset_name = 'FRANKENSTEIN'
model_save_path = 'output/models/'
model_name = 'spectral_spatial_gnn'  # 基于时域和频域双重图卷积模型

# learning_rate = 5e-3          # Adam 学习率
epochs = 10000                # 迭代次数
batch_size = 64              # batch 尺寸
es_patience = 5             # 提前停止轮数

def data_prepared(Xs, As):
    '''
    数据准备
    :param Xs:特征矩阵集合
    :param As:邻接矩阵集合
    :return:统一尺寸后的特征矩阵、邻接矩阵和归一化的邻接矩阵
    '''
    # 统一尺寸
    Xs, As = resized_graph(Xs, As)
    As = np.array([add_self_adj(A) for A in As])
    As_norm = np.array([normalize_adj(A) for A in As])
    return Xs, As, As_norm


def create_ssgnn_model(Xs, As, As_norm):
    '''
    创建基于时域和频域双重图卷积模型
    :param Xs: 特诊
    :param As:
    :param As_norm:
    :return:
    '''
    Xs_in = Input(shape=(Xs.shape[1], Xs.shape[-1]))  # (n_batch, nodes, f_dims)
    As_in = Input(shape=(As.shape[1], As.shape[-1]))  # (n_batch, nodes, nodes)
    As_norm_in = Input(shape=(As_norm.shape[1], As_norm.shape[-1]))  # (n_batch, nodes, nodes)

    # spectral 频域
    gcn_1 = GraphConvolution(units=256, support=1, activation='relu')([Xs_in, As_norm_in])
    spec_read_out_1 = GlobalMeanPooling()(gcn_1)
    gcn_2 = GraphConvolution(units=128, support=1, activation='relu')([gcn_1, As_norm_in])
    spec_read_out_2 = GlobalMeanPooling()(gcn_2)
    gcn_3 = GraphConvolution(units=64, support=1, activation='relu')([gcn_2, As_norm_in])
    spec_read_out_3 = GlobalMeanPooling()(gcn_3)

    gat_1 = GraphAttention(units=256, attn_heads=8, attn_heads_reduction='average',
                           dropout_rate=0.5)([Xs_in, As_in])
    spat_read_out_1 = GlobalMeanPooling()(gat_1)
    gat_2 = GraphAttention(units=128, attn_heads=8, attn_heads_reduction='average',
                           dropout_rate=0.5)([gat_1, As_in])
    spat_read_out_2 = GlobalMeanPooling()(gat_2)
    gat_3 = GraphAttention(units=64, attn_heads=8, attn_heads_reduction='average',
                           dropout_rate=0.5)([gat_2, As_in])
    spat_read_out_3 = GlobalMeanPooling()(gat_3)

    # meiged
    merged_layer = Concatenate(axis=-1)([spec_read_out_1, spec_read_out_2, spec_read_out_3,
                                         spat_read_out_1, spat_read_out_2, spat_read_out_3])  # [read_out1, read_out2]
    # dnn
    dense_1 = Dense(units=256, activation='relu')(merged_layer)  # dense
    dropout_1 = Dropout(rate=0.5)(dense_1)  # dropout
    dense_2 = Dense(units=128, activation='relu')(dropout_1)  # dense
    dropout_2 = Dropout(rate=0.5)(dense_2)  # dropout
    output = Dense(units=1, activation='sigmoid')(dropout_2)  # output

    # Build examples
    model = Model(inputs=[Xs_in, As_in, As_norm_in], outputs=output)
    optimizer = Adam()
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def train(Xs, As, As_norm, y, path, name):
    '''
    注意力模型训练
    :param Xs: 特征矩阵集合
    :param As: 邻接矩阵集合
    :param y: 训练标签
    :param path: 模型保存位置
    :param name: 模型名称
    :return:
    '''
    print("Train spectral_spatial_gnn examples")

    model = create_ssgnn_model(Xs, As, As_norm)

    weight_path = "{}/{}/{}_model_weights.best.h5".format(path, name, name)
    checkd_directory(weight_path)
    checkpoint = ModelCheckpoint(filepath=weight_path, monitor='val_acc', verbose=0,
                                save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_acc', patience=es_patience)
    callback_list = [checkpoint, early_stopping]
    history = model.fit([Xs, As, As_norm], y,
              epochs=epochs, batch_size=batch_size, shuffle=True,
              validation_split=0.2, callbacks=callback_list, verbose=1)
    model = load_model_weight(path, name, model)  # 加载最优模型
    save_model(path, name, model, history=history)   # 保存模型和训练参数
    plot_train_process(path, name)  # 绘制训练过程
    plot_model_architecture(path, name)  # 绘制模型结构图


def predict(model, Xs, As, As_norm):
    '''
    预测结果
    :param model: 调用模型
    :param Xs:特征矩阵集合
    :param As:邻接矩阵集合
    :return:预测值
    '''
    y_pred = model.predict([Xs, As, As_norm])
    return y_pred


def evaluate(y, y_pred, path=None, name=None):
    print("Positive Ratio: {:.4f}%".format(1.0 * np.sum(y)/len(y) * 100.0))
    result = evaluate_binary_classification(y, y_pred)
    print_binary_evaluation_dict(result)  # 控制台输出

    if not ((path is None) and (name is None)):
        metrics_path = "{}/{}/{}_metrics.txt".format(path, name, name)
        write_metrics_to_file(result, metrics_path)

        result_path = "{}/{}/{}_result.csv".format(path, name, name)
        write_result_to_file(y, y_pred, result_path)


if __name__ == "__main__":
    # execute only if run as a script
    os.chdir('../../')
    Xs, As, y = load_data(dataset_path, dataset_name)  # 从磁盘加载数据
    Xs, As, As_norm = data_prepared(Xs, As)
    Xs_train, As_train, As_norm_train, y_train, \
        Xs_test, As_test, As_norm_test, y_test = split_data(Xs, As, y, As_norm=As_norm, stratified=False)  # 划分数据集
    train(Xs_train, As_train, As_norm_train, y_train, model_save_path, model_name)
    model = load_model(model_save_path, model_name)  # 加载最优模型
    y_pred = predict(model,  Xs_test, As_test, As_norm_test)  # 预测结果
    evaluate(y_test, y_pred, model_save_path, model_name)  # 评估模型
