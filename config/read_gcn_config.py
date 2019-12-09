#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> read_gcn_config
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/12/9 16:56
@Desc   ：读取gcn_config的内容
=================================================='''

import os
import configparser

# 获取文件的当前路径（绝对路径）
cur_path = os.path.dirname(os.path.realpath(__file__))

# 获取config.ini的路径
config_path = os.path.join(cur_path, 'gcn_config.ini')

conf=configparser.ConfigParser()
conf.read(config_path, encoding="utf-8-sig")

# 数据集参数
dataset_dir = conf.get('dataset', 'dataset_dir')  # 数据集位置
dataset_name = conf.get('dataset', 'dataset_name')  # 数据集名字

# 训练参数
random_seed = conf.get('train', 'random_seed').strip().split(',')  # 随机种子设定
n_fold = int(conf.get('train', 'n_fold'))  # 折数
epoch = int(conf.get('train', 'epoch'))  # 迭代数
es_patience = int(conf.get('train', 'es_patience'))  # 提前停止数
batch_size = int(conf.get('train', 'batch_size'))  # 提前停止数
lr = float(conf.get('train', 'lr'))  # 学习率

# 模型参数
n_hidden = conf.get('model', 'lr').strip().split(',')  # 隐藏神经元数目
n_output = int(conf.get('model', 'n_output'))  # 输出类别数
pooling_type = conf.get('model', 'pooling_type')  # 池化方式
dropout = float(conf.get('model', 'dropout'))

# 保存参数
save_dir = conf.get('result', 'save_dir')

