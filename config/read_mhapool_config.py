#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> read_mhapool_config
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/12/30 12:57
@Desc   ：读取multi-head pool配置文件
=================================================='''

import os
import configparser

# 获取文件的当前路径（绝对路径）
cur_path = os.path.dirname(os.path.realpath(__file__))

# 获取config.ini的路径
config_path = os.path.join(cur_path, 'mhapool_config.ini')

conf=configparser.ConfigParser()
conf.read(config_path, encoding="utf-8-sig")

# 数据集参数
dataset_dir = conf.get('dataset', 'dataset_dir')  # 数据集位置
dataset_name = conf.get('dataset', 'dataset_name')  # 数据集名字
node_attr_type = int(conf.get('dataset', 'node_attr_type'))  # 节点特征类型

# 训练参数
random_seeds = conf.get('train', 'random_seeds').strip().split(',')  # 随机种子设定
random_seeds = [int(random_seed) for random_seed in random_seeds]
gnn_type = conf.get('train', 'gnn_type')
n_fold = int(conf.get('train', 'n_fold'))  # 折数
epoch = int(conf.get('train', 'epoch'))  # 迭代数
es_patience = int(conf.get('train', 'es_patience'))  # 提前停止数
batch_size = int(conf.get('train', 'batch_size'))  # 提前停止数
lr = float(conf.get('train', 'lr'))  # 学习率

# 模型参数
n_hidden = conf.get('model', 'n_hidden').strip().split(',')  # 隐藏神经元数目
n_hidden = [int(unit) for unit in n_hidden]
n_head = int(conf.get('model', 'n_head'))
n_channel = int(conf.get('model', 'n_channel'))
n_output = int(conf.get('model', 'n_output'))  # 输出类别数
pooling_type = conf.get('model', 'pooling_type')  # 池化方式
dropout = float(conf.get('model', 'dropout'))

# 保存参数
save_dir = "{}/{}/".format(conf.get('result', 'save_dir'), dataset_name)