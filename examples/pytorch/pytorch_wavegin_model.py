#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> pytorch_wavegin_model
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/12/2 10:15
@Desc   ：Wave Gin 测试用例
=================================================='''

import os
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import pickle

from torch.utils.data import DataLoader

from layers.pytorch import *
from utils.pytorch import *

os.chdir('../../')
# 数据集
DATA_PATH = "data/"
DATA_NAME = "FRANKENSTEIN"
GRAPH_IDX_PATH = "{}/{}/{}.graph_idx".format(DATA_PATH, DATA_NAME, DATA_NAME)
GRAPH_LABELS_PATH = "{}/{}/{}.graph_labels".format(DATA_PATH, DATA_NAME, DATA_NAME)

# 模型
MODEL_SAVE_PATH = "output/models/"
MODEL_NAME = "torch_wave_gin_with_construct_model"
BATCH_SIZE = 1024
EPOCHS = 2000
ES_PATIENCE = 20
IN_FEATS = 780
# N_HIDDEN = [512, 256, 256, 128, 128, 64, 64, 32]
N_HIDDEN = [256, 256, 256]
ACTIVATION = F.leaky_relu
ACTIVE_NODES = 16
CHANNEL = 64
DROPOUT = 0.5
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU


# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)


# 构建网络
class WaveGINModel(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 activation,
                 active_nodes,  # softattention节点数目
                 channel,  # 一维卷积后的通道数
                 dropout):
        super(WaveGINModel, self).__init__()
        self.active_nodes = active_nodes
        self.conv_size = np.sum(n_hidden)
        self.activation = activation
        self.dropout = dropout

        # WaveGin
        self.wave_gin_layers = nn.ModuleList()
        self.wave_gin_layers.append(WaveGIN(in_feats, n_hidden[0],
                                            aggregator_type='sum', init_eps=0,
                                            learn_eps=True, residual=False))
        for idx in range(len(n_hidden)-1):
            self.wave_gin_layers.append(WaveGIN(n_hidden[idx], n_hidden[idx+1],
                                                aggregator_type='sum', init_eps=0,
                                                learn_eps=True, residual=False))

        # read_out
        self.soft_att_pooling = nn.ModuleList()
        for i in range(len(n_hidden)):
            self.soft_att_pooling.append(SoftAttentionPooling(active_nodes, in_feats=n_hidden[i]))
        # self.soft_att_pooling.append(SoftAttentionPooling(active_nodes, in_feats=n_hidden[-1]))

        # conv
        self.conv_1 = nn.Conv1d(1, channel, self.conv_size, stride=self.conv_size)
        # self.conv_1 = nn.Conv1d(1, channel, n_hidden[-1], stride=n_hidden[-1])
        self.fc_1 = nn.Linear(channel * active_nodes, 128)
        self.fc_2 = nn.Linear(128, 2)

    def forward(self, graph):
        # 自定义传播层
        feat = graph.ndata['h'].float()
        wave_gins = list()
        graphs = list()
        for layer in self.wave_gin_layers:
            feat = layer(graph, feat)
            wave_gins.append(feat)
            graph.ndata['h'] = feat
            # graphs.append(graph)
        graphs.append(graph)

        # readout
        readouts = [self.soft_att_pooling[idx](graph, wave_gins[idx])
                    for idx in range(len(self.soft_att_pooling))]
        # readouts = [self.soft_att_pooling[idx](graph, wave_gins[-1])
        #             for idx in range(len(self.soft_att_pooling))]

        # meiged
        merged = torch.cat(readouts, dim=2)
        merged = merged.view(merged.size()[0], 1, -1)

        # conv
        conv1 = self.conv_1(merged)
        conv1 = self.activation(conv1)

        # mlp
        fc1 = conv1.view(conv1.size()[0], -1)  # 展成1维向量
        dropout_1 = nn.Dropout(self.dropout)(fc1)
        fc1 = self.activation(self.fc_1(dropout_1))
        dropout_2 = nn.Dropout(self.dropout)(fc1)
        fc2 = self.fc_2(dropout_2)
        out = torch.sigmoid(fc2)

        return out, graphs

    def get_flatten_size(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# 网络初始化
wave_gin_model = WaveGINModel(in_feats=IN_FEATS,
                              n_hidden=N_HIDDEN,
                              activation=ACTIVATION,
                              active_nodes=ACTIVE_NODES,
                              channel=CHANNEL,
                              dropout=DROPOUT).to(DEVICE)
optimizer = optim.Adam(wave_gin_model.parameters(), lr=LEARNING_RATE)
# criterion = nn.CrossEntropyLoss(reduction='sum')
criterion = CustomizedLoss(alpha=0.25)  # 自定义误差

# 训练集与测试集划分
all_idx, all_label = get_dataset_info(DATA_PATH, DATA_NAME)
train_graphs_idx, val_graphs_idx, y_train, y_val = split_datasets(all_idx, all_label, ratio=0.2)
y_train, y_val = np.where(y_train == -1, 1, 0).tolist(), np.where(y_val == -1, 1, 0).tolist()

# 训练集与测试集加载
dataset_names = ["train", "valid"]
train_dataset = GraphDataset(DATA_PATH, DATA_NAME, train_graphs_idx, "Train")
val_dataset = GraphDataset(DATA_PATH, DATA_NAME, val_graphs_idx, "Valid")
graph_dataset = {"train": train_dataset, "valid": val_dataset}
graph_dataloader = {x: DataLoader(graph_dataset[x],
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=0,
                                  collate_fn=collate) for x in dataset_names}

# 网络训练
early_stopping = 0
best_val_loss = np.inf
history = defaultdict(list)
model_save_dir = "{}/{}".format(MODEL_SAVE_PATH, MODEL_NAME)
model_weight_save_path = "{}/{}_best_weight.pkl".format(model_save_dir, MODEL_NAME)
model_history_path = "{}/{}_history.pkl".format(model_save_dir, MODEL_NAME)
checkd_directory(model_save_dir)  # 检查保存位置
print("Train on {} samples. Valid on {} samples".format(len(train_graphs_idx), len(val_graphs_idx)))
for epoch in range(1, EPOCHS + 1):
    print("Epoch {}/{}".format(epoch, EPOCHS))
    loss, acc = do_train_and_rebuild(wave_gin_model, DEVICE, graph_dataloader["train"], optimizer, criterion)
    val_loss, val_acc = do_test_and_rebuild(wave_gin_model, DEVICE, graph_dataloader["valid"], criterion)
    # loss, acc = do_train(wave_gin_model, DEVICE, graph_dataloader["train"], optimizer, criterion)
    # val_loss, val_acc = do_test(wave_gin_model, DEVICE, graph_dataloader["valid"], criterion)
    history['loss'].append(loss)
    history['acc'].append(acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        early_stopping = 0
        torch.save(wave_gin_model.state_dict(), model_weight_save_path)
    else:
        early_stopping += 1
        if early_stopping > ES_PATIENCE:  # 防止过拟合
            print("Early Stopping")
            break
pickle.dump(history, open(model_history_path, 'wb'))  # 保存训练结果
plot_train_process(MODEL_SAVE_PATH, MODEL_NAME)

# 加载模型并评估模型
wave_gin_best_model = WaveGINModel(in_feats=IN_FEATS,
                                   n_hidden=N_HIDDEN,
                                   activation=ACTIVATION,
                                   active_nodes=ACTIVE_NODES,
                                   channel=CHANNEL,
                                   dropout=DROPOUT).to(DEVICE)
optimizer = optim.Adam(wave_gin_best_model.parameters())
wave_gin_best_model.load_state_dict(torch.load(model_weight_save_path))
wave_gin_best_model.eval()

# 评估模型
y_pred = do_predict(wave_gin_best_model, DEVICE, val_dataset)
val_result = evaluate_binary_classification(y_val, y_pred)
# 保存结果
metrics_path = "{}/{}_evaluation.txt".format(model_save_dir, MODEL_NAME)
print_binary_evaluation_dict(val_result)
write_metrics_to_file(val_result, metrics_path)
predict_path = "{}/{}_prediction.csv".format(model_save_dir, MODEL_NAME)
write_result_to_file(y_val, y_pred, predict_path)
