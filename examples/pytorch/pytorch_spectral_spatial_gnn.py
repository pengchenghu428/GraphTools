#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> pytorch_spectral_spatial_gnn
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/25 16:52
@Desc   ：
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
from dgl.nn.pytorch import GraphConv, GATConv, AvgPooling, MaxPooling, SortPooling

from utils.pytorch import *


os.chdir('../../')
# 数据集
DATA_PATH = "data/"
DATA_NAME = "FRANKENSTEIN"
GRAPH_IDX_PATH = "{}/{}/{}.graph_idx".format(DATA_PATH, DATA_NAME, DATA_NAME)
GRAPH_LABELS_PATH = "{}/{}/{}.graph_labels".format(DATA_PATH, DATA_NAME, DATA_NAME)

# 模型
MODEL_SAVE_PATH = "output/models/"
MODEL_NAME = "torch_spectral_spatial_gnn_kl_model"
BATCH_SIZE = 1024
EPOCHS = 2000
ES_PATIENCE = 50
IN_FEATS = 780
SPEC_N_HIDDEN = [256, 256, 256]
SPAT_N_HIDDEN = [256, 256, 256]
N_HEADS = [8, 8, 8]
ACTIVATION = F.leaky_relu
DROPOUT = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU


# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)


# 构建网络
class SSGNModel(nn.Module):
    def __init__(self,
                 in_feats,
                 spec_hidden,
                 spat_hidden,
                 n_heads,
                 activation,
                 dropout):
        super(SSGNModel, self).__init__()

        self.spat_hidden = spat_hidden
        self.layers = nn.ModuleList()
        self.dropout = dropout

        # input layer
        # input layer
        self.gcn_1 = GraphConv(in_feats, spec_hidden[0], activation=activation)
        self.gcn_2 = GraphConv(spec_hidden[0], spec_hidden[1], activation=activation)
        self.gcn_3 = GraphConv(spec_hidden[1], spec_hidden[2], activation=activation)
        self.gat_1 = GATConv(in_feats, spat_hidden[0], num_heads=n_heads[0],
                               feat_drop=dropout, attn_drop=dropout, activation=activation)
        self.gat_2 = GATConv(spat_hidden[0], spat_hidden[1], num_heads=n_heads[1],
                               feat_drop=dropout, attn_drop=dropout, activation=activation)
        self.gat_3 = GATConv(spat_hidden[1], spat_hidden[2], num_heads=n_heads[2],
                               feat_drop=dropout, attn_drop=dropout, activation=activation)
        # output layer
        self.fc_1 = nn.Linear(1024, 128)
        self.fc_2 = nn.Linear(128, 2)

    def forward(self, graph):
        feature = graph.ndata['h'].float()
        # GNN
        # spec
        gcn1 = self.gcn_1(graph, feature)
        spec_read_out1 = torch.cat([AvgPooling()(graph, gcn1), MaxPooling()(graph, gcn1)], dim=1)
        gcn2 = self.gcn_2(graph, gcn1)
        spec_read_out2 = torch.cat([AvgPooling()(graph, gcn2), MaxPooling()(graph, gcn2)], dim=1)
        gcn3 = self.gcn_3(graph, gcn2)
        spec_read_out3 = torch.cat([AvgPooling()(graph, gcn3), MaxPooling()(graph, gcn3)], dim=1)
        # spat
        gat1 = self.gat_1(graph, feature)
        gat1 = torch.mean(gat1, dim=1).view(-1, self.spat_hidden[0])
        spat_read_out1 = torch.cat([AvgPooling()(graph, gat1), MaxPooling()(graph, gat1)], dim=1)
        gat2 = self.gat_2(graph, gat1)
        gat2 = torch.mean(gat2, dim=1).view(-1, self.spat_hidden[1])
        spat_read_out2 = torch.cat([AvgPooling()(graph, gat2), MaxPooling()(graph, gat2)], dim=1)
        gat3 = self.gat_3(graph, gat2)
        gat3 = torch.mean(gat3, dim=1).view(-1, self.spat_hidden[2])
        spat_read_out3 = torch.cat([AvgPooling()(graph, gat2), MaxPooling()(graph, gat3)], dim=1)
        # Merged
        # spec
        spec_merged = spec_read_out1 + spec_read_out2 + spec_read_out3
        spec_merged = spec_merged.view(-1, self.get_flatten_size(spec_merged))
        # spat
        spat_merged = spec_read_out1 + spat_read_out2 + spat_read_out3
        spat_merged = spat_merged.view(-1, self.get_flatten_size(spat_merged))
        merged = torch.cat([spec_merged, spat_merged], dim=1)
        # dnn
        dropout1 = nn.Dropout(self.dropout)(merged)
        fc1 = F.relu(self.fc_1(dropout1))
        dropout2 = nn.Dropout(self.dropout)(fc1)
        fc2 = self.fc_2(dropout2)
        out = torch.sigmoid(fc2)
        return out

    def get_flatten_size(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# 网络初始化
ssgnn_model = SSGNModel(in_feats=IN_FEATS,
                        spec_hidden=SPEC_N_HIDDEN,
                        spat_hidden=SPAT_N_HIDDEN,
                        n_heads=N_HEADS,
                        activation=ACTIVATION,
                        dropout=DROPOUT).to(DEVICE)
optimizer = optim.Adam(ssgnn_model.parameters())
criterion = nn.CrossEntropyLoss(reduction='sum')

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
    loss, acc = do_train(ssgnn_model, DEVICE, graph_dataloader["train"], optimizer, criterion)
    val_loss, val_acc = do_test(ssgnn_model, DEVICE, graph_dataloader["valid"], criterion)
    history['loss'].append(loss)
    history['acc'].append(acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        early_stopping = 0
        torch.save(ssgnn_model.state_dict(), model_weight_save_path)
    else:
        early_stopping += 1
        if early_stopping > ES_PATIENCE:  # 防止过拟合
            print("Early Stopping")
            break
pickle.dump(history, open(model_history_path, 'wb'))  # 保存训练结果
plot_train_process(MODEL_SAVE_PATH, MODEL_NAME)

# 加载模型并评估模型
ssgnn_best_model = SSGNModel(in_feats=IN_FEATS,
                      spec_hidden=SPEC_N_HIDDEN,
                      spat_hidden=SPAT_N_HIDDEN,
                      n_heads=N_HEADS,
                      activation=ACTIVATION,
                      dropout=DROPOUT).to(DEVICE)
optimizer = optim.Adam(ssgnn_best_model.parameters())
ssgnn_best_model.load_state_dict(torch.load(model_weight_save_path))
ssgnn_best_model.eval()

# 评估模型
y_pred = do_predict(ssgnn_best_model, DEVICE, val_dataset)
val_result = evaluate_binary_classification(y_val, y_pred)
# 保存结果
metrics_path = "{}/{}_evaluation.txt".format(model_save_dir, MODEL_NAME)
print_binary_evaluation_dict(val_result)
write_metrics_to_file(val_result, metrics_path)
predict_path = "{}/{}_prediction.csv".format(model_save_dir, MODEL_NAME)
write_result_to_file(y_val, y_pred, predict_path)
