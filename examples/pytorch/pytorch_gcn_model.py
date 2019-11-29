#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> pytorch_gat_model
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/25 14:01
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
from dgl.nn.pytorch import GraphConv, SumPooling, AvgPooling, MaxPooling
from sklearn.model_selection import StratifiedShuffleSplit

from utils.pytorch import *
#
os.chdir('../../')
# 数据集
DATA_PATH = "data/"
DATA_NAME = "FRANKENSTEIN"
GRAPH_IDX_PATH = "{}/{}/{}.graph_idx".format(DATA_PATH, DATA_NAME, DATA_NAME)
GRAPH_LABELS_PATH = "{}/{}/{}.graph_labels".format(DATA_PATH, DATA_NAME, DATA_NAME)

# 模型
MODEL_SAVE_PATH = "output/models/"
MODEL_NAME = "torch_gcn_kl_model"
BATCH_SIZE = 1024
EPOCHS = 1000
ES_PATIENCE = 10
IN_FEATS = 780
N_HIDDEN = [256, 256, 256]
ACTIVATION = F.relu
DROPOUT = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU

model_param = {'in_feats': IN_FEATS,
               'n_hidden': N_HIDDEN,
               'activation': ACTIVATION,
               'dropout': DROPOUT}

# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)


# 构建网络
class GCNModel(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 activation,
                 dropout):
        super(GCNModel, self).__init__()

        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.fc1_input_dim = np.sum(n_hidden)

        # input layer
        self.gcn_1 = GraphConv(in_feats, n_hidden[0], activation=activation)
        self.gcn_2 = GraphConv(n_hidden[0], n_hidden[1], activation=activation)
        self.gcn_3 = GraphConv(n_hidden[1], n_hidden[2], activation=activation)
        # output layer
        self.fc_1 = nn.Linear(768, 128)
        self.fc_2 = nn.Linear(128, 2)

    def forward(self, graph):
        feature = graph.ndata['h'].float()
        gcn1 = self.gcn_1(graph, feature)
        read_out1 = AvgPooling()(graph, gcn1)
        # read_out1 = torch.cat([AvgPooling()(graph, gcn1), MaxPooling()(graph, gcn1)], dim=1)
        gcn2 = self.gcn_2(graph, gcn1)
        read_out2 = AvgPooling()(graph, gcn2)
        # read_out2 = torch.cat([AvgPooling()(graph, gcn2), MaxPooling()(graph, gcn2)], dim=1)
        gcn3 = self.gcn_3(graph, gcn2)
        read_out3 = AvgPooling()(graph, gcn3)
        # read_out3 = torch.cat([AvgPooling()(graph, gcn2), MaxPooling()(graph, gcn3)], dim=1)
        merged = torch.cat([read_out1, read_out2, read_out3], dim=1)
        # merged = read_out1 + read_out2 + read_out3
        merged = merged.view(-1, self.get_flatten_size(merged))
        dropout1 = nn.Dropout(self.dropout)(merged)
        fc1 = F.relu(self.fc_1(dropout1))
        dropout2 = nn.Dropout(self.dropout)(fc1)
        fc2 = self.fc_2(dropout2)
        out = torch.sigmoid(fc2)
        return out

    def predict(self, graph):
        pred = F.softmax(self.forward(graph))
        ans = torch.max(pred, dim=1, keepdim=True)[:, 1]
        return ans

    def predict_proba(self, graph):
        return self.forward(graph)[:, 1]

    def get_flatten_size(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# 网络初始化
gcc_model = GCNModel(in_feats=IN_FEATS,
                 n_hidden=N_HIDDEN,
                 activation=ACTIVATION,
                 dropout=DROPOUT).to(DEVICE)
optimizer = optim.Adam(gcc_model.parameters())
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
    loss, acc = do_train(gcc_model, DEVICE, graph_dataloader["train"], optimizer, criterion)
    val_loss, val_acc = do_test(gcc_model, DEVICE, graph_dataloader["valid"], criterion)
    history['loss'].append(loss)
    history['acc'].append(acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        early_stopping = 0
        torch.save(gcc_model.state_dict(), model_weight_save_path)
    else:
        early_stopping += 1
        if early_stopping > ES_PATIENCE:  # 防止过拟合
            print("Early Stopping")
            break
pickle.dump(history, open(model_history_path, 'wb'))  # 保存训练结果
plot_train_process(MODEL_SAVE_PATH, MODEL_NAME)

# 加载模型并评估模型
gcc_best_model = GCNModel(in_feats=IN_FEATS,
                 n_hidden=N_HIDDEN,
                 activation=ACTIVATION,
                 dropout=DROPOUT).to(DEVICE)
gcc_best_model.load_state_dict(torch.load(model_weight_save_path))
gcc_best_model.eval()

# 评估模型
y_pred = do_predict(gcc_best_model, DEVICE, val_dataset)
val_result = evaluate_binary_classification(y_val, y_pred)
# 保存结果
metrics_path = "{}/{}_evaluation.txt".format(model_save_dir, MODEL_NAME)
print_binary_evaluation_dict(val_result)
write_metrics_to_file(val_result, metrics_path)
predict_path = "{}/{}_prediction.csv".format(model_save_dir, MODEL_NAME)
write_result_to_file(y_val, y_pred, predict_path)