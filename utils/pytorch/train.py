#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> train
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/25 9:35
@Desc   ：训练中的函数调用
=================================================='''

import pickle
import torch as th
import numpy as np
import torch.nn as nn
from dgl.data import Subset
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from utils.pytorch import *
from utils.pytorch.file import *
from utils.pytorch.metrics import *
from utils.pytorch.process import *
from utils.pytorch.fprint import *



# 封装训练过程
def do_train(model, device, train_loader, optimizer, criterion):
    '''
    模型训练
    :param model: 模型
    :param device: 设备位置
    :param train_loader: 训练集数据
    :param train_length: 训练集数据
    :param optimizer: 优化器
    :param criterion: 损失函数/评价指标
    :param epoch: 当前迭代轮次
    :return:
    '''
    model.train()
    loss = 0
    train_size = len(train_loader.dataset)  # 训练大小
    trained_size = 0  # 已训练大小
    correct, batch_correct = 0, 0
    progress_bar = "\r\t{}/{} [{}{}] - loss: {:.4f} - acc: {:.4f}"
    for batch_idx, (data, target) in enumerate(train_loader):

        target = target.to(device)
        optimizer.zero_grad()
        output, graphs = model(data)
        batch_loss = criterion(output, target)  # batch 损失
        loss += batch_loss  # 总体损失
        batch_loss.backward()  # 梯度下降
        optimizer.step()  # 参数更新

        # 训练进度条显示
        trained_size += len(target)
        trained_bar = round(trained_size/train_size * 20)
        arrow, dot = ">"*trained_bar, "."*(20-trained_bar)
        batch_loss /= len(target)
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        batch_correct = pred.eq(target.view_as(pred)).sum().item()
        correct += batch_correct
        print(progress_bar.format(trained_size, train_size, arrow, dot,
                                  batch_loss.item(), batch_correct/len(target)), end="")
    # 整体的loss显示
    loss /= train_size
    loss = loss.item()
    acc = correct / trained_size
    print(progress_bar.format(trained_size, train_size, arrow, dot, loss, acc), end="")
    return loss, acc


# 封装测试过程
def do_test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with th.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            target = target.to(device)
            output, graphs = model(data)
            test_loss += criterion(output, target)  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_loss = test_loss.item()
    test_acc = correct/len(test_loader.dataset)
    print(' - val_loss: {:.4f} - val_acc: {:.4f}'.format(test_loss, test_acc))
    return test_loss, test_acc


# 封装训练和重建过程
def do_train_and_rebuild(model, device, train_loader, optimizer, criterion):
    '''
    模型训练
    :param model: 模型
    :param device: 设备位置
    :param train_loader: 训练集数据
    :param train_length: 训练集数据
    :param optimizer: 优化器
    :param criterion: 损失函数/评价指标
    :param epoch: 当前迭代轮次
    :return:
    '''
    model.train()
    loss, classify_loss, rebuild_loss = 0, 0, 0
    train_size = len(train_loader.dataset)  # 训练大小
    trained_size = 0  # 已训练大小
    correct, batch_correct = 0, 0
    progress_bar = "\r\t{}/{} [{}{}] -loss:{:.4f} -loss_c:{:.4f} -loss_r:{:.4f} -acc:{:.4f}"

    for batch_idx, (data, target) in enumerate(train_loader):

        target = target.to(device)
        optimizer.zero_grad()
        output, graphs = model(data)
        batch_loss, batch_classify_loss, batch_construct_loss = criterion(output, target, graphs)  # batch 损失
        batch_loss.backward()  # 梯度下降
        optimizer.step()  # 参数更新

        # 训练进度条显示
        loss += batch_loss * len(target)  # 总体损失
        classify_loss += batch_classify_loss * len(target)  # 总体损失
        rebuild_loss += batch_construct_loss * len(target)  # 总体损失
        trained_size += len(target)
        trained_bar = round(trained_size/train_size * 20)
        arrow, dot = ">"*trained_bar, "."*(20-trained_bar)
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        batch_correct = pred.eq(target.view_as(pred)).sum().item()
        correct += batch_correct
        print(progress_bar.format(trained_size, train_size, arrow, dot,
                                  batch_loss.item(), batch_classify_loss, batch_construct_loss,
                                  batch_correct/len(target)), end="")
    # 整体的loss显示
    loss /= train_size
    classify_loss /= train_size
    rebuild_loss /= train_size
    loss = loss.item()
    acc = correct / trained_size
    print(progress_bar.format(trained_size, train_size, arrow, dot,
                              loss, classify_loss, rebuild_loss,
                              acc), end="")
    return classify_loss, acc


# 封装测试和重建过程
def do_test_and_rebuild(model, device, test_loader, criterion):
    model.eval()
    test_loss, classify_loss, rebuild_loss = 0, 0, 0
    correct = 0
    with th.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            target = target.to(device)
            output, graphs = model(data)
            # test_loss += criterion(output, target)  # 将一批的损失相加
            batch_loss, batch_classify_loss, batch_construct_loss = criterion(output, target, graphs)
            test_loss += batch_loss * len(target)  # 将一批的损失相加
            classify_loss += batch_classify_loss * len(target)  # 将一批的损失相加
            rebuild_loss += batch_construct_loss * len(target)  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    classify_loss /= len(test_loader.dataset)
    rebuild_loss /= len(test_loader.dataset)
    test_loss = test_loss.item()
    test_acc = correct/len(test_loader.dataset)
    print(' -val_loss:{:.4f} -val_loss_c:{:.4f} -val_loss_r:{:.4f} -val_acc:{:.4f}'.format(
        test_loss, classify_loss, rebuild_loss, test_acc))
    return classify_loss, test_acc


# K_Fold 训练
def k_fold_train(dataset, model_fn, n_fold,
                 save_path, model_name, device,
                 batch_size=1024, random_seed=42,
                 epochs=1000, es_patience=20):
    print("start train {}_fold with {}".format(n_fold, model_name))

    sfolder = StratifiedKFold(n_splits=n_fold, random_state=random_seed, shuffle=True)
    X_idx = range(len(dataset))
    y = dataset.target

    fold_results = list()
    fold_idx = 0
    for train_idx, val_idx in sfolder.split(X_idx, y):
        fold_idx += 1

        model, optimizer, criterion = model_fn()

        print("n_Fold:{}/{}".format(fold_idx, n_fold))
        # 数据划分
        train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)
        dataset_names = ["train", "valid"]
        graph_dataset = {"train": train_dataset, "valid": val_dataset}
        graph_dataloader = {x: DataLoader(graph_dataset[x],
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          collate_fn=collate) for x in dataset_names}

        # 训练过程
        train_model(graph_dataloader, random_seed=random_seed, model=model,
                    save_path=save_path, model_name=model_name, fold_idx=fold_idx,
                    optimizer=optimizer, criterion=criterion, device=device,
                    epochs=epochs, es_patience=es_patience)
        # 加载最优模型
        best_model = load_best_model(model, save_path=save_path,
                                     model_name=model_name, random_seed=random_seed,
                                     fold_idx=fold_idx)
        val_result = evaluate_model(best_model, val_dataset=val_dataset, device=device,
                       save_path=save_path, model_name=model_name, random_seed=random_seed,
                       fold_idx=fold_idx)
        fold_results.append(val_result)
    metrics_result = pd.DataFrame(fold_results)
    metrics_result['random_seed'] = random_seed
    metrics_result.to_csv("{}/{}/{}/metrics_result.csv".format(save_path, model_name, random_seed),
                          index=False)
    metrics_result.describe().to_csv("{}/{}/{}/metrics_result_describe.csv".format(save_path, model_name, random_seed),
                          index=False)
    return metrics_result


# 模型训练
def train_model(graph_dataloader, random_seed, model,
                save_path, model_name, fold_idx,
                optimizer, criterion, device,
                epochs=1000, es_patience=20):
    best_val_loss = np.inf
    history = defaultdict(list)
    model_save_dir = "{}/{}/{}/fold_{}".format(save_path, model_name, random_seed, fold_idx)
    model_weight_save_path = "{}/best_weight.pkl".format(model_save_dir)
    model_history_path = "{}/history.pkl".format(model_save_dir)
    checkd_directory(model_save_dir)  # 检查保存位置
    print("Train on {} samples. Valid on {} samples".format(len(graph_dataloader['train'].dataset),
                                                            len(graph_dataloader['valid'].dataset)))
    for epoch in range(1, epochs + 1):
        print("Epoch {}/{}".format(epoch, epochs))
        loss, acc = do_train(model, device, graph_dataloader["train"], optimizer, criterion)
        val_loss, val_acc = do_test(model, device, graph_dataloader["valid"], criterion)
        history['loss'].append(loss)
        history['acc'].append(acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            early_stopping = 0
            th.save(model.state_dict(), model_weight_save_path)
        else:
            early_stopping += 1
            if early_stopping > es_patience:  # 防止过拟合
                print("Early Stopping")
                break

    pickle.dump(history, open(model_history_path, 'wb'))  # 保存训练结果
    draw_train_process_cruve(model_history_path)


# 加载最优模型
def load_best_model(model, save_path, model_name, random_seed, fold_idx):
    model_save_dir = "{}/{}/{}/fold_{}".format(save_path, model_name, random_seed, fold_idx)
    model_weight_save_path = "{}/best_weight.pkl".format(model_save_dir)
    model.load_state_dict(th.load(model_weight_save_path))
    model.eval()
    return model


# 评估模型
def evaluate_model(model, val_dataset, device,
                   save_path, model_name, random_seed, fold_idx):
    y_pred = do_predict(model, device, val_dataset)
    y_val = np.array(val_dataset.dataset.target)[val_dataset.indices.tolist()]
    val_result = evaluate_binary_classification(y_val, y_pred)

    # 保存结果
    model_save_dir = "{}/{}/{}/fold_{}".format(save_path, model_name, random_seed, fold_idx)
    metrics_path = "{}/evaluation.txt".format(model_save_dir)
    print_binary_evaluation_dict(val_result)
    write_metrics_to_file(val_result, metrics_path)
    predict_path = "{}/prediction.csv".format(model_save_dir)
    write_result_to_file(y_val, y_pred, predict_path)
    val_result['TN'] = val_result['confusion_matrix'][0][0]
    val_result['FN'] = val_result['confusion_matrix'][0][1]
    val_result['FP'] = val_result['confusion_matrix'][1][0]
    val_result['TP'] = val_result['confusion_matrix'][1][1]
    val_result.pop('confusion_matrix')
    return val_result
