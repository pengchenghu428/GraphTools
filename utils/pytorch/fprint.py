#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> fprint
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/25 11:05
@Desc   ：个性化输出
=================================================='''

import numpy as np
import pandas as pd

from utils.pytorch import *
from utils.pytorch.file import *


def print_binary_evaluation_dict(res_dict):
    '''
    控制台格式化输出字典
    :param res_dict:
    :return:
    '''
    assert isinstance(res_dict, dict), "print_dict：传入的参数类型不是dict"
    for idx, key in enumerate(res_dict.keys()):
        if key == 'confusion_matrix':
            print("{}: {}".format(key, res_dict[key]))
            continue
        print("{}: {:.6f}".format(key, res_dict[key]))


def write_metrics_to_file(res_dict, filepath):
    '''
    将结果输出保存至本地文件中
    :param res_dict:
    :return:
    '''
    checkd_directory(filepath)
    with open(filepath, 'w') as file:
        for idx, key in enumerate(res_dict.keys()):
            if key == 'confusion_matrix':
                file.write("{}: {}\n".format(key, res_dict[key]))
                continue
            file.write("{}: {:.6f}\n".format(key, res_dict[key]))
        file.flush()  # 清缓冲区


def write_result_to_file(y, y_pred, filepath):
    '''
    将结果输出保存至本地文件中
    :param
    :return:
    '''
    result = pd.DataFrame()
    result['label'] = np.array(y).flatten()
    result['y_pred'] = np.array(y_pred).flatten()
    result.to_csv(filepath, index=False)