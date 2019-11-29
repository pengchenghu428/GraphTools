#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> fill
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/25 9:44
@Desc   ：文件检查、创建等操作
=================================================='''

import os


def checkd_directory(path):
    '''
    保证保存文件时，文件夹存在
    :param path: 文件或文件夹位置
    :return:
    '''
    if not os.path.exists(path):  # 判断该文件/文件夹是否存在
        dirs = path.split('/')
        if '.' in dirs[-1]:  # 如果最后是文件，则弹出最后一个元素
            dirs.pop()
        dir_path = '/'.join(dirs)  # 文件夹路径
        if not os.path.exists(dir_path):  # 文件夹不存在，则创建
            os.makedirs(dir_path)
