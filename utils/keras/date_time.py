#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraphTools -> test
@IDE    ：PyCharm
@Author ：pengchenghu
@Date   ：2019/11/11 20:06
@Desc   ：生成时间戳
=================================================='''

import time
import datetime


# function: 生成时间字符串
def generate_timestamp(suit_for_file=False):
    timestamp = time.time()
    timestruct = time.localtime(timestamp)
    if suit_for_file:
        dtime = time.strftime("%Y-%m-%d_%H-%M-%S", timestruct)
    else:
        dtime = time.strftime("%Y-%m-%d %H:%M:%S", timestruct)
    return dtime


# function: 把字符串转成datetime
def string_toDatetime(st):
    return datetime.datetime.strptime(st, "%Y-%m-%d %H:%M:%S")


# function: 把字符串转成分钟级别的datetime
def string_toDatetime_min_level(st):
    cur_time = string_toDatetime(st)
    return cur_time - datetime.timedelta(microseconds=cur_time.microsecond) - \
           datetime.timedelta(seconds=cur_time.second)


# function: 把时间戳转换成字符串
def timestamp_toString(sp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sp))[0:23]


# function: 把datetime转成字符串
def datetime_toString(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")[0:23]
