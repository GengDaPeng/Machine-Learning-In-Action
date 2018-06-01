# -*- coding: utf-8 -*-
# @Time    : 2018/5/20 21:45
# @Author  : GengDaPeng
# @File    : KNN.py
# @Software: PyCharm
# **************************************************************
#              Hello     Friends ！！
# If this runs wrong, don't ask me, I don't know why;╮(╯▽╰)╭
# If this runs right, thank god, and I don't know why.(～￣▽￣)～ 
# Maybe the answer, my friend, is blowing in the wind.(*/ω＼*)
# **************************************************************
import numpy as np
import operator
from os import listdir


def create_dataset():
    """ 创建数据集 """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classfiy0(inx, dataset, labels, k):
    """ 创建一个分类器 """
    datasize = dataset.shape[0]    # 获取数据的大小
    diffmat = np.tile(inx, (datasize, 1)) - dataset  # 计算点与点 x和y坐标之间的差值
    """
    在这里，np.tile 的作用是将输入的向量转换成跟数据集形状的向量，用来
    计算坐标差值
    >>> a = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    >>> datasize = a.shape[0]  # 可知 dataset = 4
    >>> np.tile([1,2], (datasize, 1))
    out: array([[1, 2],
              [1, 2],
              [1, 2],
              [1, 2]])
    """
    sq_diffmat = diffmat ** 2  # 差值的平方
    sq_distance = sq_diffmat.sum(axis=1)  # 计算平方和即距离的平方
    distance = sq_distance ** 0.5  # 开方计算出距离
    sort_distance_index = distance.argsort() # 返回距离的索引值，且索引值是按照距离的大小排序的。
    """
    >>> b = np.array([1, 3, 2, 4, 5])
    >>> c = b.argsort()
    >>> c
    out: array([0, 2, 1, 3, 4], dtype=int64)
    """
    class_count = {}
    for i in range(k):
        vote_label =labels[sort_distance_index[i]]  # 选出k个最小的距离的标签
        class_count[vote_label] = class_count.get(vote_label, 0) + 1




























