# -*- coding: utf-8 -*-
# ***************************************************/ 
# @Time    : 2018/6/5 14:52
# @Author  : GengDaPeng
# @contact : bingshan222@hotmail.com
# @File    : DecisionTree.py
# @Desc    : 《机器学习实战》 决策树构造章节py3代码
# ***************************************************/


import numpy as np


def cerate_dataset():
    """ 创建数据集 """
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def calcShanonEnt(dataset):
    """ 计算香农熵 """
    num_entries = len(dataset)
    label_cunt = {}     # 创建统计标签（label）的字典
    for featvec in dataset:
        current_label = featvec[-1]
        if current_label not in label_cunt.keys():
            label_cunt[current_label] = 0   # 如果标签(label)不存在，则创建新标签并设值为0
        label_cunt[current_label] += 1  # 值加1
    shannonent = 0.0    # 信息熵
    for key in label_cunt:
        prob = float(label_cunt[key]) / num_entries     # 标签概率
        shannonent -= prob * np.log2(prob)    # 求以2为底的对数
    return shannonent


def split_dataset(dataset, axis, value):
    """ 按照给定特征划分数据集
    Parameters:
        dataset - 带划分的数据集
        axis - 划分数据集的特征
        value - 需要返回的特征值
    Returns:
        ret_dataset -按照给定特征划分后返回的特征数据集
    """
    ret_dataset = []     # 创建返回的数据集列表
    for featvec in dataset:     # 遍历数据集
        if featvec[axis] == value:   # 特征是否符合要求
            reduce_featvec = featvec[:axis]  # 去除axis特征
            reduce_featvec.extend(featvec[axis+1:])   # 提取划分后的特征
            ret_dataset.append(reduce_featvec)   # 添加到数据集列表
    return ret_dataset


def chooseBestFeatureToSplit(dataset):
    """ 选择最好的数据集划分方式
    parameter:
        dataset - 划分的数据集
    return:
        bestfrature - 最佳特征
    """
    num_features = len(dataset[0]) - 1
    base_entropy = calcShanonEnt(dataset)
    bestinfogain = 0.0
    bestfeature = -1
    for i in range(num_features):
        featlist = [example[i] for example in dataset]
        uniquevals = set(featlist)
        new_entropy = 0.0
        for value in uniquevals:
            subdataset = split_dataset(dataset, i, value)
            prob = len(subdataset) / float(len(dataset))
            new_entropy += prob * calcShanonEnt(subdataset)
        infogain = base_entropy - new_entropy
        if (infogain > bestinfogain):
            bestinfogain = infogain
            bestfeature = i
    return bestfeature





if __name__ == '__main__':
    mydat, labels = cerate_dataset()
    print(mydat)
    print('----------------------')
    shan = calcShanonEnt(mydat)
    print(shan)
    print('------------------------')
    ret_dataset1 = split_dataset(mydat, 0, 1)
    print(mydat[0][:0])
    print(ret_dataset1)
    ret_dataset2 = split_dataset(mydat, 0, 0)
    print(ret_dataset2)
    print('---------------------------')