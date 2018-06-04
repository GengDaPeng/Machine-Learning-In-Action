# -*- coding:utf-8 -*-
# ***************************************************/
# Filename:KNN_py3.py
# CreateTime:Tuesday, 29th May 2018 9:43:28 am
# Author:GengDaPeng (bingshan222@hotamil.com)
# Last Modified: Friday, 1st June 2018 11:28:55 am
# ***************************************************/

import operator
from os import listdir
from typing import TextIO

import numpy as np


def create_dataset():
    """ 创建数据集 """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classfiy0(inx: object, dataset: object, labels: object, k: object) -> object:
    """ 创建一个分类器
    :type inx: object
    :type dataset: object
    :type labels: object
    :type k: int
    """
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
    sort_distance_index = distance.argsort()   # 返回距离的索引值，且索引值是按照距离的大小排序的。
    """
    >>> b = np.array([1, 3, 2, 4, 5])
    >>> c = b.argsort()
    >>> c
    out: array([0, 2, 1, 3, 4], dtype=int64)
    """
    class_count = {}
    for i in range(k):
        vote_label = labels[sort_distance_index[i]]  # 选出k个最小的距离的标签
        class_count[vote_label] = class_count.get(vote_label, 0) + 1   # 计算选中的标签中各自的数量
        """
        这里dict.get(key,0)的意思是，获得dict中key的值,如果没有key，则将此key添加进dict中，值为0
        >>> dic ={'a':1, 'b':2, 'c':3}
        >>> dic['a'] = dic.get('a', 0)+1
            dic['d'] = dic.get('d', 0)+1
        >>> dic
        Out: {'a': 2, 'b': 2, 'c': 3, 'd': 1}
        可见，有a，则a的值加1,没有d，则增加d另值为0再加1
        """
    sort_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    """
    这里class_ count.items() 返回的是一个列表.sorted中的key参数指的是按照前面对象中的哪个
    key作为排序的值，而operator.itemgetter(1)指的是前面返回的列表每一个元素中的第2个元素
    >>> dic ={'a':1, 'b':2, 'c':3}
        dic.items()
    Out: dict_items([('a', 1), ('b', 2), ('c', 3)])
    >>> sorted(dic.items(),key=operator.itemgetter(1),reverse=True)
    Out: [('c', 3), ('b', 2), ('a', 1)]
    """
    return sort_class_count[0][0]


def file2matrix(filename):
    """ 将数据文件转化成数组 """
    file = open(filename)
    number_line = len(file.readlines())
    mat = np.zeros((number_line, 3))   # 创建一个array
    class_lable = []   # 创建类别列表
    file.seek(0)  # 重置指针
    index = 0
    for line in file.readlines():
        line = line.strip()    # 去除空格
        line_list = line.split('\t')
        mat[index, :] = line_list[0:3]
        class_lable.append(line_list[-1])
        index += 1
    return mat, class_lable

def auto_norm(dataset):
    """
    归一化数据
    返回归一化后的数组，每列最值的差值和每列最小值
    """
    minvals = dataset.min(0)  # 每列最小值
    maxvals = dataset.max(0)   # 每列最大值
    """
    array.min()  # 所有中的最小数
    array.min(0) axis = 0  # 每列中的最小数
    array.min(1) axis = 1  # 每行中的最小数
    """
    ranges = maxvals- minvals
<<<<<<< HEAD
    norm_data = np.zeros(dataset.shape)
=======
    norm_data = np.zeros((dataset.shape))
>>>>>>> 262550718b962c145b25669b154d036b1c076143
    m = dataset.shape[0]    # 数组的行数
    norm_data = dataset - np.tile(minvals, (m,1))
    norm_data = norm_data / np.tile(maxvals, (m,1))
    return norm_data, ranges, minvals


def dating_class_test():
    """ 针对约会网站数据的测试 """
<<<<<<< HEAD
    hotatio  = 0.10   #
    dating_mat, dating_lables = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, minvals = auto_norm(dating_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * hotatio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classfier_result = classfiy0(norm_mat[i], norm_mat[num_test_vecs:m], dating_lables[num_test_vecs:m], 3)
        print('the classifier came back with %d, the real answer is: %d'%(classfier_result, dating_lables[i]))
        if (classfier_result != dating_lables[i]):
            error_count += 1.0
    print('the total error rate is: %f'%(error_count / float(num_test_vecs)))
    print(error_count)

    
    # def img2vector(filename):
    #     return_vetor = np.zeros((1, 1024))
    #     fr = open(filename)
    #     for i in range(32):
    #         line_str =fr.readline()
    #         for j in range(32):
    #             return_vetor[]

=======
    horatio = 0.1
    datingdata_mat , dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, minvals = auto_norm(datingdata_mat)
    m = norm_mat.shape[0]
    numtestvecs = int(m * horatio) 
    error_count = 0
    for i in range(numtestvecs):
        classifiy_result = classfiy0(norm_mat[i, :], norm_mat[numtestvecs:,:], dating_labels[numtestvecs:], 3)
        print("the classifier came back with: %d, the real answer is: %d" %(classifiy_result, dating_labels[i]))
        if (classifiy_result != dating_labels[i]):
            error_count += 1.0
    print("the total error rate is: %f"%(error_count / float(numtestvecs)))
>>>>>>> 262550718b962c145b25669b154d036b1c076143
