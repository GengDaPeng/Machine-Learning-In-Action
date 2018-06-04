# -*- coding:utf-8 -*-
#***************************************************/
# Filename:KNN_test.py
# CreateTime:Saturday, 2nd June 2018 6:03:04 pm
# Author:GengDaPeng (bingshan222@hotamil.com)
# Last Modified: Saturday, 2nd June 2018 6:03:52 pm
#***************************************************/

import KNN_py3 as knn 
import os


os.path = 'E:/Github/Machine-Learning-In-Action/python3/Ch02/'

#  fire2matrix函数测试
dating_mat, dating_label = knn.file2matrix('datingTestSet2.txt')

print(dating_label[0:3], len(dating_label))
print(dating_mat[0:3], dating_mat.shape)
print(type(dating_mat))
print(type(dating_label))
# 测试成功
print('**********************************************')
normdata, ranges, minvals = knn.auto_norm(dating_mat)
print(normdata[:10], type(normdata.shape[0]))
print(ranges)
print(minvals)
print('----------------------------------------')
m = normdata.shape[0]
a = knn.classfiy0(normdata[1,:], normdata[100:m,:], dating_label[100:m], 3)
print(a)