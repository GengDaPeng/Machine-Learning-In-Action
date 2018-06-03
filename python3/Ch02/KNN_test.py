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
dating_mat, dating_label = knn.fire2matrix('datingTestSet2.txt')

print(dating_label[0:3])
print(dating_mat[0:3])
# 测试成功

