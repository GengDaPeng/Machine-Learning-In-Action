import numpy as np
import os
import operator


def classfiy0(inX, dataSet, labels, k):
    """ 创建一个分类器
    :type k: int
    Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labels - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
    """
    dataSize = dataSet.shape[0]    # 获取数据的大小
    diffMat = np.tile(inX, (dataSize, 1)) - dataSet  # 计算点与点 x和y坐标之间的差值
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
    sq_diffmat = diffMat ** 2  # 差值的平方
    sq_distance = sq_diffmat.sum(axis=1)  # 计算平方和即距离的平方sum(0)列相加，sum(1)行相加
    distance = sq_distance ** 0.5  # 开方计算出距离
    sort_distance_index = distance.argsort()   # 返回距离的索引值，且索引值是按照距离的大小排序的。
    """
    >>> b = np.array([1, 3, 2, 4, 5])
    >>> c = b.argsort()
    >>> c
    out: array([0, 2, 1, 3, 4], dtype=int64)
    """
    class_count = {}
    for i in range(k):   # 取出前K个元素类别
        vote_label = labels[sort_distance_index[i]]  # 选出k个最小的距离的标签
        class_count[vote_label] = class_count.get(vote_label, 0) + 1   # 计算选中的标签中各自的次数
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


def img2vector(filename):
    """
    图像转换成特征向量
    return_vect: 图像的一维向量
    """
    return_vect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):     # 遍历行
        line_str = fr.readline()   # 读取当前行
        for j in range(32):     # 遍历列
            return_vect[0, 32 * i + j] = int(line_str[j])   # 将每行添加到前一行的后面
    return return_vect      # 返回一维向量


def handwriting_classtest():
    """ 手写数字识别测试 """
    hwlabels = []
    trainfile_list = os.listdir('trainingDigits')      # 文件目录
    m = len(trainfile_list)     # 计算文件数量即矩阵行数
    training_mat = np.zeros((m, 1024))      # 初始化训练矩阵
    for i in range(m):
        filename_str = trainfile_list[i]    # 文件名
        file_str = filename_str.split('.')[0]   # 提取文件名，不带后缀
        classnum_str = int(file_str.split('_')[1])     # 提取分类数字
        hwlabels.append(classnum_str)
        training_mat[i, :] = img2vector('trainingDigits/%s' % (filename_str))
    testfile_list = os.listdir('testDigits')
    error_count = 0.0  # 统计错误数量
    test_m = len(testfile_list)
    for i in  range(test_m):
        filename_str = testfile_list[i]
        file_str = filename_str.split('.')[0]
        classnum_str = int(file_str.split('_')[1])
        vector_undertest = img2vector('trainingDigits/%s' % (filename_str))
        classifier_result = classfiy0(vector_undertest, training_mat, hwlabels, 3)
        print("the classifier came back with: %d, the real answer is %d" % (classifier_result, classnum_str))
        if (classifier_result != classnum_str):
            error_count += 1.0
    print("\nthe total number of errprs is : %d" % (error_count))
    print("\nthe total error rate is: %f" % (error_count / float(test_m)))


if __name__ == "__main__":
    handwriting_classtest()