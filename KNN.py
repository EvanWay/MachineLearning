# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 15:18:47 2018

@author: Evan
"""

import numpy
import operator
import pandas
import matplotlib


def createDataSet():
    """构建一组训练数据（训练样本）"""
    group=numpy.array([
        [1.0,1.1],
        [1.0,1.0],
        [0.0,0.0],
        [0.0,0.1]
    ])
    labels=['A','A','B','B']
    return group,labels

def readcsv():
    file = pandas.read_csv('F:\pythonworkspace\MachineLearning\wc2018players.csv')
    man = file[['Pos.','Height','Weight']]
    #man = file.loc[:,['Height','Weight']].values
    pos = file['Pos.']
    posarr = numpy.array(pos).tolist()
    return man, pos
  

def classify0(inX, dataSet, lables, k):
    """
    方法只能针对实数，使用欧式距离来计算
    Args:
        inX：输入的新数据，表示为一个向量
        dataSet: 是样本集。表示为向量数组
        labels：相应样本集的标签。
        k：即所选的前K个（小于等于20）
    """
    dataSetSize = dataSet.shape[0]                          # shape[0]返回矩阵的行数
    diffMat = numpy.tile(inX, (dataSetSize, 1)) - dataSet   # tile之后inX=[ [0.1,0.1],[0.1,0.1],[0.1,0.1],[0.1,0.1] ],(x-x1),(y-y1)作差得到diffMat 
    sqDiffMat = diffMat**2                                  # 平方,即(x-x1)^2,(y-y1)^2
    sqDistance = sqDiffMat.sum(axis=1)                      # axis=1表示按照横轴，sum表示累加，即按照行进行累加.即(x-x1)^2+(y-y1)^2
    distances = sqDistance**0.5                             # 开根号
    sortedDistances = distances.argsort()                   # 按照升序进行快速排序,得到下标值的list
    classCount={}
    for i in range(k):
        label = lables[sortedDistances[i]]
        classCount[label] = classCount.get(label, 0) + 1    # 返回label的值，如果不存在就是0，然后+1
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#数据
group,labels=createDataSet()
#待分类数据
inX=[0.1,0.1]
#进行KNN分类
result=classify0(inX,group,labels,3)
print('分类结果：',result)
readcsv()




def normalize(dataset):    
    '''
    数据归一化
    '''
    return (dataset-dataset.min(0))/(dataset.max(0)-dataset.min(0))

#data,labels=read_data('testdata.txt')
#print('数据集：\n',data)
#print('标签集：\n',labels)
def read_data(filename):
    '''读取文本数据，格式：特征1、特征2等等'''
    f=open(filename,'rt')
    row_list=f.readlines()  #以每行作为列表
    f.close()
    data_array=[]
    labels_vector=[]
    while True:
        if not row_list:
            break
        row=row_list.pop(0).strip().split('\t') #去除换行号，分割制表符
        temp_data_row=[float(a) for a in row[:-1]]  #将字符型转换为浮点型
        data_array.append(temp_data_row) #取特征值
        labels_vector.append(row[-1])   #取最后一个作为类别标签
    return numpy.array(data_array),numpy.array(labels_vector)