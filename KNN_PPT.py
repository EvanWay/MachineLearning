# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt;
import operator
def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[ 0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,lables,k):
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distance=sqDistances**.5
    sortedDistIndicies=distance.argsort()
    classCount={}
    for i in range(k):
        voteILabel=labels[sortedDistIndicies[i]]
        classCount[voteILabel]=classCount.get(voteILabel,0)+1
        items=classCount.items()
        sortedClassCount=sorted(items,
        key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

group,labels=createDataSet()

v1=[.3,.3]
v2=[.6,.5]
l1=classify0(v1,group,labels,2)
l2=classify0(v2,group,labels,2)

x=[x[0] for x in group]

y=[x[1] for x in group]

plt.scatter(x,y,marker='.')
plt.scatter([v1[0],v2[0]],[v1[1],v2[1]],marker='x',c='r')

plt.show()

