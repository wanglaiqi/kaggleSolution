#-*-coding:utf-8-*-

import numpy as np
import csv

def toInt(array):
    array = np.mat(array)
    m,n = np.shape(array)
    newArray = np.zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            newArray[i,j]=int(array[i,j])
    return newArray

def nomalizing(array):
    m,n = np.shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array

def loadTrainData():
    l = []
    with open('train.csv','rb') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    l=np.array(l)
    label=l[:,0]
    data=l[:,1:]
    return nomalizing(toInt(data)),toInt(label)

def loadTestData():
    l = []
    with open('test.csv','rb') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    data = np.array(l)
    return nomalizing(toInt(data))

def saveResult(result,csvName):
    with open(csvName,'wb') as myFile:
        myWriter = csv.writer(myFile)
        for i in result:
            tmp = []
            tmp.append(i)
            myWriter.writerow(tmp)

from sklearn.neighbors import KNeighborsClassifier
def knnClassify(trainData,trainLabel,testData):
    knnClf = KNeighborsClassifier()
    knnClf.fit(trainData,np.ravel(trainLabel))
    testLabel = knnClf.predict(testData)
    saveResult(testLabel,'sklearn_knn_Result.csv')
    return testLabel

def digitRecognition():
    trainData,trainLabel = loadTrainData()
    testData = loadTestData()

    result = knnClassify(trainData,trainLabel,testData)


