#-*-coding:utf-8-*-
#link is:
"""
https://github.com/wepe/Kaggle-Solution/blob/master/Digit%20Recognizer/kNN/
use-sklearn_knn_svm_NB.py
"""
import sys
import os
import numpy as np
import csv

#transfer string to int
def toInt(array):
    array = np.mat(array)
    m,n = np.shape(array)
    newArray = np.zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            newArray[i,j]=int(array[i,j])
    return newArray

#nomalizing the data
def nomalizing(array):
    m,n = np.shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array

#load  training dataSet
def loadTrainData():
    l = []
    with open('TrainData.csv','rb') as file:
        lines = csv.reader(file)
        try:
            for line in lines:
                l.append(line) #42001*785
        except csv.Error,e:
            sys.exit('%s' % (e))
    l.remove(l[0])
    l=np.array(l)
    label=l[:,0]
    data=l[:,1:]
    return nomalizing(toInt(data)),toInt(label) #label 1*42000 data 42000*784

#load the testing data
def loadTestData():
    l = []
    with open('TestData.csv','rb') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line) #28001 * 784
    l.remove(l[0])
    l = np.array(l)
    label=l[:,0]
    data=l[:,1:]
    return nomalizing(toInt(data)),toInt(label)

#save the predict result
def saveResult(result,csvName):
    with open(csvName,'wb') as myFile:
        myWriter = csv.writer(myFile)
        for i in result:
            tmp = []
            tmp.append(i)
            myWriter.writerow(tmp)

#use the knn method deal with the handWriting
#ravel()method is save vector into matrix
from sklearn.neighbors import KNeighborsClassifier
def knnClassify(trainData,trainLabel,testData):
    #default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    knnClf = KNeighborsClassifier()
    knnClf.fit(trainData,np.ravel(trainLabel))
    testLabel = knnClf.predict(testData)
    saveResult(testLabel,'SklearnKnnResult.csv')
    return testLabel

#use the sklearn svm method
from sklearn import svm
def svcClassify(trainData,trainLabel,testData):
    #default:C=1.0,kernel = 'rbf'. you can try kernel:‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’  
    svcClf = svm.SVC(C=5.0)
    svcClf.fit(trainData,np.ravel(trainLabel))
    testLabel = svcClf.predict(testData)
    saveResult(testLabel,'SklearnSvmResult.csv')
    return testLabel

#use the GaussianNB
from sklearn.naive_bayes import GaussianNB
def GaussianNBClassify(trainData,trainLabel,testData):
    nbClf = GaussianNB()
    nbClf.fit(trainData,np.ravel(trainLabel))
    testLabel=nbClf.predict(testData)
    saveResult(testLabel,'SklearnGaussianNBResult.csv')
    return testLabel

#use the MultinomialNB
from sklearn.naive_bayes import MultinomialNB
def MultinomialNBClassify(trainData,trainLabel,testData):
    #default alpha=1.0,Setting alpha = 1 is called Laplace smoothing, while alpha < 1 is called Lidstone smoothing
    nbClf = MultinomialNB(alpha=1.0)
    nbClf.fit(trainData,np.ravel(trainLabel))
    testLabel = nbClf.predict(testData)
    saveResult(testLabel,'SklearnMultinomialNBResult.csv')
    return testLabel

#dataSet:m*n labels: m*1 inX 1*n
def classify(inX,dataSet,labels,k):
    inX = np.mat(inX)
    dataSet = np.mat(dataSet)
    labels = np.mat(labels)
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSize,1)) - dataSet
    sqDiffMat = array(diffMat)**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i],0]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    classCount[voteIlabel] = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
        
def digitRecognition():
    trainData,trainLabel = loadTrainData()
    testData = loadTestData()
    result = knnClassify(trainData,trainLabel,testData)
    
def handWritingClass():
    trainData,trainLabel = loadTrainData()
    testData,testLabel = loadTestData()
    #testLabel = loadTestResult()
    m,n = np.shape(testData)
    errorCount = 0
    resultList = []
    for i in range(m):
        classifierResult = classify(testData[i],trainData,trainLabel.transpose(),5)
        resultList.append(classifierResult)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, testLabel[0,i])
        if(classifierResult!=testLabel[0,i]):
            errorCount+=1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(m))
    
        


