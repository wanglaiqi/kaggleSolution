#import numpy as np
import pandas as pd
import csv
#import matplotlib.pyplot as plt
#%matplotlib inline
#from sklearn.preprocessing import StandardScaler


#save the predict result into csv file
def savePredictResult(image_y_predict,csvFielName):
    with open(csvFielName,'wb') as myFile:
        myWriter = csv.writer(myFile)
        for i in image_y_predict:
            tmp = []
            tmp.append(i)
            myWriter.writerow(tmp)
            
#transfer the data from string to int
def toInt(image_data):
    m,n = image_data.shape
    for i in xrange(m):
        for j in xrange(n):
            image_data.iat[i,j] = int(image_data.iat[i,j])
    return image_data

#normalizing the image data
def NormalizingData(image_data):
    m,n = image_data.shape
    for i in xrange(m):
        for j in xrange(n):
            if image_data.iat[i,j]!=0:
                image_data.iat[i,j]=1
    return image_data
          
#algorithm1: use the knn classifier
from sklearn.neighbors import KNeighborsClassifier
def knnClassifier(image_X_train,image_y_train,image_X_test):
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(image_X_train,image_y_train)
    print(knn_clf)
    #make prediction
    image_y_predict = knn_clf.predict(image_X_test)
    savePredictResult(image_y_predict,'SklearnKnnResult.csv')
	
#the main program about the program
def HandingWritingRecognition():
    #read train && test data set
    image_train_data = pd.read_csv("D:/kaggleCompetition/HandWritingRecognition/train.csv")
    image_test_data = pd.read_csv("D:/kaggleCompetition/HandWritingRecognition/test.csv")
    #get the train data and the test data
    image_train_X = image_train_data.iloc[:,1:]
    image_train_y = image_train_data.iloc[:,1]
    image_test_X = image_test_data.iloc[:,:]
    #transfer the data from string to int
    #image_train_X = toInt(image_train_X)
    #image_train_y = toInt(image_train_y)
    #image_test_X = toInt(image_test_X)
    #normalizing the data
    image_train_norm_X = NormalizingData(image_train_X)
    image_test_norm_X = NormalizingData(image_test_X)
    """
    stdsc = StandardScaler()
    image_train_std_X = stdsc.fit_transform(image_train_X)
    image_test_std_X = stdsc.transform(image_test_X)
    """
    #algorithm1: knn classifier
    knnClassifier(image_train_norm_X,image_train_y,image_test_norm_X)
    #algorithm2: Lg classifier
