# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 19:56:15 2016

@author: wlq
"""
import numpy as np
import pandas as pd
import time
from sklearn.cross_validation import cross_val_score

#read train && test data set
image_train_data = pd.read_csv("D:/kaggleCompetition/HandWritingRecognition/train.csv")
image_test_data = pd.read_csv("D:/kaggleCompetition/HandWritingRecognition/test.csv")
#get the train data and the test data
image_train_x = image_train_data.iloc[:,1:].values
image_train_y = image_train_data.iloc[:,0].values
#for fast evaluation
image_train_x_small = image_train_x[:10000,:]
image_train_y_small = image_train_y[:10000]

image_test_x = image_test_data.iloc[:,:].values

from sklearn.neighbors import KNeighborsClassifier
#begin time
start = time.clock()
#progressing
knn_clf = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', weights='distance', p=3)
score = cross_val_score(knn_clf, image_train_x_small, image_train_y_small, cv=3)
print(score.mean())
#end time
elapsed = (time.clock() - start)
print("Time used:",int(elapsed), "s")

start = time.clock()
knn_clf.fit(image_train_x,image_train_y)
elapsed = (time.clock()-start)
print("training timed used:",int(elapsed/60),"min")

knn_result = knn_clf.predict(image_test_x)
knn_result = np.c_[range(1,len(knn_result)+1), knn_result.astype(int)]
df_knn_result = pd.DataFrame(knn_result, columns=['ImageId', 'Label'])
df_knn_result.to_csv('D:/kaggleCompetition/HandWritingRecognition/knnresult.csv',index = False)
#end time
elapsed = (time.clock()-start)
print("Test Time used:",int(elapsed/60),"min")
