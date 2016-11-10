# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:29:27 2016

@author: wlq
"""
"""
the resource link:
http://blog.csdn.net/dinosoft/article/details/50734539
"""
import numpy as np
import pandas as pd
import time
from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import cross_val_score

#read train && test data set
image_train_data = pd.read_csv("D:/kaggleCompetition/HandWritingRecognition/train.csv")
image_test_data = pd.read_csv("D:/kaggleCompetition/HandWritingRecognition/test.csv")
#get the train data and the test data
image_train_x = image_train_data.iloc[:,1:].values
image_train_y = image_train_data.iloc[:,0].values
image_test_x = image_test_data.iloc[:,:].values
#for fast evaluation
image_train_x_small = image_train_x[:10000,:]
image_train_y_small = image_train_y[:10000]

from sklearn.linear_model import LogisticRegression
#begin time
start_time = time.clock()

#processing
lr_clf = LogisticRegression(penalty='l2',solver='lbfgs',multi_class='multinomial',max_iter=800,C=0.2)
parameters = {'penalty':['l2'] , 'C':[2e-2, 4e-2,8e-2, 12e-2, 2e-1]}

gs_clf =  GridSearchCV(lr_clf, parameters, n_jobs=1, verbose=True )

gs_clf.fit(image_train_x_small.astype('float')/256, image_train_y_small )

print()
for params, mean_score, scores in gs_clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"  % (mean_score, scores.std() * 2, params))
print()
#end time
elapsed = (time.clock() - start_time)
print("Time used:",elapsed)

#use all the data
start = time.clock()
lr_clf.fit(image_train_x,image_train_y)
elapsed = (time.clock() - start)
print("Training Time used:",int(elapsed/60) , "min")

result=lr_clf.predict(image_test_x)
result = np.c_[range(1,len(result)+1), result.astype(int)]
df_result = pd.DataFrame(result, columns=['ImageId', 'Label'])

df_result.to_csv('D:/kaggleCompetition/HandWritingRecognition/lr_result.csv', index=False)
#end time
elapsed = (time.clock() - start)
print("Test Time used:",int(elapsed/60) , "min")

