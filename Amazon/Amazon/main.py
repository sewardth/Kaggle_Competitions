# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/Users/Tom/.spyder2/.temp.py
"""

import pandas as pd
import sklearn.ensemble as sk
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import cross_validation


train_set = '/Users/Tom/Dropbox/Kaggle/Amazon/Amazon/train.csv'
test_set = '/Users/Tom/Dropbox/Kaggle/Amazon/Amazon/test.csv'

data = pd.read_csv(train_set)
test = pd.read_csv(test_set)
x = data.RESOURCE
y = data.MGR_ID



#colors = np.where(data.ACTION == 1,'y','r')
#sizes = np.where(data.ACTION == 1,5,45)
#plt.scatter(x, y, s=sizes, c = colors)
#data['new_label'] = data.ACTION - 10



#print data

#fig = plt.figure(figsize=(15,15), dpi=500)
#ax = fig.add_subplot(111,projection='3d')
#ax.set_xlabel('RESOURCE')
#ax.set_ylabel('ROLE_ROLLUP_1')
#ax.set_zlabel('ROLE_CODE')
#ax.scatter(data.RESOURCE, data.ROLE_DEPTNAME, data.ROLE_TITLE,s = sizes, c = colors)



#Selects first 10000 rows and removes resource



new_data= array(data.ix[:,'MGR_ID':])
new_target = array(data.ACTION)


X_train, X_test, y_train, y_test = cross_validation.train_test_split(new_data, new_target, test_size = 0.2)

forest = sk.RandomForestClassifier(n_estimators = 100)




print 'Fitting Classifer'
forest = forest.fit(new_data, new_target)
scores = forest.score(X_test, y_test)
print 'Score: = %f' %(scores)


test_data = array(test.ix[:,'MGR_ID':])


print 'Predicting Test Data'
output = forest.predict(test_data)


print 'Saving Submission'
submission = pd.read_csv('/Users/Tom/Dropbox/Kaggle/Amazon/Amazon/sampleSubmission.csv')
submission.Action = output

submission.to_csv('/Users/Tom/Dropbox/Kaggle/Amazon/Amazon/sub_5.csv')