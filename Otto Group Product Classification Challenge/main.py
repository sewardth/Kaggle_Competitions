# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import cross_validation
from sklearn import ensemble
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import grid_search
import numpy as np



#convert datasets to Pandas DataFrames
train_data = pd.DataFrame(pd.read_csv('train.csv'))
submission_test = pd.DataFrame(pd.read_csv('test.csv'))

target_train = train_data['target']
features_train =  train_data.drop('target',1).drop('id',1)

#convert string targets to numeric
target_train = target_train.map(lambda x: float(x.replace('Class_','')))


#create test and training sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     features_train, target_train, test_size=0.3, random_state=0)

#reduce training set for speed
X_train = X_train[:5000]
y_train = y_train[:5000]

#random forest creation
clf = ensemble.RandomForestClassifier(n_estimators=1000)
clf.fit(features_train, target_train)
#predict = clf.predict(X_test)
print accuracy_score(y_test,predict)
print clf.predict_proba(X_test)[0]




#gridsearch svm classifer
#parameters = {'n_estimators':[10,50,100,1000], 'criterion':['gini','entropy'],'max_features':['auto',None], 'min_samples_split':[2,10,30,50]}
#grid_svm = ensemble.RandomForestClassifier()
#grid_clf = grid_search.GridSearchCV(grid_svm,parameters)
#grid_clf.fit(X_train,y_train)

#print 'Grid Search Results: '
#print grid_clf.best_estimator_



#
submission = submission_test['id']
submission_test = submission_test.drop('id',1)
prediction = clf.predict(submission_test)


sub = pd.DataFrame({'id':submission,
                    'Class_1':(pd.Series(np.array([x[0] for x in prediction]))),
                    'Class_2':(pd.Series(np.array([x[1] for x in prediction]))),
                    'Class_3':(pd.Series(np.array([x[2] for x in prediction]))),
                    'Class_4':(pd.Series(np.array([x[3] for x in prediction]))),
                    'Class_5':(pd.Series(np.array([x[4] for x in prediction]))),
                    'Class_6':(pd.Series(np.array([x[5] for x in prediction]))),
                    'Class_7':(pd.Series(np.array([x[6] for x in prediction]))),
                    'Class_8':(pd.Series(np.array([x[7] for x in prediction]))),
                    'Class_9':(pd.Series(np.array([x[8] for x in prediction])))})
#print sub

headers =['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']
sub.to_csv('svm.csv', index = False, cols=headers)


    
    










