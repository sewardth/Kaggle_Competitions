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
X_train = X_train[:500]
y_train = y_train[:500]


#AdaBoost model creation     
#adaBoost_clf = ensemble.AdaBoostClassifier(n_estimators=100)
#adaBoost_clf.fit(X_train,y_train)
#adaBoost_predict = adaBoost_clf.predict(X_test)


#print adaBoost_clf.predict_proba(X_test)



#SVM model creation without probabilities
#clf = svm.SVC(kernel='linear')
#clf.fit(features_train,target_train)

#SVM model with probabilities
#svmProb_clf = svm.SVC(kernel='linear', probability=True)
#svmProb_clf.fit(X_train,y_train)
#svmProb_predict = svmProb_clf.predict(X_test)


#gridsearch svm classifer
#parameters = {'kernel':['linear', 'rbf', 'poly','sigmoid'], 'C':[1, 10,1000,10000,100000]}
#grid_svm = svm.SVC()
#grid_clf = grid_search.GridSearchCV(grid_svm,parameters)
#grid_clf.fit(X_train,y_train)

#print 'score for AdaBoost: ' + str(accuracy_score(y_test,adaBoost_predict))
#print 'score for SVM:' + str(accuracy_score(y_test,svm_predict))
#print 'score for SVM_Probs:' + str(accuracy_score(y_test,svmProb_predict))

#print 'Grid Search Results: '
#print grid_clf.best_estimator_




submission = submission_test['id']
submission_test = submission_test.drop('id',1)
prediction = clf.predict(submission_test)
sub = pd.DataFrame({'id':submission,
                    'Class_1':(pd.Series(np.array([1. if x ==1. else 0. for x in prediction]))),
                    'Class_2':(pd.Series(np.array([1. if x ==2. else 0. for x in prediction]))),
                    'Class_3':(pd.Series(np.array([1. if x ==3. else 0. for x in prediction]))),
                    'Class_4':(pd.Series(np.array([1. if x ==4. else 0. for x in prediction]))),
                    'Class_5':(pd.Series(np.array([1. if x ==5. else 0. for x in prediction]))),
                    'Class_6':(pd.Series(np.array([1. if x ==6. else 0. for x in prediction]))),
                    'Class_7':(pd.Series(np.array([1. if x ==7. else 0. for x in prediction]))),
                    'Class_8':(pd.Series(np.array([1. if x ==8. else 0. for x in prediction]))),
                    'Class_9':(pd.Series(np.array([1. if x ==9. else 0. for x in prediction])))})
print sub

sub.to_csv('svm.csv', index = False)


    
    










