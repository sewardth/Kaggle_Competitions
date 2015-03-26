# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import cross_validation
from sklearn import ensemble
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import grid_search
import numpy as np
from sklearn import preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import BernoulliRBM



#convert datasets to Pandas DataFrames
train_data = pd.DataFrame(pd.read_csv('train.csv'))
submission_test = pd.DataFrame(pd.read_csv('test.csv'))

target_train = train_data['target']
features_train = preprocessing.normalize(train_data.drop('target',1).drop('id',1).values.astype(float))

    


#scale data
scaler = preprocessing.MinMaxScaler()
features_train = scaler.fit_transform(features_train)

pca = PCA(n_components='mle', whiten=True)
features_train = pca.fit_transform(features_train)

#convert string targets to numeric
target_train = target_train.map(lambda x: float(x.replace('Class_',''))).values

#Neural Net
model = BernoulliRBM()
model.fit_transform(features_train)

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(
#     features_train, target_train, test_size=0.3, random_state=4)







#reduce training set for speed
#X_train = X_train[:1000]
#y_train = y_train[:1000]


#svm
#clf = svm.SVC(kernel='linear')
#clf.fit(X_train,y_train)





#random forest creation
clf = ensemble.RandomForestClassifier(n_estimators=1000, n_jobs=-1)
clf.fit(features_train, target_train)
#predict = clf.predict(X_test)
#print accuracy_score(y_test,predict)
scores = cross_validation.cross_val_score(clf,features_train,target_train)
print scores.mean()




##gridsearch svm classifer
#parameters = {'loss':['hinge','log','modified_huber','squared_hinge','perceptron'], 'penalty':['l2','elasticnet']}
#grid_svm =  SGDClassifier()
#grid_clf = grid_search.GridSearchCV(grid_svm,parameters)
#grid_clf.fit(X_train,y_train)
###
#print 'Grid Search Results: '
#print grid_clf.best_estimator_



#
submission = submission_test['id']
submission_test = preprocessing.normalize(submission_test.drop('id',1).values.astype(float))
submission_test = scaler.transform(submission_test)
submission_test = pca.transform(submission_test)
submission_test = model.transform(submission_test)
prediction = np.around(clf.predict_proba(submission_test),decimals=6)


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
sub.to_csv('SGDRegularized.csv', index = False, cols=headers)


    
    










