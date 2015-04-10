# -*- coding: utf-8 -*-
"""
Created on Thu Apr 09 14:46:30 2015

@author: seward_t
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, ensemble, metrics, grid_search
import math

training_data = pd.read_csv('train.csv')


city_group_encoder = preprocessing.LabelEncoder()
type_encoder = preprocessing.LabelEncoder()

training_data['Open_Month'] = training_data['Open Date'].map(lambda x: x[:2]).astype(float)
training_data['Open_Year'] = training_data['Open Date'].map(lambda x: x[-4:]).astype(float)
training_data['City Group'] = city_group_encoder.fit_transform(training_data['City Group'].values).astype(float)
training_data['Type'] = type_encoder.fit_transform(training_data['Type'].values).astype(float)

target = training_data.revenue.values.astype(float)

training_data.drop(['Id','Open Date','revenue','City'],axis=1, inplace=True)

features = training_data.values.astype(float)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    features, target, test_size=0.2, random_state=4)


#clf = ensemble.GradientBoostingRegressor(loss='lad',n_estimators =3000, learning_rate=.01, max_depth=5, max_features=None, min_samples_leaf=2)
clf = ensemble.RandomForestRegressor(n_estimators=100)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print 'R2 Score: ' +str(clf.score(X_test, y_test))
print 'Root Mean Squared Error: ' + str(math.sqrt(metrics.mean_squared_error(y_test, pred)))


#Grid Search
#params={'loss':['ls','huber','lad']}
##params={'loss':['ls','huber','lad'],'learning_rate':[.1,.001,.0001,.05], 'max_depth':[1,3,5,7,9], 'min_samples_split':[1,2,3,5], 'min_samples_leaf':[1,2,4,6,8], 'max_features':[None, 'Auto'],'subsample':[1,.05,.08,.01]}
#clf = grid_search.GridSearchCV(ensemble.GradientBoostingRegressor(n_estimators =3000), params, n_jobs=-1,scoring='mean_squared_error')
#clf.fit(X_train, y_train)
#
#print clf.best_params_