# -*- coding: utf-8 -*-
"""
Created on Thu Apr 09 14:46:30 2015

@author: seward_t
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, linear_model, metrics
import math

training_data = pd.read_csv('train.csv')
training_data.drop('Id',axis=1, inplace=True)
demographic_data = training_data[['P'+str(x) for x in range(1,38)]]
training_data.drop(['P'+str(x) for x in range(1,38)], axis=1, inplace=True)

scaler = preprocessing.Normalizer()
demographic_data = pd.DataFrame(scaler.fit_transform(demographic_data.values.astype(float)), columns=['P'+str(x) for x in range(1,38)])

training_data['Open_Year'] = training_data['Open Date'].map(lambda x: x[-4:]).astype(int)
training_data['Open_Month'] = training_data['Open Date'].map(lambda x: x[:2]).astype(int)
training_data.drop('Open Date',axis=1, inplace=True)

target = training_data.revenue.values.astype(float)

training_data.drop(['revenue','City'],axis=1, inplace=True)


city_group_encoder = preprocessing.LabelEncoder()
restaurant_type_encoder = preprocessing.LabelEncoder()

training_data['City Group'] = city_group_encoder.fit_transform(training_data['City Group'].values)
training_data['Type'] = restaurant_type_encoder.fit_transform(training_data['Type'].values)

final_data = training_data.join(demographic_data)

category_encoder = preprocessing.OneHotEncoder(categorical_features=[0,1,2,3])
features = category_encoder.fit_transform(final_data.values).toarray()


skf = cross_validation.StratifiedKFold(target, n_folds=10, shuffle =True)
for train_index, test_index in skf:
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = target[train_index], target[test_index]
        
        

clf = linear_model.ElasticNet()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

print 'Training R2: ' + str(clf.score(X_train, y_train))
print 'Test R2: ' + str(clf.score(X_test, y_test))
print 'Root Mean Squared Error: ' + str(math.sqrt(metrics.mean_squared_error(y_test, prediction)))
print 'Metrics R2: ' + str(metrics.r2_score(y_test, prediction))

