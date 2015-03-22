# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:55:36 2015

@author: Tom
"""

import pandas as pd
from sklearn import cross_validation


class data():
    def __init__(self, training_data, test_data):
        #convert datasets to Pandas DataFrames
        train_data = pd.DataFrame(pd.read_csv(training_data))
        self.submission_data = pd.DataFrame(pd.read_csv(test_data))

        target_train = train_data['target']
        features_train =  train_data.drop('target',1).drop('id',1)

        #convert string targets to numeric
        target_train = target_train.map(lambda x: float(x.replace('Class_','')))
        
        #create test and training sets
        self.X_train, self.X_test, self.y_train, self.y_test = cross_validation.train_test_split(
            features_train, target_train, test_size=0.3, random_state=0)

    