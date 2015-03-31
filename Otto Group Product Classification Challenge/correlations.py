# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:13:51 2015

@author: seward_t
"""

import pandas as pd


class Offenders():
    def __init__(self, data, correlation_limit=.5):
        self.cl = correlation_limit
        features = data.corr()
        self.values = self.recursive_feature_selection(features)
        
        

        
    def stop_features(self):
        return self.values[:-1]


    def recursive_feature_selection(self, features):
        feature_rows = features.index.tolist()
        correlation_map=[]
        for i, feature in enumerate(features.columns):
            correlation_map.append([feature])
            for index, value in enumerate(features[feature].values):
                if abs(value) >= self.cl and value != 1.:
                    correlation_map[i].append(feature_rows[index])
         
        
        unpacked = ' '.join([' '.join(x) for x in correlation_map if len(x) >1]).split()
        values =  sorted(set([(x, unpacked.count(x)) for x in unpacked if unpacked.count(x) > 1]), key=lambda z: z[1])
        if len(values) == 0:
            return [1]
        
        else:
            bad_feature = values.pop()
            return [bad_feature[0]] + self.recursive_feature_selection(features.drop(bad_feature[0], axis=1).drop(bad_feature[0], axis=0))
        
 
if __name__ == "__main__":
    train_data = pd.DataFrame(pd.read_csv('train.csv'))
    #train_data = train_data[(train_data.target == 'Class_2') | (train_data.target == 'Class_3') | (train_data.target == 'Class_4') ]
    target_train = train_data['target']
    features_train = train_data.drop('target',1).drop('id',1)
    print Offenders(features_train).stop_features()