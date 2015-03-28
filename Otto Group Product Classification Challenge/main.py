# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import smtplib
from sklearn import preprocessing, cross_validation, ensemble, metrics





def training_data():
    #convert datasets to Pandas DataFrames
    train_data = pd.DataFrame(pd.read_csv('train.csv'))
    target_train = train_data['target']
    features_train = train_data.drop('target',1).drop('id',1)
    return (features_train, target_train)


def submission_data():
    submission = pd.DataFrame(pd.read_csv('test.csv'))
    sub_id = submission['id']
    sub_features = submission.drop('id',1)
    return (sub_features, sub_id)
    

def transform_features(train, submission):
    scaler = preprocessing.StandardScaler()
    train = scaler.fit_transform(train)
    submit = scaler.transform(submission)
    return (train, submit)
    
    
def train_test_split(features, target):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        features, target, test_size=0.2, random_state=4)
    return {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}
    
   
def  StratifiedSplit(features, target):
    skf = cross_validation.StratifiedKFold(target_train, n_folds=10)
    for train_index, test_index in skf:
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]
    return {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}
    

def classifer(X,y):
    clf = ensemble.RandomForestClassifier(n_estimators = 500, n_jobs =-1)
    clf.fit(X,y)
    return clf
    
    
    
def score_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    class_probabilities = model.predict_proba(X_test)
    scores = cross_validation.cross_val_score(model, X_test, y_test)
    
    print 'Model Score: ' + str(model.score(X_test, y_test))
    print 'Cross Validation Score: ' + " %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    print 'F1 Score: ' + str(metrics.f1_score(y_test, prediction))
    print 'Log-Loss : ' + str(metrics.log_loss(y_test, class_probabilities))
    


def main():
    #pull training set
    features, target = training_data()
    
    #encode target classes
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(target.values).astype(float)
    
    #pull submission data
    sub_features, sub_id = submission_data()
    
    #normalize feature sets
    train_X, sub_X = transform_features(features.values.astype(float), sub_features.values.astype(float))
    
    #create training / testing splits
    data = train_test_split(train_X, train_y)
    stratified_data = StratifiedSplit(train_X, train_y)
    
    #train model
    model = classifer(stratified_data['X_train'], stratified_data['y_train'])
    
    #score model
    score_model(model, stratified_data['X_test'], stratified_data['y_test'])
    
    


    
if __name__ == "__main__":
    main()









