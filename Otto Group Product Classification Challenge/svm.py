# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:06:41 2015

@author: Tom
"""
from sklearn import svm

class svm():
    def __init__(self):
        #SVM model creation without probabilities
            clf = svm.SVC()
            clf.fit(features_train,target_train)