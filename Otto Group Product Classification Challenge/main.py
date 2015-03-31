# -*- coding: utf-8 -*-
 
import pandas as pd
import numpy as np
import smtplib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from correlations import Offenders
from sklearn import preprocessing, cross_validation, ensemble, metrics, svm, grid_search, decomposition, feature_selection, linear_model
 
 
 
 
 
def training_data():
    #convert datasets to Pandas DataFrames
    train_data = pd.DataFrame(pd.read_csv('train.csv'))
    #train_data = train_data[(train_data.target == 'Class_2') | (train_data.target == 'Class_3') | (train_data.target == 'Class_4') ]
    target_train = train_data['target']
    features_train = train_data.drop('target',1).drop('id',1)

    return (features_train, target_train)
 
 
def submission_data():
    submission = pd.DataFrame(pd.read_csv('test.csv'))
    sub_id = submission['id']
    sub_features = submission.drop('id',1)

    return (sub_features, sub_id)
    
 
def transform_features(train, submission, principal_components=False):
    #scale data
    scaler = preprocessing.StandardScaler()    
    train = scaler.fit_transform(train)
    submit = scaler.transform(submission)
    
    #apply PCA
    if principal_components:
        print 'Fitting Principal Components'
        pca = decomposition.PCA(n_components = 'mle', whiten=True )
        train = pca.fit_transform(train)
        submit = pca.transform(submit)
    
    return (train, submit)
    
    
def train_test_split(features, target):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    features, target, test_size=0.2, random_state=4)
    return {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}


def StratifiedSplit(features, target):
    skf = cross_validation.StratifiedKFold(target, n_folds=10, shuffle =True)
    for train_index, test_index in skf:
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]
    return {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}
 
 
def classifer(X,y):
    clf = svm.SVC(kernel='rbf', C=10, gamma =0.01, class_weight = None, probability = True)
    #clf = ensemble.RandomForestClassifier(n_estimators =200, n_jobs =-1)
    clf.fit(X,y)
    return clf


def score_model(model, X_test, y_test, X_train, y_train, encoder, plot=True):
    prediction = model.predict(X_test)
    class_probabilities = model.predict_proba(X_test)
    scores = cross_validation.cross_val_score(model, X_test, y_test)
    print 'Model Test Score: ' + str(model.score(X_test, y_test))
    print 'Model Training Score: ' + str(model.score(X_train, y_train))
    print 'Cross Validation Score: ' + " %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    print 'F1 Score: ' + str(metrics.f1_score(y_test, prediction))
    print 'Log-Loss: ' + str(metrics.log_loss(y_test, class_probabilities))
 
 
    #calculate and plot confusion matrix
    calculate_confusion_matrix(y_test, prediction, encoder)
 
 
def calculate_confusion_matrix(y_test, y_pred, encoder):
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, encoder, title='Normalized confusion matrix')


 
def plot_confusion_matrix(cm, encoder, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(encoder.classes_))
    plt.xticks(tick_marks, encoder.classes_, rotation=45)
    plt.yticks(tick_marks, encoder.classes_)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


 
def parameter_tune(model, parameters, X_train, y_train):
    grid_clf = grid_search.GridSearchCV(model,parameters, n_jobs =-1)
    grid_clf.fit(X_train,y_train)
    #print 'Grid Search Results: '
    #print grid_clf.best_estimator_
    #print 'Grid Scores: '
    #print grid_clf.grid_scores_
    print 'Best Parameters: '
    print grid_clf.best_params_



def key_features(X_train, y_train, sub, varience_test=True):
    print 'Features before reduction: ' + str(len(X_train[0]))
    if varience_test:
        #remove features with low variance
        sel = feature_selection.VarianceThreshold(threshold=(.8 * (1 - .8)))
        X_train = sel.fit_transform(X_train)
        sub = sel.transform(sub)
        print 'Features after variance reduction: ' +str(len(X_train[0]))
    
    estimator = linear_model.SGDClassifier(n_jobs =-1, class_weight ='auto')
    selector = feature_selection.RFECV(estimator, step=1, cv=5)
    features = selector.fit_transform(X_train, y_train)
    submission = selector.transform(sub)
    
    print 'Features after recursive elimination: ' + str(len(features[0]))
    
    return (features, submission)
    

def file_output(model, sub_features, sub_ids, title, encoder):
    columns = encoder.classes_
    prediction = model.predict_proba(sub_features)
    dataframe = pd.DataFrame.from_records(prediction, index=sub_ids, columns=columns)
    dataframe.index.name = 'id'
    path = '\Submissions\{}.csv'.format(title)
    dataframe.to_csv(path)


def remove_correlations(features, sub_features, cl=.5):
    #find correlations
    offenders = Offenders(features, correlation_limit=cl)
    drop_features = offenders.stop_features()
    
    #transform feature sets
    features = features.drop(drop_features,axis=1)
    sub_features = sub_features.drop(drop_features, axis=1)
    
    return (features, sub_features)
    
 
if __name__ == "__main__":
    #pull training set
    features, target = training_data()
    
    #encode target classes
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(target.values).astype(float)
    
    #pull submission data
    sub_features, sub_id = submission_data()
    
    #remove correlations
    features, sub_features = remove_correlations(features, sub_features, cl=.8)
    
    #normalize feature sets
    train_X, sub_X = transform_features(features.values.astype(float), sub_features.values.astype(float), principal_components=True)
    
    #Search for key features
    train_X, sub_X = key_features(train_X, train_y, sub_X)
    
    #create training / testing splits
    data = train_test_split(train_X, train_y)
    stratified_data = StratifiedSplit(train_X, train_y)
    
    
    #train model
    model = classifer(stratified_data['X_train'], stratified_data['y_train'])
    
    #score model
    score_model(model, stratified_data['X_test'], stratified_data['y_test'], stratified_data['X_train'], stratified_data['y_train'],encoder)
    
    #Grid Search
    #params ={'C':[1,10],'kernel':['rbf','linear'], 'gamma':[.01,0.0,.1],'class_weight':['auto',None]}
    #parameter_tune(svm.SVC(), params, data['X_train'][:10000], data['y_train'][:10000])
     
    #file_output(model, sub_X, sub_id, 'Optimized SVM', encoder)