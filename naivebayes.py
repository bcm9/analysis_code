# -*- coding: utf-8 -*-
"""
naivebayes(X,y,test_pc)
Returns Naive Bayes output for classification with scatter plot and confusion matrix.

X = matrix of predictor data
y = array of class data
test_pc = % of data for test set

@author: BCM
"""
######################################################################################################
# Import packages
import numpy as np 
import pandas as pd 
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def naivebayes(X,y,test_pc):
    # Naive (simple, independent) Bayes function uses Bayes' theorem to predict class probabilities
    # Most likely class is the class with the highest probability (Maximum A Posteriori - MAP)
    # Naive Bayes assumes feature independence, hence Naive
    ######################################################################################################
    # Split data into training/test sets
    # Set % of data for test set
    ntest=np.round_(X.shape[0]*(test_pc/100))
    # convert float to int
    ntest=int(ntest)
    
    # Randomly select indices to split rows into training/testing sets 
    trainidx=np.arange(0,X.shape[0])
    testidx=np.array(random.sample(range(X.shape[0]), ntest))
    trainidx = np.delete(trainidx, testidx)
    
    # Split predictor data into training/testing sets
    X_train = X[trainidx,:]
    X_test = X[testidx,:]
    
    # Split target data into training/testing sets
    y_train = y[trainidx]
    y_test = y[testidx]
    
    # # Split predictor data into training/testing sets
    # X_train = X[:-ntest]
    # X_test = X[-ntest:]
    
    # # Split target data into training/testing sets
    # y_train = y[:-ntest]
    # y_test = y[-ntest:]
    
    ######################################################################################################
    # Train a Gaussian Naive Bayes classifier
    # Setup model
    mdl = GaussianNB()
    
    # Fit model
    mdl.fit(X_train, y_train)
    print(mdl)
    
    # Predict y
    y_pred = mdl.predict(X_test)
    
    # Fit summary
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    
    ######################################################################################################
    # Metrics, evaluation
    # Print accuracy
    print('Model accuracy = {0:0.3f}'. format(accuracy_score(y_test, y_pred)))
    
    # Check if overfit
    print('Training set score = {:.3f}'.format(mdl.score(X_train, y_train)))
    print('Test set score = {:.3f}'.format(mdl.score(X_test, y_test)))
    
    ######################################################################################################
    # Visualization
    # Plot scatter
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=90, edgecolors='black');
    plt.title("Test Set Classification")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    # Make confusion matrix if single class 
    if np.unique(y_test).shape[0]==2:
        cfmat = confusion_matrix(y_test, y_pred)
        
        print('Confusion matrix\n\n', cfmat)
        print('\nTrue positives = ', cfmat[0,0])
        print('\nTrue negatives = ', cfmat[1,1])
        print('\nFalse positives = ', cfmat[0,1])
        print('\nFalse negatives = ', cfmat[1,0])
        
        # Visualize with heatmap
        cm_matrix = pd.DataFrame(data=cfmat, columns=['True positive:1', 'True negative:0'], 
                                         index=['Predicted positive:1', 'Predicted negative:0']) 
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='GnBu')
