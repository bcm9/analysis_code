# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:24:25 2022

mdl=olsregression(X,y,test_pc)
Ordinary least squares regression with plot

X = matrix of predictor data
y = array of target data
test_pc = % of data for test set
mdl = model output

@author: BCM
"""

######################################################################################################
# Import packages, pre-processing
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def olsregression(X,y,test_pc):

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

    # Check size of data
    print("training set size = \n", X_train.size)
    print("test set size = \n", X_test.size)

    ######################################################################################################
    # Create linear regression model
    mdl = linear_model.LinearRegression()

    # Train the model using the training sets
    mdl.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = mdl.predict(X_test)

    # Coefficients
    print("coefficients = \n", mdl.coef_)
    # Intercept
    print("intercept = ", mdl.intercept_)  
    # Mean squared error
    print("mean squared error = %.2f" % mean_squared_error(y_test, y_pred))
    # r2 coefficient of determination
    print("r-squared = %.2f" % r2_score(y_test, y_pred))
        
    ######################################################################################################
    # Plot output if one predictor variable
    if X_test.shape[1]==1:
        plt.scatter(X_test, y_test,facecolors='none', edgecolors='black')
        plt.plot(X_test, y_pred, color="red", linewidth=2)
        plt.title("OLS Regression; r = " + str(np.round_(r2_score(y_test, y_pred),2)))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(color = 'black', linestyle = '--', linewidth = 0.1)
        plt.show()
    else:
        # Plot predictor coefficient
        plt.bar([x for x in range(len(mdl.coef_))], mdl.coef_)
        plt.xlabel("Predictor")
        plt.ylabel("Coefficients")
        plt.show()
        
    return mdl
