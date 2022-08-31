# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:24:25 2022

Ordinary least squares regression with numpy, sklearn, and matplotlib

@author: BCM
"""

######################################################################################################
# Import packages, pre-processing
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def linearregression(X,y,test_pc):

    # Set % of data for test set
    ntest=np.round_(X.size*(test_pc/100))
    # convert float to int
    ntest=int(ntest)

    # Split predictor data into training/testing sets
    X_train = X[:-ntest]
    X_test = X[-ntest:]

    # Split target data into training/testing sets
    y_train = y[:-ntest]
    y_test = y[-ntest:]

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
    # Mean squared error
    print("mean squared error = %.2f" % mean_squared_error(y_test, y_pred))
    # r2 coefficient of determination
    print("r-squared = %.2f" % r2_score(y_test, y_pred))

    ######################################################################################################
    # Plot output
    plt.scatter(X_test, y_test,facecolors='none', edgecolors='black')
    plt.plot(X_test, y_pred, color="red", linewidth=2)
    plt.text(np.amax(X_test)-(np.amax(X_test)*0.3), np.amin(y_test)+(np.amin(y_test)*0.2),'r = ' + str(np.round_(r2_score(y_test, y_pred),2)),color='red')
    plt.title("Ordinary Least Squares Regression")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
