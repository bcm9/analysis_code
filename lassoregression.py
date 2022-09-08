# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:24:25 2022

Lasso regression (L1 Regularization) with numpy, sklearn, and matplotlib

@author: BCM
"""

######################################################################################################
# Import packages, pre-processing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso

def lassoregression(X,y,test_pc,alpha):
    
    # Set % of data for test set
    ntest=np.round_(X.shape[0]*(test_pc/100))
    # convert float to int
    ntest=int(ntest)
    
    # Split predictor data into training/testing sets
    X_train = X[:-ntest]
    X_test = X[-ntest:]
    
    # Split target data into training/testing sets
    y_train = y[:-ntest]
    y_test = y[-ntest:]
    
    ######################################################################################################
    # Create model
    # Alpha equivalent to lamba, the penalty coefficient; increasing lambda = increasing shrinkage; lambda 0 = least squares
    lasso = Lasso(alpha=alpha)
    
    y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
    r2_score_lasso = r2_score(y_test, y_pred_lasso)
    
    # Coefficients
    print("coeffs: ", lasso.coef_)    
    # Intercept
    print("intercept: ", lasso.intercept_)    
    # r2 Coefficient of determination
    print("r-squared = %.2f" % r2_score_lasso)
    
    return lasso
    
    ######################################################################################################
    # Plot output if one predictor variable
    if X_test.shape[1]==1:
        plt.scatter(X_test, y_test,facecolors='none', edgecolors='black')
        plt.plot(X_test, y_pred_lasso, color="red", linewidth=2)
        plt.text(np.amax(X_test)-(np.amax(X_test)*0.3), np.amin(y_test)+(np.amin(y_test)*0.2),'r = ' + str(np.round_(r2_score(y_test, y_pred_lasso),2)),color='red')
        plt.title("Lasso Regression")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
