# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:24:25 2022

Lasso regression (L1 Regularization) with cross validation to optimise alpha. 

@author: BCM
"""

######################################################################################################
# Import packages, pre-processing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

def lassoregression(X,y,test_pc):

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
    # Find optimal alpha (lamba) using cross validation
    # Alpha equivalent to lamba, the penalty coefficient; increasing lambda = increasing shrinkage; lambda 0 = least squares    
    
    # Lasso with 5 fold cross-validation
    lasso = LassoCV(cv=5, random_state=0, max_iter=10000)
    
    # Fit model
    lasso.fit(X_train, y_train)
    
    # Optimal alpha
    print("optimal alpha: ",lasso.alpha_)  

    ######################################################################################################    
    # Now fit optimised model
    lasso_opt = Lasso(alpha=lasso.alpha_)
    lasso_opt.fit(X_train, y_train)
    
    y_lasso_opt = lasso.fit(X_train, y_train).predict(X_test)
    r2_score_lasso = r2_score(y_test, y_lasso_opt)
    
    # Coefficients
    print("coeffs: ", lasso.coef_)    
    # Intercept
    print("intercept: ", lasso.intercept_)    
    # r2 Coefficient of determination
    print("r-squared = %.2f" % r2_score_lasso)
    
    return lasso_opt
    
    ######################################################################################################
    # Plot output if one predictor variable
    if X_test.shape[1]==1:
        plt.scatter(X_test, y_test,facecolors='none', edgecolors='black')
        plt.plot(X_test, y_lasso_opt, color="red", linewidth=2)
        plt.text(np.amax(X_test)-(np.amax(X_test)*0.3), np.amin(y_test)+(np.amin(y_test)*0.2),'r = ' + str(np.round_(r2_score(y_test, y_lasso_opt),2)),color='red')
        plt.title("Lasso Regression")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()