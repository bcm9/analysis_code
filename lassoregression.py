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
    
    # Standardise (z-score) data 
    X=(X-np.mean(X))/np.std(X)
    y=(y-np.mean(y))/np.std(y)
    
    # Set % of data for test set
    ntest=np.round_(X.shape[0]*(test_pc/100))
    # Convert float to int
    ntest=int(ntest)
    
    # Split predictor data into training/testing sets
    X_train = X[:-ntest]
    X_test = X[-ntest:]

    # Split target data into training/testing sets
    y_train = y[:-ntest]
    y_test = y[-ntest:]
    
    ######################################################################################################
    # Find optimal alpha (lamba) using k-fold cross validation
    # Alpha equivalent to lamba, the penalty coefficient; increasing lambda = increasing shrinkage; lambda 0 = least squares    
    k=5
    lasso = LassoCV(cv=k, random_state=0, max_iter=10000)
    
    # Fit model
    lasso.fit(X_train, y_train)
    
    # Optimal alpha
    print("optimal alpha = ",lasso.alpha_)  

    ######################################################################################################    
    # Now fit optimised model
    lasso_opt = Lasso(alpha=lasso.alpha_)
    lasso_opt.fit(X_train, y_train)
    
    y_lasso_opt = lasso.fit(X_train, y_train).predict(X_test)
    r2_score_lasso = r2_score(y_test, y_lasso_opt)
    
    # Coefficients
    print("coeffs = ", lasso_opt.coef_)    
    # Intercept
    print("intercept = ", lasso_opt.intercept_)    
    # r2 Coefficient of determination
    print("r-squared = %.2f" % r2_score_lasso)
        
    ######################################################################################################
    # Plot output if one predictor variable
    if X_test.shape[1]==1:
        plt.scatter(X_test, y_test,facecolors='none', edgecolors='black')
        plt.plot(X_test, y_lasso_opt, color="red", linewidth=2)
        plt.title("Lasso Regression; r = " + str(np.round_(r2_score(y_test, y_lasso_opt),2)))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        
    return lasso_opt
