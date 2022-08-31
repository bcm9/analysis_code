# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:48:26 2022

Logistic function

@author: BC478
"""

######################################################################################################
# Import packages, pre-processing    
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# example:
#   X=np.array([0.5, 1, 2, 4, 6, 12, 18, 36, 48, 60])
#   y=np.array([10, 8, 20, 18, 50, 75, 85, 82, 88, 86])
def logisticfunction_bcmf(X,y,ydiff):
    ######################################################################################################
    # L scales output from [0,1] to [0,L]
    # b adds bias to output and changes range from [0,L] to [b,L+b]
    # k scales input, which remains in (-inf,inf)
    # x0 inflection point
    def logistic(x, L ,x0, k, b):
        y = L / (1 + np.exp(-k*(x-x0))) + b
        return (y)
    
    ######################################################################################################
    # Fit logistic function using curve fit
    # Starting point for parameters
    params0 = [max(y), np.median(X),1,min(y)]
    
    # Fit function
    paramsopt, params_cov = curve_fit(logistic, X, y, params0)
    
    # Create array of x values to fit function to
    X2=np.arange(np.amin(X), np.amax(X),0.1)
    y2 = logistic(X2, *paramsopt)
    
    # find point within ydiff of asymptote
    idx=np.amin(np.where(np.round_(y2)==np.round_(np.amax(y2))-ydiff))
    threshold=X2[idx]
    
    ######################################################################################################
    # Plot output
    plt.plot(X, y, 'o', label='data',color="black")
    plt.plot(X2,y2, label='fit',color="red", linewidth=2)
    plt.legend(loc='lower right')
    plt.title("Logistic Function")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axvline(threshold,color='blue',ls=':')
    plt.show()
