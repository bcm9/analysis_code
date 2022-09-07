# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:48:26 2022

Fit logistic function to data, estimate threshold

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
#   ydiff=10
#   paramsopt, params_cov,residuals, r_squared, threshold = logisticfunction(X,y,ydiff)
def logisticfunction(X,y,ydiff):
    ######################################################################################################
    # L = upper asymptote, scales from [0,1] to [0,L]
    # b = bias to the output, changes range from [0,L] to [b,L+b]
    # k =  scaling the input, which remains in (-inf,inf)
    # x0 = inflection point
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
    
    # Find point within ydiff of asymptote
    idx=np.amin(np.where(np.round_(y2)==np.round_(np.amax(y2))-ydiff))
    threshold=X2[idx]
    
    ######################################################################################################
    # Compute r-squared
    residuals = y-logistic(X, *paramsopt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

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
    return(paramsopt, params_cov,residuals, r_squared, threshold)