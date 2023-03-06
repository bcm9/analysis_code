#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculates Fleiss' kappa

def fleissk_calculator(n,K,k,k_table):
    return fleissk
    
    n = raters
    N = subjects
    k = categories
    k_table = N x k matrix
    
example:
    n=14
    N=10
    k=5
    
    import numpy as np

    k_table=np.array([[0,0,0,0,14],[0,2,6,4,2],[0,0,3,5,6],[0,3,9,2,0],[2,2,8,1,1],[7,7,0,0,0]\
                      ,[3,2,6,3,0],[2,5,3,2,2],[6,5,2,1,0],[0,2,2,3,7]])
    
Created on Tue Jun 21 18:07:50 2022

@author: bcm9
"""
def fleissk_calculator(n,N,k,k_table):
    p_j=sum(k_table)/sum(sum(k_table))

    P_i = []    
    for nK in range(0, N):
        P_i.append((1/(n*(n-1))*(sum((k_table[nK,:])**2)-n)))
        
    P_hat=sum(P_i)*(1/N)
    
    P_hat_e=sum(p_j**2)
    
    fleissk=(P_hat-P_hat_e)/(1-P_hat_e)
    
    return fleissk
