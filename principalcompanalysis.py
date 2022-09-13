# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:28:06 2022

coeff_loadings, score, latent, explained = principalcompanalysis(X)
conducts principal component analysis with biplot

X = matrix of data
coeff_loadings = each column of coeff contains coefficients for one principal component. columns are in order of descending component variance, latent. PC coefficients are recipe for counting any given PC
score = how each individual observation is composed of the PCs. matrix of PCs x observations
latent = eigenvalues of the covariance matrix, returned as a column vector
    eigenvalue is the total amount of variance in the variables in the dataset explained by the common factor
explained = contribution of each PC to variability in data

@author: BCM
"""
######################################################################################################
# Import packages, pre-processing    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def principalcompanalysis(X):
    # If X is not dataframe, convert
    if not isinstance(X, pd.DataFrame):
        X=pd.DataFrame(X)
        
    # remove rows with NaNs
    X=X.dropna()
    
    # Standardize data so mean = 0, stdev = 1 (z-score)
    stdX=(X-np.mean(X))/np.std(X)
    
    ######################################################################################################
    # Conduct PCA
    #   explained = the contribution of each PC to variability in data
    Xsize=np.shape(stdX)
    pca = PCA(n_components=Xsize[1])
    score  = pca.fit_transform(stdX)
    #score_data = pd.DataFrame(data = score
    #             , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4'])
    
    latent=pca.explained_variance_
    explained=pca.explained_variance_ratio_
    coeff_loadings=pca.components_
    
    ######################################################################################################
    # Plot output
    # Scree plot
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, latent.shape[0]),latent, 'k-o')
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalues")
    plt.axhline(y=1,linewidth= 1, color='r', linestyle= 'dotted')
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.1)
    plt.xticks(np.arange(len(latent)), np.arange(1, len(latent)+1))
    plt.yticks(np.arange(0, np.max(latent)+.5, step=0.5))

    def biplot(score,coeff_loadings,labels=None):
        xs = score[:,0]
        ys = score[:,1]
        n = coeff_loadings.shape[0]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        plt.scatter(xs * scalex,ys * scaley, s=30, facecolors='none', edgecolors='k')
        for i in range(n):
            plt.arrow(0, 0, coeff_loadings[i,0], coeff_loadings[i,1],color = 'r',alpha = 0.5)
            if labels is None:
                plt.text(coeff_loadings[i,0]* 1.15, coeff_loadings[i,1] * 1.15, X.columns[i], color = 'g', ha = 'center', va = 'center')
            else:
                plt.text(coeff_loadings[i,0]* 1.15, coeff_loadings[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))
        plt.grid(color = 'black', linestyle = '--', linewidth = 0.1)
    # plot the first 2 PCs
    plt.subplot(2, 1, 2)
    biplot(score[:,0:2],np.transpose(pca.components_[0:2, :]))
    plt.show()
    return coeff_loadings, score, latent, explained
