# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:37:03 2019

@author: Matteo
"""

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import iqr
import numpy as np


def plotter(params_vect, numLayers, plot_hist = True):
    
    params_vect = np.asarray(params_vect)
    
    if (plot_hist == True):
        
        for i in range(numLayers):
            
            plt.figure(figsize=(10,6))
            
            # weights at params_vect[i], i = 0,2,4,..
            plt.subplot(1,2,1)
            j = 2*i
            weights = params_vect[j].flatten()
            sns.distplot(weights, rug=True, kde=False, norm_hist=False)
                        
            title = "Weights frequencies of layer "+str(i+1)
            plt.title(title,fontsize=20)
            plt.xlabel("Weight")
            plt.ylabel("Frequency")

            # biases OTOH at params_vect[i], i = 1,3,5,..
            plt.subplot(1,2,2)
            j = 2*i + 1
            biases = params_vect[j].flatten()
            sns.distplot(biases, rug=True, kde=False, norm_hist=False)
            title = "Biases frequencies of layer "+str(i+1)
            plt.title(title,fontsize=20)
            plt.xlabel("Bias")
            plt.ylabel("Frequency")
        plt.show()
        
    else:
        
        
        for i in range(numLayers):
            
            plt.figure(figsize=(10,6))
            plt.subplot(1,2,1)
            j = 2*i
            weights = params_vect[j].flatten()
            bandwidth = 1.06 * weights.std() * weights.size ** (-1 / 5.)
            lgnd = "Bandwidth " + '%.2f' % bandwidth
            sns.kdeplot(weights, bw=bandwidth, shade=True, label = lgnd)
            title = "Weights distribution of layer "+str(i+1)
            plt.title(title,fontsize=20)
            plt.xlabel("Weight")
            plt.ylabel("Pdf")

            plt.subplot(1,2,2)
            j = 2*i + 1
            biases = params_vect[j].flatten()
            bandwidth = 1.06 * biases.std() * biases.size ** (-1 / 5.)
            lgnd = "Bandwidth " + '%.2f' % bandwidth
            sns.kdeplot(biases, bw=bandwidth, shade=True, label = lgnd)
            title = "Biases distribution of layer "+str(i+1)
            plt.title(title,fontsize=20)
            plt.xlabel("Bias")
            plt.ylabel("Pdf")
        plt.show()
#enddef

def jointPlotter(numLayers, params_pre, params_post, plot_hist = True):
    
    params_pre = np.asarray(params_pre)
    params_post = np.asarray(params_post)
    
    if (plot_hist == True):
        
        for i in range(numLayers):
    
            j = 2*i
            # as before: weights on the even slots
            plt.figure(figsize=(14,6))
            ax = plt.subplot(1,2,1)
            w1 = params_pre[j].flatten()
            w2 = params_post[j].flatten()
            
            interMax = np.max(w2)
            interMin = np.min(w2)
            print(interMax, interMin)
            h = 2 * iqr(w2) * w2.shape[0]**(-1/3.)
            nBins = (interMax - interMin)/h
            bins = np.linspace(interMin, interMax, nBins)
            
#            plt.hist([w1, w2], bins, label=['Weights pre-training','Weights post-training'])
            plt.hist(w1, bins, alpha = 0.5, label = "Weights before training")
            plt.hist(w2, bins, alpha = 0.5, label = "Weights after training")
#            sns.distplot(w1, rug=False, kde=False, norm_hist=False, label = "Weights before train")
#            sns.distplot(w2, rug=False, kde=False, norm_hist=False, label = "Weights after train")
            title = "Weights frequencies of layer "+str(i+1)
            ax.set_xticks(bins)
            plt.xticks(bins, rotation = 45)
            plt.title(title,fontsize=20)
            plt.xlabel("Weight")
            plt.ylabel("Frequency")
            plt.legend()

            j = 2*i + 1
            # and biases on odd slots
            plt.subplot(1,2,2)
            b1 = params_pre[j].flatten()
            b2 = params_post[j].flatten()
            
            interMax = np.max(b2)
            interMin = np.min(b2)
            
            h = 0.5 * iqr(b2) * b2.shape[0]**(-1/3.)
            nBins = (interMax - interMin)/h
            
            bins = np.linspace(interMin, interMax, nBins)
            
#            plt.hist([b1, b2], bins, label=['Biases pre-training','Biases post-training'])
            plt.hist(b1, bins, alpha = 0.5, label = "Biases before training")
            plt.hist(b2, bins, alpha = 0.5, label = "Biases after training")
#            sns.distplot(b1, rug=False, kde=False, norm_hist=False, label = "Biases before train")
#            sns.distplot(b2, rug=False, kde=False, norm_hist=False, label = "Biases after train")
            title = "Biases frequencies of layer "+str(i+1)
            plt.xticks(bins, rotation = 45)
            plt.title(title,fontsize=20)
            plt.xlabel("Bias")
            plt.ylabel("Frequency")
            plt.legend()
        plt.show()

    else:
            
        for i in range(numLayers):
    
            j = 2*i
            plt.figure(figsize=(14,6))
            plt.subplot(1,2,1)
            w1 = params_pre[j].flatten()
            bandwidth = 1.06 * w1.std() * w1.size ** (-1 / 5.)
#            lgnd = "Bandwidth " + '%.2f' % bandwidth
            sns.kdeplot(w1, bw=bandwidth, shade=True, label = "Weights before train")

            w2 = params_post[j].flatten()
            bandwidth = 1.06 * w2.std() * w2.size ** (-1 / 5.)
#            lgnd = "Bandwidth " + '%.2f' % bandwidth
            sns.kdeplot(w2, bw=bandwidth, shade=True, label = "Weights after train")
            title = "Weights distribution of layer "+str(i+1)
            plt.title(title,fontsize=20)
            plt.xlabel("Weight")
            plt.ylabel("Pdf")
            plt.legend()

            j = 2*i + 1
            plt.subplot(1,2,2)
            b1 = params_pre[j].flatten()
            bandwidth = 1.06 * b1.std() * b1.size ** (-1 / 5.)
#            lgnd = "Bandwidth " + '%.2f' % bandwidth
            sns.kdeplot(b1, bw=bandwidth, shade=True, label = "Biases before train")

            b2 = params_post[j].flatten()
            bandwidth = 1.06 * b2.std() * b2.size ** (-1 / 5.)
#            lgnd = "Bandwidth " + '%.2f' % bandwidth
            sns.kdeplot(b2, bw=bandwidth, shade=True, label = "Biases after train")
            plt.legend()
            title = "Bias distribution of layer "+str(i+1)
            plt.title(title,fontsize=20)
            plt.xlabel("Bias")
            plt.ylabel("Pdf")
        plt.show()
#enddef
        
        
        
