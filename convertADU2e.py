#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:16:17 2020

@author: danielgift
"""
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import scipy.stats
from scipy.signal import find_peaks

#Finds ADU to electron conversion
def findConv(dat, ID, quad, verbose, fast):
    #This fits 2 gaussians to our data. We assume its all 0 and 1 electron events, which is pretty true
    def doubleGauss(val, normalizer,width,ADU,offset,n2):
        return (totArea-normalizer/stepsize)*stepsize/(np.sqrt(2*np.pi*width**2))*np.exp(-(val-offset)**2/(2*width**2))+n2/(np.sqrt(2*np.pi*width**2))*np.exp(-(val-ADU)**2/(2*width**2))
    e1p = roughGuessConv(dat) #Get a rough idea of where the separation is 
    #This will look only at the data that is 0 or 1 electrons
    #Ballpark parameters estimated based on our ADU-to-electron guess
    minADU = -.5*e1p
    maxADU = 1.4*e1p#3.2*e1p
    stepsize = e1p/40
    xdata = np.arange(minADU+stepsize/2., maxADU-stepsize/2.,stepsize)
    ydata = np.histogram(dat,np.arange(minADU,maxADU,stepsize))[0]
    totArea=sum(ydata)
    #This is not the ideal or correct way to do the errors, but it doesn't change things much. Should be edited.
    yerr = np.sqrt(ydata)
    for j in range(len(yerr)):
        if yerr[j] == 0.0:
            yerr[j] = 1.0
    vals, errs = scipy.optimize.curve_fit(doubleGauss, 
                                          xdata, 
                                          ydata, 
                                          p0=[e1p*100,e1p/6,e1p,0,e1p*100],
                                          bounds=[[0,e1p/100,e1p/1.4,-.2*e1p,0],
                                                  [np.inf,.7*e1p,1.4*e1p,.2*e1p,np.inf]], 
                                          sigma=yerr, absolute_sigma=True) 
    bestNorm, bestWidth, bestADU, bestOffset, bestN = vals

    if verbose:
        print("ADU: " + str(bestADU) + "; Width: " + str(bestWidth)+ 
              "("+str(bestWidth/bestADU)+"); Offset: " + str(bestOffset))
    if not fast:
            # Plots the histogram and our fit, as a check
         predicted =[]
         for i in xdata:
              predicted.append(doubleGauss(i, bestNorm,bestWidth, bestADU,bestOffset,bestN))            
        
       # Plot the fit to the histogram
         minADU = -1.2*e1p
         maxADU = 3.2*e1p#3.2*e1p
         xdataplot = np.arange(minADU+stepsize/2., maxADU-stepsize/2.,stepsize)
         ydataplot = np.histogram(dat,np.arange(minADU,maxADU,stepsize))[0]
         plt.plot(xdata, predicted, 'm-')
         plt.plot(xdataplot,ydataplot)
         plt.yscale("log")
         plt.ylim([.05,200000])
         plt.xlim(-1.25*e1p,3.25*e1p)
         plt.title("Histogram and best fit to events, ID " + str(ID)+"; Quad " + str(quad))
         plt.xlabel("ADUs")
         plt.ylabel("Number of events")
         plt.show()
    return([bestADU, bestWidth, bestOffset])

#Finds dips in the data, assuming 0 peak is at 0 ADU
def roughGuessConv(dat):   
    histMin = 0
    histMax = 1000
    histStep = (histMax-histMin)/100
    #Guess values for the histogram--we want the 0 and 1 data
    firstHist = np.histogram(dat,np.arange(histMin, histMax, histStep))[0]
    xdata = np.arange(histMin+histStep/2., histMax-histStep/2.,histStep)

    searching = True
    i = 0
    while (searching and (i < (len(firstHist)))):
        if firstHist[i] < firstHist[0]/10:
            if i < 3: #We are too far zoomed out
                histMax /= 10
                histStep = (histMax-histMin)/100
                firstHist = np.histogram(dat,np.arange(histMin, histMax, histStep))[0]
                xdata = np.arange(histMin+histStep/2., histMax-histStep/2.,histStep)
                i = 0
            else: #We found where the first peak starts to drop off
                searching = False
        elif i == len(firstHist)-1: #we need to zoom out more
            histMax *= 10
            histStep = (histMax-histMin)/100
            firstHist = np.histogram(dat,np.arange(histMin, histMax, histStep))[0]
            xdata = np.arange(histMin+histStep/2., histMax-histStep/2.,histStep)

            i = 0
        else:
            i += 1
    peaks, _ = find_peaks(firstHist,distance=i*4)
    if len(peaks) == 1 or peaks[0] > 3:
        g = peaks[0]
    else:
        g = peaks[1]
    #Returns rough ADU value for where the 1e peak is
    return xdata[g]