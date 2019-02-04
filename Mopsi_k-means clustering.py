#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 21:30:52 2018

@author: ChiaYen
"""
import numpy as np
import math
import scipy.cluster as sc
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt

#k-means clustering
def standard(data):
    standardData = data.copy()
    
    rows = data.shape[0]
    cols = data.shape[1]

    for j in range(cols):
        sigma = np.std(data[:,j])
        mu = np.mean(data[:,j])

        for i in range(rows):
            standardData[i,j] = (data[i,j] - mu)/sigma

    return standardData



def dist(p1, p2): 
    sumTotal = 0
    for c in range(len(p1)):
        sumTotal = sumTotal + pow((p1[c] - p2[c]),2)
    return math.sqrt(sumTotal)


dataRaw = [];
DataFile = open("MopsiLocations2012-Joensuu.txt", "r")
while True:
    theline = DataFile.readline()
    if len(theline) == 0:
         break  
    readData = theline.split(" ")
    for pos in range(len(readData)):
        readData[pos] = float(readData[pos]);
    dataRaw.append(readData)

DataFile.close()

data = np.array(dataRaw)



standardisedData = standard(data)

centroids, distortion = sc.vq.kmeans(standardisedData, 2) #if we assume 2 clusters


plt.figure(figsize=(6,4))

plt.plot(standardisedData[:,0],standardisedData[:,1],'.')

plt.plot(centroids[0,0],centroids[0,1],'rx')

plt.plot(centroids[1,0],centroids[1,1],'gx')

plt.savefig("kmeans.pdf")

plt.close()


group1 = np.array([])

group2 = np.array([])

for d in standardisedData:
    if (dist(d, centroids[0,:]) < dist(d, centroids[1,:])):
        if (len(group1) == 0):
            group1 = d
        else:
            group1 = np.vstack((group1,d))
    else:
        if (len(group2) == 0):
            group2 = d
        else:
            group2 = np.vstack((group2,d))

plt.figure(figsize=(6,4))

plt.plot(group1[:,0],group1[:,1],'r.')
plt.plot(group2[:,0],group2[:,1],'g.')

plt.plot(centroids[0,0],centroids[0,1],'rx')
plt.plot(centroids[1,0],centroids[1,1],'gx')

plt.savefig("kmeansClassified.pdf")

plt.close()


#reference: lab note