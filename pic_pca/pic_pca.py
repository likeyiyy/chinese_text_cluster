#!/usr/bin/env python
# coding=utf-8

from PIL import Image
from numpy import *
import matplotlib.pyplot as plt
import sys

def LoadPicture(name):
    img = Image.open(name)
    img = img.convert("L")
    #print img.mode,imgMat.shape
    #plt.gray()
    #plt.imshow(imgMat)
    #plt.show()
    imgMat = matrix(img)
    #print imgMat
    return imgMat

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def pca_analyize(dataMat,threshold):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[::-1]#reverse
    sortedEigVals = eigVals[eigValInd]
    total = sum(sortedEigVals)
    varPercentage = sortedEigVals/total*100
    count = 0
    index = 0
    for per in varPercentage:
        index += 1
        count += per 
        if count > threshold:
            break
    return varPercentage,index

def pca_viewer(varPercentage):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, 51), varPercentage[:50], marker='^')
    plt.xlabel('Principal Component Number')
    plt.ylabel('Percentage of Variance')
    plt.show()

def del_imaginary(reconMat):
    m,n = reconMat.shape[:2]
    realMat = matrix(zeros([m,n]))
    for i in range(m):
        for j in range(n):
            realMat[i,j] = reconMat[i,j].real
    return realMat

if __name__ == "__main__":
    imgMat = LoadPicture(sys.argv[1])
    plt.subplot(2,3,1)
    threshold = [20,50,80,90,95,99]
    for i in range(len(threshold)):
        varPercentage,index = pca_analyize(imgMat,threshold[i])
        print index
        #pca_viewer(varPercentage)
        lowImgMat, reconMat = pca(imgMat,index)
        realMat = del_imaginary(reconMat)
        plt.gray()
        plt.subplot(2,3,i+1)
        plt.imshow(realMat)
    plt.show()


