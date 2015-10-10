#!/usr/bin/env python
# coding=utf-8

from PIL import Image
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import sys

def LoadPicture(name):
    img = Image.open(name)
    m,n = img.size
    img = img.convert("L")
    realMat = array(img)
    #print img.mode,imgMat.shape
    #plt.gray()
    #plt.imshow(imgMat)
    #plt.show()
    imgMat = realMat
    return imgMat


def LoadImage(file):
    img = Image.open(file)
    m,n = img.size
    imgArray = np.array(img)
    a = np.zeros([n,m],np.uint32)
    for i in range(n):
        for j in range(m):
            temp = np.uint32(imgArray[i,j][2])
            temp = (temp << 8) | imgArray[i,j][1]
            temp = (temp << 8) | imgArray[i,j][0]
            a[i,j] = temp
    return a



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
    count = 0.0
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
    realMat = matrix(zeros([m,n]),np.uint32)
    for i in range(m):
        for j in range(n):
            realMat[i,j] = np.uint32(reconMat[i,j].real)
    return realMat

def GeneratorImage(imgArray):
    m,n = imgArray.shape
    a = np.zeros([m,n,3],np.uint8)
    for i in range(m):
        for j in range(n):
            n0 = (imgArray[i,j]) & 0xff
            n1 = ((imgArray[i,j]) >> 8) & 0xff
            n2 = ((imgArray[i,j]) >> 16) & 0xff
            a[i,j] = np.array([n0,n1,n2])
    return a
    img = Image.fromarray(a,"RGB")
    return img

if __name__ == "__main__":
    imgMat = LoadImage(sys.argv[1])
    plt.subplot(2,3,1)
    threshold = [1,30,50,100,164,165]
    for i in range(len(threshold)):
        #varPercentage,index = pca_analyize(imgMat,threshold[i])
        #print index
        #pca_viewer(varPercentage)
        lowImgMat, reconMat = pca(imgMat,threshold[i])
        realMat = del_imaginary(reconMat)
        img = GeneratorImage(realMat)
        plt.subplot(2,3,i+1)
        plt.gray()
        plt.imshow(img)
    plt.show()


