#!/usr/bin/env python
# coding=utf-8

from PIL import Image
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation 

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

def SVD_analyize(E,threshold):
    E_square = E ** 2
    Total = sum(E_square)
    Thre  = Total * threshold / 100
    for i in range(len(E_square)):
        if sum(E_square[:i]) > Thre:
            break
    return i

def svd_pic(imgMat,index):
    U,E,VT = linalg.svd(imgMat)
    #index = SVD_analyize(E,threshold)
    #print index
    SigRecon = mat(zeros((index,index)))
    for k in range(index):
        SigRecon[k,k] = E[k]
    reconMat = U[:,:index] * SigRecon * VT[:index,:]
    return reconMat


fig = plt.figure()
axes1 = fig.add_subplot(111) 
imgArray = LoadPicture(sys.argv[1])
reconMat = svd_pic(imgArray,2)
aIMG = plt.imshow(reconMat)

def update(data):
    
    



if __name__ == "__main__":
    pass
    '''
    imgArray = LoadPicture(sys.argv[1])
    plt.gray()
    plt.subplot(2,4,1)
    indexs = [2,4,8,16,10,20,40,60]
    for i in range(8):
        reconMat = svd_pic(imgArray,int(indexs[i]))
        plt.subplot(2,4,i+1)
        a = plt.imshow(reconMat)
        print a
    plt.show()
    '''



