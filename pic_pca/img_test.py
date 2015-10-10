#!/usr/bin/env python
# coding=utf-8

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

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
            n0 = imgArray[i,j] & 0xff
            n1 = (imgArray[i,j] >> 8) & 0xff
            n2 = (imgArray[i,j] >> 16) & 0xff
            a[i,j] = np.array([n0,n1,n2])
    return a


a = LoadImage(sys.argv[1])


imgMat1 = GeneratorImage(a)
imgMat2 = GeneratorImage(a.T)
plt.subplot(1,2,1)
plt.imshow(imgMat1)
plt.subplot(1,2,2)
plt.imshow(imgMat2)
plt.show()

