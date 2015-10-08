#!/usr/bin/env python
# coding=utf-8

import sys,os
reload(sys)
sys.setdefaultencoding( "utf-8"  )

import numpy as np
from numpy import *
import jieba
import math
import jieba.analyse
jieba.load_userdict("userdict.txt")

def read_from_file(file_name):
    with open(file_name,"r") as fp:
        words = fp.read()
    return words
def stop_words(stop_word_file):
    words = read_from_file(stop_word_file)
    result = jieba.cut(words)
    new_words = []
    for r in result:
        new_words.append(r)
    return set(new_words)

def gen_sim(A,B):
    num = float(np.dot(A,B.T))
    denum = np.linalg.norm(A) * np.linalg.norm(B)
    if denum == 0:
        denum = 1
    cosn = num / denum
    sim = 0.5 + 0.5 * cosn
    return sim

def del_stop_words(words,stop_words_set):
    result = jieba.cut(words)
    new_words = []
    for r in result:
        if r not in stop_words_set:
            new_words.append(r)
            #print r.encode("utf-8"),
    #print len(new_words),len(set(new_words))
    return new_words

def tfidf(term,doc,word_dict,docset):
    tf = float(doc.count(term)) / (len(doc) + 0.001)
    idf = math.log(float(len(docset)) / word_dict[term])
    return tf * idf

def idf(term,word_dict,docset):
    idf = math.log(float(len(docset)) / word_dict[term])
    return idf

def word_in_docs(word_set,docs):
    word_dict = {}
    for word in word_set:
        #print word.encode("utf-8")
        word_dict[word] = len([doc for doc in docs if word in doc])
        #print word_dict[word],
    return word_dict

def get_all_vector(file_path,stop_words_set):
    names = [ os.path.join(file_path,f) for f in os.listdir(file_path) ]
    posts = [ open(name).read() for name in names ]
    docs = []
    word_set = set()
    for post in posts:
        doc = del_stop_words(post,stop_words_set)
        docs.append(doc)
        word_set |= set(doc)
        #print len(doc),len(word_set)

    word_set = list(word_set)
    docs_vsm = []
    #for word in word_set[:30]:
        #print word.encode("utf-8"),
    for doc in docs:
        temp_vector = []
        for word in word_set:
            temp_vector.append(doc.count(word) * 1.0)
        #print temp_vector[-30:-1]
        docs_vsm.append(temp_vector)

    docs_matrix = np.array(docs_vsm)
    #print docs_matrix.shape
    #print len(np.nonzero(docs_matrix[:,3])[0])
    column_sum = [ float(len(np.nonzero(docs_matrix[:,i])[0])) for i in range(docs_matrix.shape[1]) ]
    column_sum = np.array(column_sum)
    column_sum = docs_matrix.shape[0] / column_sum
    idf =  np.log(column_sum)
    idf =  np.diag(idf)
    #print idf.shape
    #row_sum    = [ docs_matrix[i].sum() for i in range(docs_matrix.shape[0]) ]
    #print idf
    #print column_sum
    for doc_v in docs_matrix:
        if doc_v.sum() == 0:
            doc_v = doc_v / 1
        else:
            doc_v = doc_v / (doc_v.sum())


    tfidf = np.dot(docs_matrix,idf)

    return names,tfidf

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids
    
def kMeans(dataSet, k, distMeas=gen_sim, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    counter = 0
    while counter <= 50:
        counter += 1
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; 
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; 
                    minIndex = j
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        #print centroids
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment



if __name__ == "__main__":
    stop_words = stop_words("./stop_words.txt")
    names,tfidf_mat = get_all_vector("./chinese/",stop_words)
    myCentroids,clustAssing = kMeans(tfidf_mat, 3, gen_sim, randCent)
    for label,name in zip(clustAssing[:,0],names):
        print label,name

