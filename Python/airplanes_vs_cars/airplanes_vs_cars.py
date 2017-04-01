# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 00:13:33 2017

@author: Akshay
"""

import cv2
import numpy as np
import os
from scipy.cluster.vq import *
# Importing the library which classifies set of observations into clusters
from sklearn.preprocessing import StandardScaler
# Importing the library that supports centering and scaling vectors
from sklearn.svm import LinearSVC

pFiles = os.listdir("E:\Online Learning\Datasets\Cars_vs_Airplanes\Airplanes")
nFiles = os.listdir("E:\Online Learning\Datasets\Cars_vs_Airplanes\Cars")

print "reading files"
X_file_names = []
Y = []
for i in range(len(pFiles)):
    f=pFiles[i]
    X_file_names.append("airplane_"+f)
    Y.append(1)

for i in range(len(nFiles)):
    f=nFiles[i]
    X_file_names.append("cars_"+f)
    Y.append(-1)
    
print "Create feature extraction and keypoint detector objects"
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")

print "List where all the descriptors are stored"
des_list = []

for image in pFiles:
    im = cv2.imread("E:\Online Learning\Datasets\Cars_vs_Airplanes\Airplanes\\" + image)
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    des_list.append(("airplane_" + image, des))  
    
for image in nFiles:
    im = cv2.imread("E:\Online Learning\Datasets\Cars_vs_Airplanes\Cars\\" + image)
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    des_list.append(("cars_" + image, des))  

print "Stack all the descriptors vertically in a numpy array"
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  

print "Perform k-means clustering"
k = 200
voc, variance = kmeans(descriptors, k, 1) 

print "Calculating the histogram of features"
im_features = np.zeros((len(X_file_names), k), "float32")
for i in xrange(len(X_file_names)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1 # Caluculating the histogram of features
       
print "Perform Tf-Idf vectorization"
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
# Calculating the number of occurrences
idf = np.array(np.log((1.0*len(X_file_names)+1) / (1.0*nbr_occurences + 1)), 'float32')
# Giving weight to one that occurs more frequently

print "Scaling the words"
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)  # Scaling the visual words for better Prediction


print "Randomly shuffle"
Y = np.asarray(Y)
X_file_names = np.asarray(X_file_names)
index = np.arange(im_features.shape[0])
np.random.shuffle(index)
im_features = im_features[index,:]
Y = Y[index]
X_file_names = X_file_names[index]                            



                              