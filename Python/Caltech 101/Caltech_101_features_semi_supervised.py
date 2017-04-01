# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 19:00:42 2017

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
from sklearn.externals import joblib
from skimage.measure import compare_ssim as compare_ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import scipy.special as sp
from PIL import Image as Img
import scipy.spatial.distance as dist
from sklearn.metrics.pairwise import euclidean_distances
import random

#Categories = os.listdir("E:\\Online Learning\\Datasets\\101_ObjectCategories\\")
'''
print "reading files"
X_file_names = []
Y = []
for i in range(len(Categories)):
    Category=Categories[i]
    Category_Images=os.listdir("E:\\Online Learning\\Datasets\\101_ObjectCategories\\"+Category)
    for j in range(len(Category_Images)):
        f=Category_Images[j]
        X_file_names.append(Category+"_"+f)
        Y.append(i)
#joblib.dump(X_file_names, "X_file_names_Caltech101.pkl", compress=3)
#joblib.dump(Y, "Y_Caltech101.pkl", compress=3)
'''
#X_file_names = joblib.load("E:\\Online Learning\\activelearning\\Python\\Caltech 101\\X_file_names_Caltech101.pkl")
#Y = joblib.load("E:\\Online Learning\\activelearning\\Python\\Caltech 101\\Y_Caltech101.pkl")
'''
print "Reading Images"
Images = []
for i in range(len(Categories)):
    Category=Categories[i]
    Category_Images=os.listdir("E:\\Online Learning\\Datasets\\101_ObjectCategories\\"+Category)
    print Category
    for j in range(len(Category_Images)):
        f=Category_Images[j]
        im = Img.open("E:\\Online Learning\\Datasets\\101_ObjectCategories\\" + Category +"\\" + f)
        im = im.resize([200, 200])
        pix = im.load()
        Images.append([im, pix]) 

print "Flattening Images"

flat_Images = []
for index, (image, pix) in enumerate(Images):
    height, width = image.size
    one_im = np.zeros([height*width, 3])
    count = 0
    for i in range(height):
            for j in range(width):
                one_im[count] = np.array(pix[i,j])
                count += 1
    #one_im = np.ndarray.flatten(one_im)
    flat_Images.append(one_im)


flat_Images = np.array(flat_Images)
flat_Images = flat_Images.reshape(-1, 120000)
#joblib.dump(flat_Images, "flat_Images_Caltech101.pkl", compress=3)


flat_Images = joblib.load("E:\\Online Learning\\activelearning\\Python\\Caltech 101\\flat_Images_Caltech101_array.pkl")
flat_Images = flat_Images.reshape(-1, 120000)

print "Computing the Weights using pixel wise Distance"    
#TODO: Choose scaling factor                       
#W = np.zeros((len(flat_Images), len(flat_Images)))
#W1 = np.zeros((len(Images), len(Images)))

#W[:5000, :5000] = euclidean_distances(flat_Images[:5000, :], flat_Images[:5000, :])
#W[:5000, 5000:] = euclidean_distances(flat_Images[:5000, :], flat_Images[5000:, :])
#W[5000:, :5000] = euclidean_distances(flat_Images[5000:, :], flat_Images[:5000, :])
#W[5000:, 5000:] = euclidean_distances(flat_Images[5000:, :], flat_Images[5000:, :])
#joblib.dump(W, "W2.pkl", compress=3)

print "Finding 100 nearest neighbours"
W2 = np.zeros((len(Images), len(Images)))  
for i, arr in enumerate(W1):
    hundred_nearest = np.argsort(arr)[:100]
    #print ten_nearest
    for j, ind in enumerate(hundred_nearest):
        W2[i][ind] = W1[i][ind]
        W2[ind][i] = W1[ind][i]



joblib.dump(im_features, "im_fea.pkl", compress=3)  
joblib.dump(Y, "im_feat_Y.pkl", compress=3)  
'''                            

#W = joblib.load("E:\\Online Learning\\activelearning\\Python\\Caltech 101\\W1.pkl")
#W4 = joblib.load("E:\\Online Learning\\activelearning\\Python\\Caltech 101\\W4.pkl")
#W = W+W4
#joblib.dump(W, "W_Images.pkl", compress=3)  

print "Randomly shuffle"

W = joblib.load("E:\\Online Learning\\activelearning\\Python\\Caltech 101\\W_Images.pkl")
W_old = np.array(W)
Y = np.asarray(Y)
X_file_names = np.asarray(X_file_names)

indices = np.arange(W.shape[0])
np.random.shuffle(indices)

for index, point in enumerate(indices):
    #swap rows
    W[index] = W_old[point]
    W[point] = W_old[index]
    #swap columns
    W[:, [point, index]] = W[:, [index, point]]
    X_file_names[[index, point]] = X_file_names[[point, index]]
    Y[[index, point]] = Y[[point, index]]

for i in range(len(W)):
    if W[i][i] != 0:
        print "Error"
