# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 22:14:46 2017

@author: Akshay
"""

from skimage.measure import compare_ssim as compare_ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import scipy.special as sp
from PIL import Image as Img

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
    Y.append(0)

print "Randomly shuffle"
Y = np.asarray(Y)
X_file_names = np.asarray(X_file_names)
index = np.arange(len(X_file_names))
np.random.shuffle(index)
Y = Y[index]
X_file_names = X_file_names[index]  

print "Reading Images"
Images = []
for index, image in enumerate(X_file_names):
    #print index, image, Y[index]
    if Y[index] == 1:
        im = Img.open("E:\Online Learning\Datasets\Cars_vs_Airplanes\Airplanes\\" + image[9:])
        im = im.resize([200, 200])
        pix = im.load()
        # convert the images to grayscale
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        Images.append([im, pix]) 
        
    else:
        im = Img.open("E:\Online Learning\Datasets\Cars_vs_Airplanes\Cars\\" + image[5:])
        im = im.resize([200, 200])
        pix = im.load()
        # convert the images to grayscale
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        Images.append([im, pix]) 


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
    

print "Computing the Weights using pixel wise Distance"    
#TODO: Choose scaling factor                       
W = np.zeros((len(Images), len(Images)))
W1 = np.zeros((len(Images), len(Images)))

for index, pix in enumerate(flat_Images):
    for index1, pix1 in enumerate(flat_Images):
        diff = pix - pix1
        W_sum = (diff**2).sum(axis = 1)
        W_sum = np.sqrt(W_sum.sum())
        W[index, index1] = np.exp(-1 * W_sum / 250000.)
        W1[index, index1] = W_sum.sum()

W2 = np.zeros((len(Images), len(Images)))  
for i, arr in enumerate(W1):
    ten_nearest = np.argsort(arr)[:10]
    #print ten_nearest
    for j, ind in enumerate(ten_nearest):
        W2[i][ind] = W1[i][ind]
        W2[ind][i] = W1[ind][i]


'''print "Computing the Weights using Structural Similarity"    
#TODO: Choose scaling factor                       
W = np.zeros((len(Images), len(Images)))
for index, (image, im) in enumerate(Images):
    for index1, (image1, im1) in enumerate(Images):
        if image == image1:
            W[index, index1] = 0.
        else:
            W[index, index1] = np.exp(10*(1 - abs(compare_ssim(im, im1))))
'''
'''
print "Computing the Weights using Manhattan Distance"    
#TODO: Choose scaling factor                       
W = np.zeros((len(Images), len(Images)))
for index, (image, im) in enumerate(Images):
    for index1, (image1, im1) in enumerate(Images):
        #if image == image1:
        #    W[index, index1] = 0.
        #else:
        diff = im1 - im
        W[index, index1] = np.sqrt(sum(diff**2))#np.exp(-1. * sum(diff**2) / 1000.**2)
'''