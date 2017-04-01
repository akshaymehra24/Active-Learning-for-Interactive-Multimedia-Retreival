# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 02:54:32 2017

@author: Akshay
"""
#Tutorial : https://ianlondon.github.io/blog/how-to-sift-opencv/
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os as os

os.chdir('E:\Online Learning\Datasets')
print 'OpenCV Version (should be 3.1.0, with nonfree packages installed, for this tutorial):'
print cv2.__version__

# I cropped out each stereo image into its own file.
# You'll have to download the images to run this for yourself
octo_front = cv2.imread('2.jpg')
octo_offset = cv2.imread('3.jpg')

def show_rgb_img(img):
    """Convenience function to display a typical color image"""
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

octo_front_gray = to_gray(octo_front)
octo_offset_gray = to_gray(octo_offset)

plt.imshow(octo_front_gray, cmap='gray');
          
def gen_sift_features(gray_img):
    sift = cv2.SIFT()
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))

# generate SIFT keypoints and descriptors
octo_front_kp, octo_front_desc = gen_sift_features(octo_front_gray)
octo_offset_kp, octo_offset_desc = gen_sift_features(octo_offset_gray)

print 'Here are what our SIFT features look like for the front-view octopus image:'
show_sift_features(octo_front_gray, octo_front, octo_front_kp);
                  
# create a BFMatcher object which will match up the SIFT features
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(octo_front_desc, octo_offset_desc)

# Sort the matches in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# draw the top N matches
N_MATCHES = 100

#match_img = cv2.drawMatches(octo_front, octo_front_kp, octo_offset, octo_offset_kp, matches[:N_MATCHES], octo_offset.copy(), flags=0)

#plt.figure(figsize=(12,6))
#plt.imshow(match_img);
print octo_front_desc[0]

print len(octo_front_kp), 'keypoints in the list'
print octo_front_kp[0]

def explain_keypoint(kp):
    print 'angle\n', kp.angle
    print '\nclass_id\n', kp.class_id
    print '\noctave (image scale where feature is strongest)\n', kp.octave
    print '\npt (x,y)\n', kp.pt
    print '\nresponse\n', kp.response
    print '\nsize\n', kp.size

print 'this is an example of a single SIFT keypoint:\n* * *'
explain_keypoint(octo_front_kp[0])

print 'SIFT descriptors are vectors of shape', octo_front_desc[0].shape
print 'they look like this:', octo_front_desc[0]

# visualized another way:
plt.imshow(octo_front_desc[0].reshape(16,8), interpolation='none');