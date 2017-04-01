# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 04:26:50 2017

@author: Akshay
"""
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#from airplanes_vs_cars_semi_supervised import *

def Accuracy(Y, Y_hat):
    correct = (Y_hat == Y)
    return float(sum(correct)) / len(correct)

print "Generating the Diagonal Matrix"
D = np.zeros([X.shape[0], X.shape[0]])
for i in range(X.shape[0]):
    D[i][i] = W[i].sum()

print "Generating Graph Laplacian = D - W"
Graph_Laplacian = D - W

training_samples = []
precisions = []
recalls = []
accuracies = []
f1scores = []


labelled_points = 5
test_points = 1000 #first test point

while labelled_points <= 1000:
    print "labelled points", labelled_points
    
    l = np.arange(0, labelled_points)
    u = np.arange(labelled_points, len(W))
    
    W_ll = Graph_Laplacian[np.ix_(l, l)]
    W_lu = Graph_Laplacian[np.ix_(l, u)]
    W_ul = Graph_Laplacian[np.ix_(u, l)]
    W_uu = Graph_Laplacian[np.ix_(u, u)]
    f_l = Y[0:labelled_points]
    
    #print "Inverting W_uu"
    W_uu_inv = -1 * inv(W_uu)
    
    #print "Computing labels for unlabelled points"
    f_u =  np.dot(np.dot(W_uu_inv, W_ul), f_l)
    
    pos_items = np.argwhere(abs(f_u) >= 0.5).flatten()
    neg_items = np.argwhere(abs(f_u) < 0.5).flatten()
    #print len(pos_items), len(neg_items)
    f_u[np.ix_(pos_items)] = 1
    f_u[np.ix_(neg_items)] = 0
    
    Y_new = np.append(f_l, f_u)
    #print "CR only unlabelled points", classification_report(Y[labelled_points:], f_u)
    #print "CR all points", classification_report(Y, Y_new)
    #print "Accuracy on unlabelled points", Accuracy(Y[labelled_points:], f_u)
    #print "Accuracy ", Accuracy(Y, Y_new)
    
    training_samples.append(labelled_points)
    '''accuracies.append(accuracy_score(Y, Y_new))
    precisions.append(average_precision_score(Y, Y_new, average="weighted"))
    recalls.append(recall_score(Y, Y_new, average="weighted"))
    f1scores.append(f1_score(Y, Y_new, average="weighted"))
    '''
    accuracies.append(accuracy_score(Y[test_points:], Y_new[test_points:]))
    precisions.append(average_precision_score(Y[test_points:], Y_new[test_points:], average="weighted"))
    recalls.append(recall_score(Y[test_points:], Y_new[test_points:], average="weighted"))
    f1scores.append(f1_score(Y[test_points:], Y_new[test_points:], average="weighted"))
    
    labelled_points += 1
    
plt.title("Classification Report")  
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.plot(training_samples, accuracies, label = "Accuracy")
plt.plot(training_samples, precisions, label = "Precision")
plt.plot(training_samples, recalls, label = "Recall")
plt.plot(training_samples, f1scores, label = "F1 Scores")
plt.legend(loc="best")