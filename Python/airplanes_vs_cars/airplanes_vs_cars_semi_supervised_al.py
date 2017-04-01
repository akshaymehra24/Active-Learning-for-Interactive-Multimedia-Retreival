# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 19:25:48 2017

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
D = np.zeros((len(X_file_names), len(X_file_names)))
for i in range(len(X_file_names)):
    D[i][i] = W2[i].sum()

print "Generating Graph Laplacian = D - W"
Graph_Laplacian = D - W2

Z = np.arange(0, len(W2))
Y_use = np.array(Y)

training_samples_al = []
precisions_al = []
recalls_al = []
accuracies_al = []
f1scores_al = []


labelled_points = 5
label_points = 5
test_points = 200 #first test point
z_i = 0
while labelled_points < 200:
    l = np.arange(0, labelled_points)
    u = np.arange(labelled_points, len(W2))
    
    W_ll = Graph_Laplacian[np.ix_(l, l)]
    W_lu = Graph_Laplacian[np.ix_(l, u)]
    W_ul = Graph_Laplacian[np.ix_(u, l)]
    W_uu = Graph_Laplacian[np.ix_(u, u)]
    f_l = Y_use[0:labelled_points]
    
    #print "Inverting W_uu"
    W_uu_inv = -1 * inv(W_uu)
    
    #print "Computing labels for unlabelled points"
    f_u = []
    f_u =  np.dot(np.dot(W_uu_inv, W_ul), f_l)
    
    GL_old = np.array(Graph_Laplacian)
    Y_old = np.array(Y_use)
    Z_old = np.array(Z)
    
    pos_items = np.argwhere(abs(f_u) >= 0.5).flatten()
    neg_items = np.argwhere(abs(f_u) < 0.5).flatten()
    #print len(pos_items), len(neg_items)
    f_unlabelled_points = np.zeros(len(f_u))
    f_unlabelled_points[np.ix_(pos_items)] = 1
    f_unlabelled_points[np.ix_(neg_items)] = 0
    
    Y_new = np.append(f_l, f_unlabelled_points)
    f_u = np.append(f_l, f_u)
    #print "CR only unlabelled points", classification_report(Y_use[labelled_points:], f_unlabelled_points)
    #print "CR all points", classification_report(Y_use, Y_new)
    #print "Accuracy ", Accuracy(Y_use, Y_new)
    #print "Accuracy on unlabelled points", Accuracy(Y_use[labelled_points:], f_unlabelled_points)
    
    training_samples_al.append(labelled_points)
    #accuracies_al.append(accuracy_score(Y_use, Y_new))
    #precisions_al.append(average_precision_score(Y_use, Y_new, average="weighted"))
    #recalls_al.append(recall_score(Y_use, Y_new, average="weighted"))
    #f1scores_al.append(f1_score(Y_use, Y_new, average="weighted"))
    
    accuracies_al.append(accuracy_score(Y_use[test_points:], Y_new[test_points:]))
    precisions_al.append(average_precision_score(Y_use[test_points:], Y_new[test_points:], average="weighted"))
    recalls_al.append(recall_score(Y_use[test_points:], Y_new[test_points:], average="weighted"))
    f1scores_al.append(f1_score(Y_use[test_points:], Y_new[test_points:], average="weighted"))
    
    most_u_p = []
    f_u_train = f_u[labelled_points:test_points]
    if len(f_u_train) == 0:
        break
    #print "len", len(f_u_train)
    for index, point in enumerate(f_u_train):
        #print index+labelled_points
        most_u_p.append([min(point, 1.0-point), index+labelled_points])
    most_u_p = sorted(most_u_p)
        
    #break
    most_uncertain_points = []
    most_uncertain_points = most_u_p[-label_points:]#np.sort(labelled_points + np.argsort(0.5 - f_u)[:label_points])
    #print most_uncertain_points
    for index, value in enumerate(most_uncertain_points):
        #print labelled_points, point
        point = int(value[1])
        z_i += 1
        #print point
        #print z_i, " ", point, len(f_u_train), labelled_points
        #swap rows
        Graph_Laplacian[labelled_points] = GL_old[point]
        Graph_Laplacian[point] = GL_old[labelled_points]
        #swap columns
        Graph_Laplacian[:, [point,labelled_points]] = Graph_Laplacian[:, [labelled_points,point]]
        
        Y_use[[labelled_points,point]] = Y_use[[point,labelled_points]]
        
        Z[[labelled_points,point]] = Z[[point,labelled_points]]
        labelled_points += 1
    

plt.figure()
plt.title("Accuracy Report")  
plt.xlabel("Training examples")
plt.ylabel("Accuracy Score")
plt.plot(training_samples, accuracies, label="Accuracy")
plt.plot(training_samples_al, accuracies_al, label = "Accuracy AL")
plt.legend(loc="best")

plt.figure()
plt.title("Precision Report")  
plt.xlabel("Training examples")
plt.ylabel("Precision Score")
plt.plot(training_samples, precisions, label="Precision")
plt.plot(training_samples_al, precisions_al, label = "Precision AL")
plt.legend(loc="best")

plt.figure()
plt.title("Recall Report")  
plt.xlabel("Training examples")
plt.ylabel("Recall Score")
plt.plot(training_samples, recalls, label="Recall")
plt.plot(training_samples_al, recalls_al, label = "Recall AL")
plt.legend(loc="best")

plt.figure()
plt.title("F1 Scores Report")  
plt.xlabel("Training examples")
plt.ylabel("F1 Scores Score")
plt.plot(training_samples, f1scores, label="F1 Scores")
plt.plot(training_samples_al, f1scores_al, label = "F1 Scores AL")
plt.legend(loc="best") 
