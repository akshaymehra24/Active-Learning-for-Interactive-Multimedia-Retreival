# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 04:42:04 2017

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

Z = np.arange(0, len(W))
Y_use = np.array(Y)

training_samples_al_risk = []
precisions_al_risk = []
recalls_al_risk = []
accuracies_al_risk = []
f1scores_al_risk = []


labelled_points = 5
label_points = 1
test_points = 1000 #first test point
z_i = 0
while labelled_points <= 1000:
    print "labelled points", labelled_points
    
    l = np.arange(0, labelled_points)
    u = np.arange(labelled_points, len(W))
    
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
    
    f_u_new = np.append(f_l, f_u)
                        
    Y_new = np.append(f_l, f_unlabelled_points)
    #print "CR only unlabelled points", classification_report(Y_use[labelled_points:], f_unlabelled_points)
    #print "CR all points", classification_report(Y_use, Y_new)
    #print "Accuracy ", Accuracy(Y_use, Y_new)
    #print "Accuracy on unlabelled points", Accuracy(Y_use[labelled_points:], f_unlabelled_points)
    
    training_samples_al_risk.append(labelled_points)
    #accuracies_al.append(accuracy_score(Y_use, Y_new))
    #precisions_al.append(average_precision_score(Y_use, Y_new, average="weighted"))
    #recalls_al.append(recall_score(Y_use, Y_new, average="weighted"))
    #f1scores_al.append(f1_score(Y_use, Y_new, average="weighted"))
    
    accuracies_al_risk.append(accuracy_score(Y_use[test_points:], Y_new[test_points:]))
    precisions_al_risk.append(average_precision_score(Y_use[test_points:], Y_new[test_points:], average="weighted"))
    recalls_al_risk.append(recall_score(Y_use[test_points:], Y_new[test_points:], average="weighted"))
    f1scores_al_risk.append(f1_score(Y_use[test_points:], Y_new[test_points:], average="weighted"))
    
    
    risk_if_include_point_i = 1000000000
    min_risk_point = 0
    f_u_train = f_u_new[labelled_points:test_points]
    for index, point in enumerate(f_u_train):
        risk_after_inclusion = 0
        f_u_new_0 = f_u + (0 - point) *  W_uu_inv[:, index] / W_uu_inv[index, index] 
        f_u_new_1 = f_u + (1 - point) *  W_uu_inv[:, index] / W_uu_inv[index, index] 
        risk_f_u_new_0 = 0
        risk_f_u_new_1 = 0
        for ind in range(len(f_u_train)):
            risk_f_u_new_0 += min(f_u_new_0[ind], 1 - f_u_new_0[ind])
            risk_f_u_new_1 += min(f_u_new_1[ind], 1 - f_u_new_1[ind])
        risk_after_inclusion = (1 - point) * risk_f_u_new_0 + point * risk_f_u_new_1
        #print  index + labelled_points, risk_after_inclusion                       
        if risk_after_inclusion < risk_if_include_point_i:
            risk_if_include_point_i = risk_after_inclusion
            min_risk_point = index + labelled_points
     
    point = int(min_risk_point)
    z_i += 1
    #print "including point ", point, min_risk_point, labelled_points, z_i, len(f_u_train)
    
    #print point
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
plt.plot(training_samples_al_risk, accuracies_al_risk, label = "Accuracy AL risk")
plt.legend(loc="best")

plt.figure()
plt.title("Precision Report")  
plt.xlabel("Training examples")
plt.ylabel("Precision Score")
plt.plot(training_samples, precisions, label="Precision")
plt.plot(training_samples_al, precisions_al, label = "Precision AL")
plt.plot(training_samples_al_risk, precisions_al_risk, label = "Precision AL risk")
plt.legend(loc="best")

plt.figure()
plt.title("Recall Report")  
plt.xlabel("Training examples")
plt.ylabel("Recall Score")
plt.plot(training_samples, recalls, label="Recall")
plt.plot(training_samples_al, recalls_al, label = "Recall AL")
plt.plot(training_samples_al_risk, recalls_al_risk, label = "Recall AL risk")
plt.legend(loc="best")

plt.figure()
plt.title("F1 Scores Report")  
plt.xlabel("Training examples")
plt.ylabel("F1 Scores Score")
plt.plot(training_samples, f1scores, label="F1 Scores")
plt.plot(training_samples_al, f1scores_al, label = "F1 Scores AL")
plt.plot(training_samples_al_risk, f1scores_al_risk, label = "F1 Scores AL risk")
plt.legend(loc="best") 
