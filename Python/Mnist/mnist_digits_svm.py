# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 00:00:23 2017

@author: Akshay
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
import os as os
###########################
# LOADING TRAINING DATA
###########################
os.chdir('E:\Online Learning\Datasets')
trainfile = open('mnist_train.csv')
for line in trainfile:
    header = line.rstrip().split(',')
    break

y = []
X = []

for line in trainfile:
    splitted = line.rstrip().split(',')
    label = int(splitted[0])
    A_features = [float(item) for item in splitted[1:784]]
    y.append(label)
    X.append(A_features)
trainfile.close()

y = np.array(y)
X = np.array(X)

rng = np.random.RandomState(0)
indices = np.arange(len(X))
rng.shuffle(indices)

X = X[indices[:42000]]
y = y[indices[:42000]]

n_total_samples = len(y)
n_labeled_points = 5000

unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]
f = plt.figure()
y_train=y[1:n_labeled_points]
X_train=X[1:n_labeled_points]
iterations = 5
for i in range(0,iterations):        
    #model=svm.SVC(kernel='linear', C=1.0, probability=True, decision_function_shape='ovr').fit(X_train, y_train)  
    model=svm.LinearSVC(multi_class='crammer_singer').fit(X_train, y_train) 
    predicted_labels = model.predict(X)   
    #if i+1 == iterations:
        #print(i)
    print("%s\n" % (classification_report(y, predicted_labels)))
    # compute the entropies of transduced label distributions
    df_matrix=[]
    df_matrix=model.decision_function(X)
    df_matrix_max=[]
    image_number=[]
    for j in range(0, len(df_matrix)) :
        #print(df_matrix[j][np.argmax(np.abs(df_matrix[j]))]) 
        m1=0
        m2=0
        if predicted_labels[j] >= 0:
            image_number.append(j)            
            for k in range(0, len(df_matrix[j])):
                ele=np.abs(df_matrix[j][k])
                if ele>m1:
                    m2=m1
                    m1=ele
            df_matrix_max.append(m1-m2)
    # select five digit examples that the classifier is most uncertain about
    uncertainty_index_most = []
    index = 0
    co = 0
    while co < 1000:
        element=np.argsort(df_matrix_max)[index]
        #print(element)
        if np.searchsorted(unlabeled_indices, element) != len(df_matrix_max):
            uncertainty_index_most = np.append(uncertainty_index_most, int(element))   
            co += 1
            index += 1
        else:
            index += 1
    #uncertainty_index_most = np.argsort(df_matrix_max)[:5]
    uncertainty_index_least = np.argsort(df_matrix_max)[-100:]
    uncertainty_index_least = uncertainty_index_least[::-1] #reversing the array
    uncertainty_index = np.append(uncertainty_index_most,uncertainty_index_least)
    # keep track of indices that we get labels for
    delete_indices = np.array([])
    
    for index, image_index in enumerate(uncertainty_index):
        #print(image_index)
        image_index=int(image_index)
        #image = images[image_number[image_index]]        
        #sub = f.add_subplot(iterations, 10, index + 1 + (10 * i))
        #sub.imshow(image, cmap=plt.cm.gray_r)
        #sub.set_title('image: %i\npredict: %i\ntrue: %i' % (image_number[image_index],predicted_labels[image_number[image_index]], y[image_number[image_index]]), size=10)
        #sub.axis('off')

        # labeling 5 points, remote from labeled set
        if index < 1000:
            #print(image_index)
            unlabeled_indices = np.delete(unlabeled_indices, np.searchsorted(unlabeled_indices,image_number[image_index]))
            X_train=np.vstack([X_train,X[image_number[image_index]]])
            y_train=np.append(y_train, np.array(y[image_number[image_index]]))

#f.suptitle("Active learning with SVM.\nRows show 5 most uncertain labels to learn with the next model.")
#plt.subplots_adjust(0.5, 0.5, 1.8, 1.8, 3.2, 3.2)
#plt.show()
###########################
# READING TEST DATA
###########################

testfile = open('mnist_test.csv')
#ignore the test header
for line in testfile:
    break

X_test = []
for line in testfile:
    splitted = line.rstrip().split(',')
    A_features = [float(item) for item in splitted[0:783]]
    X_test.append(A_features)
testfile.close()

X_test = np.array(X_test)
# compute decision function
p_test = model.predict(X_test)
###########################
# WRITING SUBMISSION FILE
###########################
predfile = open('predictions.csv','w+')
predfile.write('ImageId,Label\n')
i=1;
for line in p_test:
    x=str(i)+','+str(line)
    predfile.write(x)
    predfile.write('\n')
    i=i+1

predfile.close()