# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 22:25:01 2017

@author: Akshay
"""

from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

training_samples_al = []
precisions_al = []
recalls_al = []
accuracies_al = []
f1scores_al = []

batch_len = 10
X_train = im_features[0:batch_len,:]
Y_train = Y[0:batch_len]

X_test = im_features[200:,]
Y_test = Y[200:]

images_in_train = []
for i in range(batch_len):
    images_in_train.append(i)
    
while len(X_train) < 200:

    model=svm.SVC(kernel='linear', C=1.0, probability=True, decision_function_shape='ovr').fit(X_train, Y_train)    
    predicted_labels = model.predict(im_features)   
    predicted_labels_test_set = model.predict(X_test)
    report = classification_report(Y_test, predicted_labels_test_set)
    #print("%s\n" % (classification_report(Y, predicted_labels)))
    
    training_samples_al.append(len(X_train))
    #accuracies_al.append(accuracy_score(Y, predicted_labels))
    #precisions_al.append(average_precision_score(Y, predicted_labels, average="weighted"))
    #recalls_al.append(recall_score(Y, predicted_labels, average="weighted"))
    #f1scores_al.append(f1_score(Y, predicted_labels, average="weighted"))
    
    accuracies_al.append(accuracy_score(Y_test, predicted_labels_test_set))
    precisions_al.append(average_precision_score(Y_test, predicted_labels_test_set, average="weighted"))
    recalls_al.append(recall_score(Y_test, predicted_labels_test_set, average="weighted"))
    f1scores_al.append(f1_score(Y_test, predicted_labels_test_set, average="weighted"))
    
    # compute the entropies of transduced label distributions
    df_matrix=[]
    df_matrix=model.decision_function(im_features)
    df_matrix_max=[]
    image_number=[]
    for j in range(0, len(df_matrix)) :
        #print(df_matrix[j][np.argmax(np.abs(df_matrix[j]))]) 
        m1=0
        image_number.append(j)            
        for k in range(0,1):#, len(df_matrix[j])):
            ele=np.abs(df_matrix[j])#[k])
            #print ele
            if ele>m1:
                m1=ele
        df_matrix_max.append([m1,j])
    df_matrix_max.sort()
    # select five digit examples that the classifier is most uncertain about
    uncertainty_index_most = [] 
    count = 0
    for index, j in enumerate(df_matrix_max):
        image_num = j[1]
        if images_in_train.count(image_num) == 0:
            count +=1
            uncertainty_index_most.append(j)
            if count == 5:
                break
        
    '''uncertainty_index_most = np.argsort(df_matrix_max)[:5]
    uncertainty_index_least = np.argsort(df_matrix_max)[-5:]
    uncertainty_index_least = uncertainty_index_least[::-1] #reversing the array
    uncertainty_index = np.append(uncertainty_index_most,uncertainty_index_least)'''
    
    for index, element in enumerate(uncertainty_index_most):
        #print(image_index)
        image_index=int(element[1])
        #image = images[image_number[image_index]]        
        #sub = f.add_subplot(iterations, 10, index + 1 + (10 * i))
        #sub.imshow(image, cmap=plt.cm.gray_r)
        #sub.set_title('image: %i\npredict: %i\ntrue: %i' % (image_number[image_index],predicted_labels[image_number[image_index]], y[image_number[image_index]]), size=10)
        #sub.axis('off')
        #print X_file_names[image_index], predicted_labels[image_index], Y[image_index]

        # labeling 5 points, remote from labeled set
        if index < 5:
            #print(image_index)
            X_train=np.vstack([X_train,im_features[image_index]])
            Y_train=np.append(Y_train, np.array(Y[image_index]))
            images_in_train.append(image_index)
    #print "\n\n"
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