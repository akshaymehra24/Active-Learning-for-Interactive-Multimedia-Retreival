# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 19:59:35 2017

@author: Akshay
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm

digits = datasets.load_digits()
rng = np.random.RandomState(0)
indices = np.arange(len(digits.data))
rng.shuffle(indices)

X = digits.data[indices[:1700]]
y = digits.target[indices[:1700]]
images = digits.images[indices[:1700]]

n_total_samples = len(y)
n_labeled_points = 50

unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]
f = plt.figure()
y_train=y[1:n_labeled_points]
X_train=X[1:n_labeled_points]
iterations = 5
for i in range(0,iterations):        
    model=svm.SVC(kernel='linear', C=1.0, probability=True, decision_function_shape='ovr').fit(X_train, y_train)    
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
        if predicted_labels[j] >= 0:
            image_number.append(j)            
            for k in range(0, len(df_matrix[j])):
                ele=np.abs(df_matrix[j][k])
                if ele>m1:
                    m1=ele
            df_matrix_max.append(m1)
    # select five digit examples that the classifier is most uncertain about
    uncertainty_index_most = []
    index = 0
    co = 0
    while co < 5:
        element=np.argsort(df_matrix_max)[index]
        #print(element)
        if np.searchsorted(unlabeled_indices, element) != len(df_matrix_max):
            uncertainty_index_most = np.append(uncertainty_index_most, int(element))   
            co += 1
            index += 1
        else:
            index += 1
    #uncertainty_index_most = np.argsort(df_matrix_max)[:5]
    uncertainty_index_least = np.argsort(df_matrix_max)[-5:]
    uncertainty_index_least = uncertainty_index_least[::-1] #reversing the array
    uncertainty_index = np.append(uncertainty_index_most,uncertainty_index_least)
    # keep track of indices that we get labels for
    delete_indices = np.array([])
    
    for index, image_index in enumerate(uncertainty_index):
        #print(image_index)
        image_index=int(image_index)
        image = images[image_number[image_index]]        
        sub = f.add_subplot(iterations, 10, index + 1 + (10 * i))
        sub.imshow(image, cmap=plt.cm.gray_r)
        sub.set_title('image: %i\npredict: %i\ntrue: %i' % (image_number[image_index],predicted_labels[image_number[image_index]], y[image_number[image_index]]), size=10)
        sub.axis('off')

        # labeling 5 points, remote from labeled set
        if index < 5:
            #print(image_index)
            unlabeled_indices = np.delete(unlabeled_indices, np.searchsorted(unlabeled_indices,image_number[image_index]))
            X_train=np.vstack([X_train,X[image_number[image_index]]])
            y_train=np.append(y_train, np.array(y[image_number[image_index]]))

#f.suptitle("Active learning with SVM.\nRows show 5 most uncertain labels to learn with the next model.")
plt.subplots_adjust(0.5, 0.5, 1.8, 1.8, 3.2, 3.2)
plt.show()