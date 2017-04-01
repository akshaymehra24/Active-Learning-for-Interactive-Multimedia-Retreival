# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 01:09:55 2017

@author: Akshay
"""

from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.externals import joblib

#im_features = joblib.load("E:\\Online Learning\\activelearning\\Python\\Caltech 101\\im_features.pkl")
#Y = joblib.load("E:\\Online Learning\\activelearning\\Python\\Caltech 101\\im_features_Y.pkl")


training_samples = []
precisions = []
recalls = []
accuracies = []
f1scores = []

batch_len = 1000
i = 0
while batch_len <= 8000:
    print "Round: ", i+1
    i+=1
    X_train = im_features[0:batch_len,:]
    Y_train = Y[0:batch_len]
    
    X_test = im_features[8000:,]
    Y_test = Y[8000:]
    
    model=svm.SVC(kernel='linear', C=1.0, probability=True, decision_function_shape='ovr').fit(X_train, Y_train)    
    #model=svm.SVC(kernel='rbf', C=2.0, probability=True, decision_function_shape='ovr').fit(X_train, Y_train)    
    #model=svm.LinearSVC(multi_class='ovr', C=1.0).fit(X_train, Y_train)
    predicted_labels = model.predict(im_features)  
    predicted_labels_test_set = model.predict(X_test)
    report = classification_report(Y_test, predicted_labels_test_set)
    #print("%s\n" % (classification_report(Y, predicted_labels)))

    training_samples.append(batch_len)
    accuracies.append(accuracy_score(Y, predicted_labels))
    precisions.append(precision_score(Y, predicted_labels, average="weighted"))
    recalls.append(recall_score(Y, predicted_labels, average="weighted"))
    f1scores.append(f1_score(Y, predicted_labels, average="weighted"))
    
    #accuracies.append(accuracy_score(Y_test, predicted_labels_test_set))
    #precisions.append(precision_score(Y_test, predicted_labels_test_set, average="weighted"))
    #recalls.append(recall_score(Y_test, predicted_labels_test_set, average="weighted"))
    #f1scores.append(f1_score(Y_test, predicted_labels_test_set, average="weighted"))
    
    batch_len += 1000
    
plt.figure()
plt.title("Accuracy Report")  
plt.xlabel("Training examples")
plt.ylabel("Accuracy Score")
plt.plot(training_samples, accuracies, label="Accuracy")
plt.legend(loc="best")

plt.figure()
plt.title("Precision Report")  
plt.xlabel("Training examples")
plt.ylabel("Precision Score")
plt.plot(training_samples, precisions, label="Precision")
plt.legend(loc="best")

plt.figure()
plt.title("Recall Report")  
plt.xlabel("Training examples")
plt.ylabel("Recall Score")
plt.plot(training_samples, recalls, label="Recall")
plt.legend(loc="best")

plt.figure()
plt.title("F1 Scores Report")  
plt.xlabel("Training examples")
plt.ylabel("F1 Scores Score")
plt.plot(training_samples, f1scores, label="F1 Scores")
plt.legend(loc="best") 