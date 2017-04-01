# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 21:00:11 2017

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
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


training_samples = []
precisions = []
recalls = []
accuracies = []
f1scores = []

batch_len = 10
while batch_len < 200:
    X_train = im_features[0:batch_len,:]
    Y_train = Y[0:batch_len]
    X_test = im_features[200:,]
    Y_test = Y[200:]
    
    model=svm.SVC(kernel='linear', C=1.0, probability=True, decision_function_shape='ovr').fit(X_train, Y_train)    
    predicted_labels = model.predict(im_features)  
    predicted_labels_test_set = model.predict(X_test)
    report = classification_report(Y_test, predicted_labels_test_set)
    
    training_samples.append(batch_len)
    #accuracies.append(accuracy_score(Y, predicted_labels))
    #precisions.append(average_precision_score(Y, predicted_labels, average="weighted"))
    #recalls.append(recall_score(Y, predicted_labels, average="weighted"))
    #f1scores.append(f1_score(Y, predicted_labels, average="weighted"))
    
    accuracies.append(accuracy_score(Y_test, predicted_labels_test_set))
    precisions.append(average_precision_score(Y_test, predicted_labels_test_set, average="weighted"))
    recalls.append(recall_score(Y_test, predicted_labels_test_set, average="weighted"))
    f1scores.append(f1_score(Y_test, predicted_labels_test_set, average="weighted"))
    
    batch_len += 5

'''plt.title("Classification Report")  
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.plot(training_samples, accuracies, label = "Accuracy")
plt.plot(training_samples, precisions, label = "Precision")
plt.plot(training_samples, recalls, label = "Recall")
plt.plot(training_samples, f1scores, label = "F1 Scores")
plt.legend(loc="best")'''