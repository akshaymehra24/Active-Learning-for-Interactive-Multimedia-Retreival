# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 02:25:38 2017

@author: Akshay
"""
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

#categories = ['rec.sport.baseball', 'rec.sport.hockey']
#categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
categories = ['sci.electronics', 'sci.med']
twenty_train = fetch_20newsgroups(categories=categories, shuffle=True, random_state=42)
#print twenty_train.target_names

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
#print X_train_counts.shape
#print count_vect.vocabulary_.get(u'algorithm')

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print X_train_tfidf.shape
#print X_train_tfidf[0]

X = X_train_tfidf
Y = twenty_train.target

print "Computing Cosine Similarities"

W = np.zeros([X.shape[0], X.shape[0]])
X_cosines = cosine_similarity(X)
W = (1 - X_cosines) / 0.03
W = np.exp(-W)

