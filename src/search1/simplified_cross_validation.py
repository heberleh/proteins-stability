# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 23:45:08 2015

@author: Henry
"""

from sklearn.cross_validation import KFold
from sklearn import preprocessing
import numpy as np
import pandas
from sklearn import metrics


class CrossValidation(object):
    """Executes a cross-validation.

    Attributes
        classifier_type: the class of the classifier will be used
        classifier: created instance of a given classifier class
        x: the numeric matrix of samples, lines as samples, columns are 
        genes (attributes).
        y: the classes of each sample      
        k: the number os folds (k-folds)
        n: the number of repetitions of k-fold cross-validation
        repetitions_metrics_list: list of results (metrics) from N validations
                                                                (repetitions)
    """
    ACCURACY_MEAN = 'accuracy_mean'
    ACCURACY_STD = 'accuracy_std'
    PRECISION_MEAN = 'precision_mean'
    PRECISION_STD = 'precision_std'
    RECALL_MEAN = 'recall_mean'
    RECALL_STD = 'recall_std'
    F1_MEAN = 'f1_mean'
    F1_STD = 'f1_std'

    def __init__(self, classifier, x, y, scale=False, normalize=False, k=10, n=3):
        self.classifier = classifier()
        self.x = x
        self.normalize = normalize
        self.scale = scale
        self.y = y
        self.k = k
        self.n = n
        self.repetitions_metrics_list = []

    def kfold_cross_validation(self):
        """ Executes one k-fold cross-validation using k defined when the object
        CrossValidation was created, as well the predefined x_train, etc.
        
        Returns a dict with mean and std of accuracy, precision, recall
        and f1 metrics.
        """            
        folds = KFold(len(self.x), n_folds=self.k, shuffle=True)
        accuracy_list = []

        for train_index, test_index in folds:

            x_train, x_test = self.x[train_index,:], self.x[test_index,:]
            y_train, y_test = self.y[train_index], self.y[test_index]
    
            # normalize attributes (genes) to min-max [0-1]
            if self.normalize:
                normalizer = preprocessing.StandardScaler().fit(x_train)
                x_train = normalizer.transform(x_train)
                x_test = normalizer.transform(x_test)
            
            # standardize attributes to mean 0 and desv 1 (z-score)
            if self.scale:
                std_scale = preprocessing.StandardScaler().fit(x_train)
                x_train = std_scale.transform(x_train)
                x_test = std_scale.transform(x_test)

            self.classifier.fit(x_train, y_train)
            accuracy_list.append(metrics.accuracy_score(y_test, self.classifier.predict(x_test)))
        return np.mean(accuracy_list)

    def run(self):
        """Runs the N repetitions of K-fold cross-validation.
        Stores the N results from the N validations in a list.
        """
        # n repetition of k-fold cross-validation
        acc_list = []
        for i in range(self.n):
            acc_list.append(self.kfold_cross_validation())
        return np.mean(acc_list)
