# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 23:45:08 2015

@author: Henry
"""

from sklearn.cross_validation import KFold
from sklearn import preprocessing
import numpy as np
import pandas
from sklearn.metrics import accuracy_score
from sklearn import metrics
import itertools

class CrossValidation(object):
    """Executes a cross-validation.

    Attributes
        classifier_type: the class of the classifier will be used
        classifier: created instance of a given classifier class
        x: the numeric matrix of samples, lines as samples, columns are
        genes (attributes).
        genes: the list of genes' names
        y: the classes of each sample
        k: the number os folds (k-folds)
        n: the number of repetitions of k-fold cross-validation
        repetitions_metrics_list: list of retuls (metrics) from N validations
                                                                (repetitions)
    """
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1 = 'f1'


    def __init__(self, classifier_type, x, y, scale=False, normalize=False, k=5, n=3):
        self.classifier_init = classifier_type
        self.x = x
        self.normalize = normalize
        self.scale = scale
        self.factor = pandas.factorize(y)
        self.y = self.factor[0]
        self.k = k
        self.n = n
        self.repetitions_metrics_list = []



    def kfold_crossvalidation(self):
        """ Executes one k-fold crossvalidation using k defined when the object
        CrossValidation was created, as well the predefined x_train, etc.

        Returns a dict with mean and std of accuracy, precision, recall
        and f1 metrics.
        """

        folds = KFold(len(self.x), n_folds=self.k, shuffle=True)


        accuracy_list, precision_list, recall_list, f1_list = [], [], [], []

        complete_y_true = []
        complete_y_pred = []
        for train_index, test_index in folds:
            #print "train index", train_index
            #print "test index", test_index

            x_train, x_test = self.x[train_index,:], self.x[test_index,:]
            y_train, y_test = self.y[train_index], self.y[test_index]

            #normalize attributes (genes) to min-max [0-1]
            if self.normalize:
                normalizer = preprocessing.StandardScaler().fit(x_train)
                x_train = normalizer.transform(x_train)
                x_test = normalizer.transform(x_test)

            #standardize attributes to mean 0 and desv 1 (z-score)
            if self.scale:
                std_scale = preprocessing.StandardScaler().fit(x_train)
                x_train = std_scale.transform(x_train)
                x_test = std_scale.transform(x_test)

            classifier = self.classifier_init(x_train, y_train, x_test, y_test)
            classifier.fit()
            ypred = classifier.test()

            complete_y_pred = list(itertools.chain(complete_y_pred, ypred))
            complete_y_true = list(itertools.chain(complete_y_true, y_test))



        print complete_y_true, complete_y_pred
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        accuracy = 0.0

        #print complete_y_true, complete_y_pred

        if len(np.unique(complete_y_true)) == 2:
            precision = metrics.precision_score(complete_y_true, complete_y_pred)
            recall = metrics.recall_score(complete_y_true, complete_y_pred)
            f1 = metrics.f1_score(complete_y_true, complete_y_pred)
            accuracy = accuracy_score(complete_y_true, complete_y_pred)
        else:
            precision = metrics.precision_score(complete_y_true, complete_y_pred, average="macro")
            recall = metrics.recall_score(complete_y_true, complete_y_pred, average="macro")
            f1 = metrics.f1_score(complete_y_true, complete_y_pred, average="macro")
            accuracy = accuracy_score(complete_y_true, complete_y_pred)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
            }





    def run(self):
        """Runs the N repetitions of K-fold crossvalidation.
        Stores the N results from the N validations in a list.
        """
        # n repetition of k-fold crossvalidation
        for i in range(self.n):
            metrics = self.kfold_crossvalidation()
            self.repetitions_metrics_list.append(metrics)


    def get_list(self, metric):
        """ Returns the list values of required metric based on the N repetitions.
        Parameters:
            metric: 'accuracy', 'accuracy'... precision_... recall_...
            f1_...
        """
        l = []
        for rep_scores in self.repetitions_metrics_list:
            l.append(rep_scores[metric])
        #print l
        return l

    def get_mean(self, metric):
        """ Returns the mean of required metric based on the N repetitions.
        Parameters:
            metric: 'accuracy', ... precision_... recall_...
            f1_...
        """
        l = []

        for rep_scores in self.repetitions_metrics_list:
            l.append(rep_scores[metric])
        #print l
        return np.mean(l)


    def get_std(self, metric):
        """ Returns the std of required metric based on the N repetitions.
        Parameters:
            metric: 'accuracy', ... precision_... recall_...
            f1_...
        """
        l = []
        for rep_scores in self.repetitions_metrics_list:
            l.append(rep_scores[metric])
        return np.std(l)





