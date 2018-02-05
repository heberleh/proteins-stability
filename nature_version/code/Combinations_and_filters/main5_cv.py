# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20

@author: Henry
"""
from pandas import factorize
from dataset import Dataset
#from wilcoxon import WilcoxonRankSumTest
#from kruskal import KruskalRankSumTest3Classes
from numpy import min, unique
import time
import itertools
from multiprocessing import Pool, Lock, cpu_count
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from wilcoxon import WilcoxonRankSumTest
from kruskal import KruskalRankSumTest3Classes
from signature import Signature
from possibleEdge import PossibleEdge
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from numpy import mean, std, median
from sklearn import metrics
from datetime import datetime
import numpy as np
import gc
import csv

from pylab import *
import matplotlib.pyplot as plt


import math

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


class CrossValidation():

    @staticmethod
    def kfold_cross_validation(classifier, x, y, folds, min_break_accuracy):
        """ Executes one k-fold cross-validation

        :param classifier:
        :param x: the numeric matrix of samples, lines as samples, columns are genes (attributes).
        :param y: the classes of each sample
        :param k: the number os folds (k-folds)
        :param min_break_accuracy: if any accuracy calculated is < min_break_accuracy, the algorithm stops
        :return: mean of k-fold accuracies
        """

        accuracy_list = []

        for train_index, test_index in folds:

            x_train, x_test = x[train_index, :], x[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            # standardize attributes to mean 0 and desv 1 (z-score)
            std_scale = preprocessing.StandardScaler().fit(x_train)

            classifier.fit(std_scale.transform(x_train), y_train)
            accuracy = metrics.accuracy_score(y_test, classifier.predict(std_scale.transform(x_test)))

            if accuracy > min_break_accuracy:
                accuracy_list.append(accuracy)
            else:
                return accuracy
        return mean(accuracy_list)

    @staticmethod
    def run(genes,n, folds, x, y, classifier_class, min_break_accuracy):
        """ Runs the N repetitions of K-fold cross-validation.

        :param genes: sublist of attributes that will be considered in cross-validation
        :param n: the number cross-validation will be repeated
        :param folds: the folds for k-fold
        :param x: the data matrix
        :param y: the samples classes
        :param classifier_class: the class of the classifier will be used in cross-validation
        :param min_break_accuracy: the min accuracy for any test in cross-validation, otherwise the algorithm stops
        :return: the mean accuracy from n repetitions of k-fold cross-validations: mean of means
        """

        x_ = x[:, genes]

        classifier = classifier_class()
        # n repetition of k-fold cross-validation
        acc_list = []
        for i in range(n):
            mean_cross_acc = CrossValidation.kfold_cross_validation(classifier, x_, y, folds, min_break_accuracy)
            if mean_cross_acc < min_break_accuracy:
                return mean_cross_acc
            else:
                acc_list.append(mean_cross_acc)

        return mean(acc_list)



def key(genes):
    return sum([2**i for i in genes])


def evaluate(args):
    classifiers_class = {"svm-linear":svm.LinearSVC, "tree":tree.DecisionTreeClassifier, "nsc":NearestCentroid, "naive_bayes":GaussianNB, "glm": LogisticRegression, "sgdc":SGDClassifier,"mtElasticNet":MultiTaskElasticNet,"elasticNet":ElasticNet,"perceptron":Perceptron, "randForest":RandomForestClassifier, "svm-rbf":SVC}

    dataset, folds, classifier_name = args[1], args[2], args[3]#, args[4], args[5]
    t = []
    p = []
    count = 0
    accs = []
    for train_index, test_index in folds: #bootstrap? (100 iterations...)
        count += 1
        newdataset1 = dataset.get_sub_dataset_by_samples(train_index)        
        test_dataset1 = dataset.get_sub_dataset_by_samples(test_index)

        x_train = newdataset1.matrix[:, args[0]]  # data matrix        
        y_train = factorize(newdataset1.labels)[0]  # classes/labels of each sample from matrix x

        x_test = test_dataset1.matrix[:, args[0]]
        y_test = factorize(test_dataset1.labels)[0]

        classifier = classifiers_class[classifier_name]()
        std_scale = preprocessing.StandardScaler().fit(x_train)
        classifier.fit(std_scale.transform(x_train), y_train)
        
        t.extend(y_test)
        p.extend(classifier.predict(std_scale.transform(x_test)))
        
        if count == 8:
            accs.append(metrics.accuracy_score(t,p))
            #accs.append(metrics.f1_score(t,p))
            count = 0
            t = []
            p = []

    return {'g':args[0], 'accs':accs}
    

def split(arr, count):
     return [arr[i::count] for i in range(count)]


if __name__ == '__main__':

    shuffle_labels = False

    start = time.time()

    global x, y, n, k, ns, nk, max_len, min_acc, classifier_class, min_break_accuracy, dataset


    dataset = Dataset("./dataset/independent_train.txt", scale=False, normalize=False, sep='\t')

    dataset_test = Dataset("./dataset/independent_test.txt", scale=False, normalize=False, sep='\t')

    if shuffle_labels:
        dataset.shuffle_labels()
        dataset_test.shuffle_labels()
    
    # loading classifiers
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors.nearest_centroid import NearestCentroid
    from sklearn import tree
    from sklearn import svm
    from sklearn.linear_model import SGDClassifier, Perceptron
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import ElasticNet
    from sklearn.linear_model import MultiTaskElasticNet
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    # classifiers that will be considered
    classifiers_names = ["nsc","svm-linear","svm-rbf", "tree","perceptron","naive_bayes"]
    #["svm","tree","nsc","naive_bayes","glm","sgdc","perceptron","svm-rbf"]#["svm","tree","nsc","naive_bayes","glm","sgdc","perceptron", "randForest"] #["svm","tree","nsc","naive_bayes"]
    classifiers_class = {"svm-linear":svm.LinearSVC, "tree":tree.DecisionTreeClassifier, "nsc":NearestCentroid, "naive_bayes":GaussianNB, "glm": LogisticRegression, "sgdc":SGDClassifier,"mtElasticNet":MultiTaskElasticNet,"elasticNet":ElasticNet,"perceptron":Perceptron, "randForest":RandomForestClassifier, "svm-rbf":SVC}

    rankes_classifier_names = ["rank_nsc.csv","rank_svm_linear.csv","rank_svm_rbf.csv","rank_kruskal.csv"]
    classifiers_names = ["nsc","svm-linear","svm-rbf", "tree"]
    
    view_rankes_classifier_names = {"rank_nsc.csv":"NSC","rank_svm_linear.csv":"SVM-RFE (linear)","rank_svm_rbf.csv":"SVM-RFE (radial)","rank_kruskal.csv": "Kruskal"}
    view_classifiers_names = {"nsc":"NSC","svm-linear":"SVM (linear)","svm-rbf":"SVM (radial)", "tree":"Decision Tree"}

    #checking consistency
    #gene P61586_RHOA
    #print dataset.matrix[:, [230]]#, 139, 159, 117, 251, 29]]   

    n_cpu = cpu_count()        
    n_splits = n_cpu*10
    pool = Pool(processes=n_cpu)
        
    signatures = []
    with open('./dataset/important_signatures.csv', 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            signatures.append(row)

    print "numer of signatures", len(signatures)

    labels = []
    for signature in signatures:
        labels.append(str(signature).replace('[','').replace(']','').replace('\'',''))

    n_classes = len(unique(dataset.labels))

    k = 8  # k-fold cross validations - outer    
    n_repeats = 100        

    accuracy_list = []

    # BOOTSTRAP OR K-FOLD?
    #
    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats,random_state=25684)
    gen = rskf.split(dataset.matrix, dataset.labels)
    folds = []
    for ti, tk in gen:
        folds.append([ti,tk])
                    
    for classifier_name in classifiers_names:            
        current_args = []  
        accs_independent = []
        matrix = []    
        for signature in signatures:           
            genes_index = []
            for g in signature:                   
                genes_index.append(dataset.genes.index(g))                
            current_args.append((genes_index, dataset, folds, classifier_name)) 

            x_train = dataset.matrix[:, genes_index]  # data matrix
            y_train = factorize(dataset.labels)[0]  # classes/labels of each sample from matrix x

            x_test = dataset_test.matrix[:, genes_index]
            y_test = factorize(dataset_test.labels)[0]

            classifier = classifiers_class[classifier_name]()
            std_scale = preprocessing.StandardScaler().fit(x_train)
            classifier.fit(std_scale.transform(x_train), y_train)                       
            independent_prediction = classifier.predict(std_scale.transform(x_test))

            acc_independent = metrics.accuracy_score(y_test,independent_prediction)
            f1_independent = metrics.f1_score(y_test,independent_prediction,average="weighted")
            precision_independent = metrics.precision_score(y_test,independent_prediction,average="weighted")
            recall_independent = metrics.recall_score(y_test,independent_prediction,average="weighted")

            accs_independent.append(acc_independent)

        result_part = pool.map(evaluate, current_args)
        gc.collect()

        print "resultpart size", len(result_part)
        matrix_cv = []                      
        for i in range(len(result_part)):            
            result = result_part[i]
            group = [dataset.genes[i] for i in result["g"]]
            accs_cv = result['accs']
            matrix_cv.append(accs_cv)
                
        gc.collect()

        matrix = np.matrix(matrix_cv).astype(float).transpose()

        print "colunas", len(matrix[0])
        print "linhas", len(matrix)

        # multiple box plots on one figure
        figure(figsize=(11,8),dpi=90)
        # NYCdiseases['chickenPox'] is a matrix 
        # with 30 rows (1 per year) and 12 columns (1 per month)
        boxplot(matrix)
        plt.ylim(-0.02, 1.02)

        xticks(range(1,len(labels)+1),labels, rotation=75)
        yticks()
        xlabel('Signatures')
        ylabel('Accuracies')#('F1')
        title(view_classifiers_names[classifier_name] + 'accuracies.')# ' F1 score')
        plt.tight_layout()
        savefig('./results/cv/'+'cv_boxplot'+classifier_name+'.png')
        plt.close()
    
    pool.close()
    pool.join()
    pool.terminate()
    gc.collect()
