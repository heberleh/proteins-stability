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
    for train_index, test_index in folds: #bootstrap? (100 iterations...)
        newdataset1 = dataset.get_sub_dataset_by_samples(train_index)        
        test_dataset1 = dataset.get_sub_dataset_by_samples(test_index)

        x_train = newdataset1.matrix[:, args[0]]  # data matrix        
        y_train = factorize(newdataset1.labels)[0]  # classes/labels of each sample from matrix x

        x_test = test_dataset1.matrix[:, args[0]]
        y_test = factorize(test_dataset1.labels)[0]

        classifier = classifiers_class[classifier_name]()
        std_scale = preprocessing.StandardScaler().fit(x_train)
        classifier.fit(std_scale.transform(x_train), y_train)
        
        t.append(y_test)
        p.append(classifier.predict(std_scale.transform(x_test)))
    return {'g':args[0], 't':t, 'p':p}
    

def split(arr, count):
     return [arr[i::count] for i in range(count)]


if __name__ == '__main__':

    start = time.time()

    global x, y, n, k, ns, nk, max_len, min_acc, classifier_class, min_break_accuracy, dataset

    dataset = Dataset("./dataset/train_6_samples_independent.txt", scale=False, normalize=False, sep='\t')

    dataset_test = Dataset("./dataset/test_6_samples_independent.txt", scale=False, normalize=False, sep='\t')


    rankes_names = ["rank_nsc.csv","rank_svm_linear.csv","rank_svm_rbf.csv","rank_kruskal.csv"]

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
    classifiers_names = ["nsc","svm-linear","svm-rbf", "tree"]#["svm","tree","nsc","naive_bayes","glm","sgdc","perceptron","svm-rbf"]#["svm","tree","nsc","naive_bayes","glm","sgdc","perceptron", "randForest"] #["svm","tree","nsc","naive_bayes"]
    classifiers_class = {"svm-linear":svm.LinearSVC, "tree":tree.DecisionTreeClassifier, "nsc":NearestCentroid, "naive_bayes":GaussianNB, "glm": LogisticRegression, "sgdc":SGDClassifier,"mtElasticNet":MultiTaskElasticNet,"elasticNet":ElasticNet,"perceptron":Perceptron, "randForest":RandomForestClassifier, "svm-rbf":SVC}

    #checking consistency
    #gene P61586_RHOA
    #print dataset.matrix[:, [230]]#, 139, 159, 117, 251, 29]]   

    n_cpu = cpu_count()        
    n_splits = n_cpu*10
    pool = Pool(processes=n_cpu)
        
    for rank_name in rankes_names:

        rank = []
        with open('./dataset/'+rank_name, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                rank.append(row[0])

        print "rank size", len(rank)

        n_classes = len(unique(dataset.labels))

        min_accuracy = 0
        static_min_accuracy = 0
        #min_accuracy = 0.8
        
        k = 7  # k-fold cross validations - outer    
        n_repeats = 100    
        rank_size = len(rank)

        accuracy_list = []

        all_signatures = {}
        ci = 1
        for name in classifiers_names:
            all_signatures[name] = {"idx":ci,"signatures":[]}
            ci+=1

        # BOOTSTRAP OR K-FOLD?
        #
        rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats,random_state=25684)
        gen = rskf.split(dataset.matrix, dataset.labels)
        folds = []
        for ti, tk in gen:
            folds.append([ti,tk])
                           
        maxAcc = 0.0    
        maxF1 = 0.0
        for name in classifiers_names:            
            current_args = []          
            for topN in range(1, rank_size+1):  
                genes_names = []
                for i in range(0,topN):
                    genes_names.append(rank[i])
                                
                genes_index = []
                for g in genes_names:                   
                    genes_index.append(dataset.genes.index(g))                
                current_args.append((genes_index, dataset, folds, name)) 

            result_part = pool.map(evaluate, current_args)
            gc.collect()

            print "resultpart size", len(result_part)
            matrix_cv = []
            matrix_independent = []            
            for i in range(len(result_part)):            
                result = result_part[i]
                group = [dataset.genes[i] for i in result["g"]]
                truth = result['t']
                predicted = result['p']
                accs_cv = []
                accs_independent = []
                print len(truth)
                print len(predicted)

                for j in range(len(truth)):
                    t = truth[j]
                    p = predicted[j]

                    acc = metrics.accuracy_score(t,p)
                    f1 = metrics.f1_score(t,p,average="weighted")
                    precision = metrics.precision_score(t,p,average="weighted")
                    recall = metrics.recall_score(t,p,average="weighted")
                    
                    accs_cv.append(acc)

                    x_train = dataset.matrix[:, result["g"]]  # data matrix
                    y_train = factorize(dataset.labels)[0]  # classes/labels of each sample from matrix x

                    x_test = dataset_test.matrix[:, result["g"]]
                    y_test = factorize(dataset_test.labels)[0]

                    classifier = classifiers_class[name]()
                    std_scale = preprocessing.StandardScaler().fit(x_train)
                    classifier.fit(std_scale.transform(x_train), y_train)                       
                    independent_prediction = classifier.predict(std_scale.transform(x_test))

                    acc_independent = metrics.accuracy_score(y_test,independent_prediction)
                    f1_independent = metrics.f1_score(y_test,independent_prediction,average="weighted")
                    precision_independent = metrics.precision_score(y_test,independent_prediction,average="weighted")
                    recall_independent = metrics.recall_score(y_test,independent_prediction,average="weighted")

                    accs_independent.append(acc_independent)

                    if acc > maxAcc:
                        maxAcc = acc 
                    if f1 > maxF1:
                        maxF1 = f1                       

                matrix_cv.append(accs_cv)
                matrix_independent.append(accs_independent)
            
            gc.collect()

            with open('./results/cv/cv_'+rank_name+'_'+name+'.csv', 'w') as f:
                line = "top-1"
                for topN in range(2, rank_size+1): 
                    line = line + ",top-"+str(topN)
                f.write(line+"\n")

                print "tamanhos\n"
                print len(matrix_cv), len(matrix_cv[0])
                for li in range(len(matrix_cv[0])):
                    line = str(matrix_cv[0][li])
                    for topN in range(1, rank_size):
                        line = line+","+str(matrix_cv[topN][li])
                    f.write(line+"\n")
                f.close()                
                gc.collect()

            with open('./results/cv/independent_'+rank_name+'_'+name+'.csv', 'w') as f:
                line = "top-1"
                for topN in range(2, rank_size+1): 
                    line = line + ",top-"+str(topN)
                f.write(line+"\n")

                for li in range(len(matrix_cv[0])):
                    line = str(matrix_cv[0][li])
                    for topN in range(1, rank_size):
                        line = line+","+str(matrix_cv[topN][li])
                    f.write(line+"\n")
                f.close()                
                gc.collect()
    
    pool.close()
    pool.join()
    pool.terminate()
    gc.collect()