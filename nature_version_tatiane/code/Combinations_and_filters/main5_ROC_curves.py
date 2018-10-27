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
from sklearn.model_selection import StratifiedKFold
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

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


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

    classifiers_names = ["naive_bayes","glm"] #"svm-linear","svm-rbf",

    view_classifiers_names = {"svm-linear":"SVM - Linear","svm-rbf":"SVM - Radial", "glm":"Logistic Regression", "naive_bayes":"Gaussian Naive Bayes"}

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

    k = 8

    accuracy_list = []
    skf = StratifiedKFold(k)
    rskf = RepeatedStratifiedKFold(k,100)

    mean_aucs = {}
    for signature in signatures:
        mean_aucs[str(signature)] = {}
        for classifier_name in classifiers_names:
            mean_aucs[str(signature)][classifier_name] = -1.0

    aucs = {}
    for signature in signatures:
        aucs[str(signature)] = {}
        for classifier_name in ["naive_bayes","glm"]:
            aucs[str(signature)][classifier_name] = -1.0

    for classifier_name in classifiers_names:                     
        accs_independent = []
        matrix = []        
        
        for signature in signatures:
            figure(figsize=(11,8),dpi=90)
            plt.title(str(signature)+'ROC curve '+' ('+view_classifiers_names[classifier_name]+')')
            genes_index = []
            for g in signature:                   
                genes_index.append(dataset.genes.index(g))            

            line = str(len(signature))+","

            x_train = dataset.matrix[:, genes_index]  # data matrix
            y_train = factorize(dataset.labels)[0]  # classes/labels of each sample from matrix x

            x_test = dataset_test.matrix[:, genes_index]
            y_test = factorize(dataset_test.labels)[0]

            independent_probs = None
            if classifier_name == "naive_bayes":
                classifier = GaussianNB()
                std_scale = preprocessing.StandardScaler().fit(x_train)
                classifier.fit(std_scale.transform(x_train), y_train)                
                independent_probs = classifier.predict_proba(std_scale.transform(x_test))[:,1]

            elif classifier_name == "glm":
                classifier = LogisticRegression()
                std_scale = preprocessing.StandardScaler().fit(x_train)
                classifier.fit(std_scale.transform(x_train), y_train)                
                independent_probs = classifier.predict_proba(std_scale.transform(x_test))[:,1]

            elif classifier_name == "svm-rbf":
                classifier = SVC()
                std_scale = preprocessing.StandardScaler().fit(x_train)
                classifier.fit(std_scale.transform(x_train), y_train)                
                independent_probs = classifier.decision_function(std_scale.transform(x_test))

            elif classifier_name == "svm-linear":
                classifier = svm.LinearSVC()
                std_scale = preprocessing.StandardScaler().fit(x_train)
                classifier.fit(std_scale.transform(x_train), y_train)                
                independent_probs = classifier.decision_function(std_scale.transform(x_test))

            # save roc score for signature Sig for Classifier Clas
            print classifier_name            
            roc_score = roc_auc_score(y_test, independent_probs)

            # Curve
            fpr, tpr, thresholds = metrics.roc_curve(y_test, independent_probs)
            roc_auc = auc(fpr, tpr)       
            plt.plot(fpr, tpr, label="(AUC = %0.3f)"%(roc_auc))                  

            plt.legend(loc='lower right')
            plt.plot([0,1],[0,1],'r--')
            plt.xlim([-0.01,1.01])
            plt.ylim([-0.01,1.01])  #sensitivity vs 1-Specificity.
            plt.ylabel('Sensitivity')
            plt.xlabel('1 - Specificity')
            plt.tight_layout()
            savefig('./results/roc/'+'independent_roc_'+str(signature)+'_________'+classifier_name+'_'+'.png')
            plt.close()

            







            if classifier_name == "naive_bayes" or classifier_name == "glm":
                y_folds = []
                probs_folds = []

                tprs = []
                base_fpr = np.linspace(0, 1, 101)
                plt.figure(figsize=(12, 11))

                i = 0
                independent_probs = []
                for ktrain, ktest in rskf.split(x_train, y_train):
                    kx_train, kx_test = x_train[ktrain], x_train[ktest]
                    ky_train, ky_test = y_train[ktrain], y_train[ktest]                
                                        
                    if classifier_name == "naive_bayes":
                        classifier = GaussianNB()
                        std_scale = preprocessing.StandardScaler().fit(kx_train)
                        classifier.fit(std_scale.transform(kx_train), ky_train)                
                        independent_probs = classifier.predict_proba(std_scale.transform(kx_test))[:,1]

                    elif classifier_name == "glm":
                        classifier = LogisticRegression()
                        std_scale = preprocessing.StandardScaler().fit(kx_train)
                        classifier.fit(std_scale.transform(kx_train), ky_train)                
                        independent_probs = classifier.predict_proba(std_scale.transform(kx_test))[:,1]

                    probs_folds.extend(independent_probs)
                    y_folds.extend(ky_test)

                    if i == k-1:
                        fpr, tpr, _ = roc_curve(y_folds, probs_folds)
                        plt.plot(fpr, tpr, 'gray', alpha=0.10)
                        
                        tpr = interp(base_fpr, fpr, tpr)
                        tpr[0] = 0.0
                        tprs.append(tpr)
                    
                        i = 0                        
                        y_folds = []
                        probs_folds = []

                    else:
                        i += 1                      

                print "Probabilidades tamanho", len(probs_folds)
                
                plt.title(view_classifiers_names[classifier_name]+' ROC curve for '+str(signature))
                                                                                
                tprs = np.array(tprs)
                mean_tprs = tprs.mean(axis=0)
                std = tprs.std(axis=0)
                tprs_upper = np.minimum(mean_tprs + std, 1)
                tprs_lower = mean_tprs - std
                
                # mean_fprs = tprs.mean(axis=1)
                roc_auc = auc(base_fpr, mean_tprs)

                aucs[str(signature)][classifier_name] = roc_auc

                plt.plot(base_fpr, mean_tprs, 'b', label="ROC (AUC = %0.3f)"%(roc_auc))
                plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

                plt.legend(loc='lower right')
                plt.plot([0,1],[0,1],'r--')
                plt.xlim([-0.01,1.01])
                plt.ylim([-0.01,1.01])  #sensitivity vs 1-Specificity.
                plt.ylabel('Sensitivity')
                plt.xlabel('1 - Specificity')
                # plt.tight_layout()
                savefig('./results/roc/'+'cv_roc_CONCAT_'+str(signature)+'_________'+classifier_name+'_'+'.png')
                plt.close()










            
            tprs = []
            aucsl = []          
            mean_fpr = np.linspace(0,1,100)            
            for ktrain, ktest in rskf.split(x_train, y_train):
                kx_train, kx_test = x_train[ktrain], x_train[ktest]
                ky_train, ky_test = y_train[ktrain], y_train[ktest]
                
                classifier = None
                if classifier_name == "svm-rbf":
                    classifier = SVC()
                    std_scale = preprocessing.StandardScaler().fit(kx_train)
                    classifier.fit(std_scale.transform(kx_train), ky_train)                
                    independent_probs = classifier.decision_function(std_scale.transform(kx_test))
                elif classifier_name =="svm-linear":
                    classifier = svm.LinearSVC()                    
                    std_scale = preprocessing.StandardScaler().fit(kx_train)
                    classifier.fit(std_scale.transform(kx_train), ky_train)                
                    independent_probs = classifier.decision_function(std_scale.transform(kx_test))
                elif classifier_name == "naive_bayes":
                    classifier = GaussianNB()
                    std_scale = preprocessing.StandardScaler().fit(kx_train)
                    classifier.fit(std_scale.transform(kx_train), ky_train)                
                    independent_probs = classifier.predict_proba(std_scale.transform(kx_test))[:,1]
                elif classifier_name == "glm":
                    classifier = LogisticRegression()
                    std_scale = preprocessing.StandardScaler().fit(kx_train)
                    classifier.fit(std_scale.transform(kx_train), ky_train)                
                    independent_probs = classifier.predict_proba(std_scale.transform(kx_test))[:,1]
                
                fpr, tpr, t = roc_curve(ky_test, independent_probs)
                tprs.append(interp(mean_fpr, fpr, tpr))
                roc_auc = auc(fpr,tpr)
                aucsl.append(roc_auc)
        
            mean_tpr = np.mean(tprs, axis=0) 
            mean_tpr[-1] = 1.0
            std_auc = np.std(aucsl)
            mean_auc = auc(mean_fpr, mean_tpr)
            mean_aucs[str(signature)][classifier_name] = mean_auc

            figure(figsize=(11,8),dpi=90)
            plt.title(str(signature)+'ROC curve '+' ('+view_classifiers_names[classifier_name]+')')

        
            #AUC = %0.2f $\pm$ %0.2f
            plt.plot(fpr, tpr, label="Mean ROC (AUC = %0.3f)" % (mean_auc),lw=2,alpha=.8)
            
            plt.legend(loc='lower right')
            plt.plot([0,1],[0,1],'r--')
            plt.xlim([-0.01,1.01])
            plt.ylim([-0.01,1.01])  #sensitivity vs 1-Specificity.
            plt.ylabel('Sensitivity')
            plt.xlabel('1 - Specificity')
            plt.tight_layout()
            savefig('./results/roc/'+'cv_roc_MEAN_'+str(signature)+'_________'+classifier_name+'_'+'.png')
            plt.close()


    with open('./results/roc/crossvalidation_100rep_8fold_AUC.csv', 'w') as f:            
        f.write("n")
        for classifier_name in classifiers_names:          
            f.write(","+classifier_name+"(mean)")
        for classifier_name in ["glm","naive_bayes"]:
            f.write(","+classifier_name+"(concat probs)")
        f.write(",signature\n")

        for signature in signatures:
            f.write(str(len(signature)))
            for classifier_name in classifiers_names:          
                f.write(","+str(mean_aucs[str(signature)][classifier_name]))
            for classifier_name in ["glm","naive_bayes"]:
                f.write(","+str(aucs[str(signature)][classifier_name]))
            for element in signature:
                f.write(","+element)
            f.write("\n")
        