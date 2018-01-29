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
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from numpy import mean, std, median
from sklearn import metrics
from datetime import datetime
import gc

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
    genes, n, x, y, min_break_accuracy, k, classifiers_names, folds = args[0], args[1], \
                                                              args[2], args[3], args[4], args[5], args[6], args[7]
    acc_list = []
    classifiers_class = {"tree":tree.DecisionTreeClassifier}#{"svm":svm.LinearSVC, "tree":tree.DecisionTreeClassifier, "nsc":NearestCentroid, "naive_bayes":GaussianNB}
    for classifier_name in classifiers_names:        
        acc_list.append(CrossValidation.run(genes, n, folds, x, y, classifiers_class[classifier_name], min_break_accuracy))
    gc.collect()

    return {"group":genes, "accs":acc_list}


def split(arr, count):
     return [arr[i::count] for i in range(count)]


def evaluate_genes(dataset, min_accuracy, n, k, max_group_size, classifiers_names):

    # breaks a k-fold if a minimum acc is not reached...
    # supose 5 folds, considering 4 folds with 100% acc, and we want min_accuracy of 95%
    # the fold must have at least 5*0.95 - 4 = 75% of acc... otherwise the average of acc of all folds would never reach 95% of acc
    min_break_accuracy = (k*min_accuracy) - (k-1)  #may be negative.. so any acc is considered, including zero, that is... all loops of k-fold will be executed

    folds = StratifiedKFold(y=dataset.labels, n_folds=k)
    print "folds", folds


    time_now = str(datetime.now()).replace("-", "_").replace(" ", "__").replace(":", "_").replace(".", "_")

    # fixed folds to calculate accuracy and compare the results between different filters
    # if folds were shuffled every time k-folds runs, each time a 4-gene-set would give a different number of
    # lists of genes with accuracy > min_accuracy
    header = "N"

    for classifier_name in classifiers_names:
        header = header + ",acc_" + classifier_name
    header = header + ",groups"

    genes_index = [i for i in range(len(dataset.genes))]

    # calculate N value using number of samples and number of classes
    x = dataset.matrix  # data matrix
    y = factorize(dataset.labels)[0]  # classes/labels of each sample from matrix x

    last_individual_time = 4.34675638278*60/39340   # log the time

          # if a group is > min_accuracy in ANY classifier (k-fold) it will be stored in this dictionary:
    high_acc_groups = {}


    for sub_list_size in range(1, max_group_size):
        genes_freq =[0 for i in range(len(dataset.genes))]
        with open('groups_accuracy_higher_'+str(min_accuracy)+'_size_'+str(sub_list_size)+'__'+time_now+'.csv', 'a') as f:

            f.write(header+"\n")

            print "Testing lists of", sub_list_size, "proteins."

            all_possible_groups = itertools.combinations(genes_index, sub_list_size)
            n_possible_groups = nCr(len(genes_index),sub_list_size)

            print "There are ", n_possible_groups, "lists to be tested. Estimated time to complete:",\
                n_possible_groups*last_individual_time/60/60, "horas."

            #n_splits = int(n_possible_groups/40)#1000*(3/(sub_list_size*1.0)))
            n_cpu = 6#cpu_count()
            n_splits = n_cpu*300
            start = time.time()
            pool = Pool(processes=n_cpu)
            hasnext = True
            executed = 0
            groups_found_count = 0
            while(hasnext):
                current_args = []
                for rep in range(n_splits):
                    try:
                        g = next(all_possible_groups)
                        current_args.append((g, n, x, y, min_break_accuracy, k, classifiers_names, folds))
                    except:
                        hasnext = False
                        break
                gc.collect()
               
                acc_list_part = pool.map(evaluate, current_args)

                for i in range(len(acc_list_part)):
                    if max(acc_list_part[i]["accs"]) >= min_accuracy:
                        result = acc_list_part[i]
                        for protidx in result["group"]:
                            genes_freq[protidx] += 1
                        group = [dataset.genes[i] for i in result["group"]]
                        accs = result["accs"]
                        line = str(len(group))
                        for acc in accs:
                            line = line + "," + str(acc)
                        for protein in group:
                            line = line + "," + protein
                        f.write(line+"\n")
                        groups_found_count += 1

                gc.collect()
                executed = executed + len(current_args)
                print "Restam ", n_possible_groups - executed, ". Grupos encontrados: ", groups_found_count
            f.close()
        with open('genes_freq_accuracy_higher_'+str(min_accuracy)+'_size_'+str(sub_list_size)+'__'+time_now+'.csv', 'a') as f2:
            f2.write('protein, frequency\n')
            for protidx in range(len(genes_freq)):
                f2.write(dataset.genes[protidx] +','+str(genes_freq[protidx])+'\n')
            f2.close()

        delta = time.time()-start
        last_individual_time = delta/float(n_possible_groups)
        del all_possible_groups
        print "The pool of processes took ", delta/60, "minutes to finish the job.\n"
        pool.close()
        pool.join()
        gc.collect()





if __name__ == '__main__':

    start = time.time()

    global x, y, n, k, ns, nk, max_len, min_acc, classifier_class, min_break_accuracy, dataset

    dataset = Dataset("./dataset/train_6_samples_independent.txt", scale=False, normalize=False, sep='\t')

    n_classes = len(unique(dataset.labels))
    print "Number of classes: ", str(n_classes)

    # loading classifiers
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors.nearest_centroid import NearestCentroid
    from sklearn import tree
    from sklearn import svm

    # classifiers that will be considered
    classifiers_names = ["tree"] #["svm","tree","nsc","naive_bayes"]

    min_accuracy = 0.8182
    #min_accuracy = 0.8

    n = 1  # n repetitions of k-fold cross validation
    k = 4  # k-fold cross validations
    max_group_size = 2 # the script will test groups of size 2, 3, ..., max_group_size-1 = [2, max_group_size)

    evaluate_genes(dataset, min_accuracy, n, k, max_group_size, classifiers_names)

    print "\n\n Time to complete the algorithm", (time.time() - start)/60, "minutes."



