# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20

@author: Henry
"""


from pandas import factorize
from dataset import Dataset
from sklearn import svm
from wilcoxon import WilcoxonRankSumTest
from numpy import unique
import time
import itertools
from multiprocessing import Pool, Lock, cpu_count
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from numpy import mean
from sklearn import metrics
import gc


class CrossValidation():

    @staticmethod
    def kfold_cross_validation(classifier, x, y, k, min_break_accuracy):
        """ Executes one k-fold cross-validation

        :param classifier:
        :param x: the numeric matrix of samples, lines as samples, columns are genes (attributes).
        :param y: the classes of each sample
        :param k: the number os folds (k-folds)
        :param min_break_accuracy: if any accuracy calculated is < min_break_accuracy, the algorithm stops
        :return: mean of k-fold accuracies
        """

        folds = KFold(len(x), n_folds=k, shuffle=True)
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
    def run(genes, k, x, y, classifier_class, min_break_accuracy):
        """ Runs the N repetitions of K-fold cross-validation.

        :param genes: sublist of attributes that will be considered in cross-validation
        :param k: the number of folds in k-fold cross-validation
        :param x: the data matrix
        :param y: the samples classes
        :param classifier_class: the class of the classifier will be used in cross-validation
        :param min_break_accuracy: the min accuracy for any test in cross-validation, otherwise the algorithm stops
        :return: the mean accuracy from n repetitions of k-fold cross-validations: mean of means
        """

        classifier = classifier_class()
        acc = CrossValidation.kfold_cross_validation(classifier, x[:, genes], y, k, min_break_accuracy)
        #gc.collect()
        return acc



def key(genes):
    return sum([2**i for i in genes])


def evaluate(args):
    genes, k, x, y, classifier_class, min_break_accuracy = args[0], args[1], \
                                                              args[2], args[3], args[4], args[5]
    return CrossValidation.run(genes, k, x, y, classifier_class, min_break_accuracy)


def split(arr, count):
     return [arr[i::count] for i in range(count)]


def genes_find(genes):
    global min_acc, max_len, k, x, y, classifier_class, min_break_accuracy
    print "Total number of genes is", len(genes)
    all_high_acc = []
    print
    print
    last_individual_time = 0.0017

    for sub_list_size in range(2, max_len+1):
        print "Testing lists of", sub_list_size, "genes."

        lists = [(subgenes, k, x, y, classifier_class, min_break_accuracy)
                 for subgenes in itertools.combinations(genes, sub_list_size)]
        n_lists = len(lists)
        print "There are ", n_lists, "lists to be tested. Estimated time to complete:",\
            n_lists*last_individual_time/60, "minutes."

        gc.collect()

        pool = Pool(processes=cpu_count()-1)
        start = time.time()

        n_splits = len(lists)/10000
        if n_splits == 0:
            n_splits = 1

        l = []
        splited_lists = split(lists, n_splits)
        del lists
        gc.collect()

        while len(splited_lists) > 0:
            lists_part = splited_lists[0]
            acc_list_part = pool.map(evaluate, lists_part)
            l += [(key(lists_part[i][0]), acc_list_part[i]) for i in range(len(acc_list_part)) if acc_list_part[i] > min_acc]
            del splited_lists[0]
        gc.collect()

        print "There are ", len(l), "lists with accuracy > ", min_acc
        all_high_acc.append(l)

        # TODO calculate the frequency of each gene and store in a file

        delta = time.time() - start
        last_individual_time = delta/float(n_lists)
        print "The pool of processes took ", delta/60, "minutes to finish the job.\n"

        pool.close()
        pool.join()
        gc.collect()

    return all_high_acc



if __name__ == '__main__':
    global x, y, n, k, ns, nk, max_len, min_acc, classifier_class, min_break_accuracy

    dataset = Dataset("../spectral.csv", scale=True)

    wil = WilcoxonRankSumTest(dataset)
    z, p = wil.run()

    print "número de p-values calculados", len(p)
    wil_genes = [dataset.genes[i] for i in range(len(p)) if p[i] < 1.1]
    subdataset = dataset.get_sub_dataset(wil_genes)

    genes = [i for i in range(len(subdataset.genes))]

    # calculate N value using number of samples and number of classes

    x = subdataset.matrix  # data matrix

    y = factorize(subdataset.labels)[0]  # classes/labels of each sample from matrix x

    nk = len(unique(y))  # number of classes

    ns = len(x)  # number of samples

    # max number of genes to train a classifier, with ns samples and nk classes, taking no dimensionality curse.
    max_len = (ns/(5*nk))  # -1???   (ns/nk)/breadth > 5 according to Wang
    max_len = 6  # temporary ignoring the line above  TODO remover essa linha

    k = 10  # k-fold cross validations

    min_acc = 0.95  # only if accuracy of a subset of genes > min_acc it will be stored
    min_break_accuracy = 0.7  # if any loop of a k-fold returns acc < min_break_accuracy
    # it will break computation for the correspondent subset of genes

    classifier_class = svm.LinearSVC



    start = time.time()
    d = genes_find(genes=genes)

    # for size in d:
    #     count = 0
    #     for subset in size:
    #         count += 1
    #     print count

    print ""
    print "\n\n Time to complete the algorithm", (time.time() - start)/60, "minutes."

    # max_acc = max(d.values())
    # delta = 0.03
    # print
    # print
    # print "Sets with accuracy [", max_acc-delta, ",", max_acc, "]"
    # print
    # print
    # selected_sublists = []
    # for key in d.keys():
    #     if d[key] > max_acc-delta:
    #         selected_sublists.append(key)
    #
    # import evaluation
    # for key in selected_sublists:
    #     index = evaluation.AccuracyHash.get_genes(key)
    #     print [subdataset.genes[i] for i in index]



