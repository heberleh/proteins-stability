# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20

@author: Henry
"""
from pandas import factorize
from dataset import Dataset
from sklearn import svm
from wilcoxon import WilcoxonRankSumTest
from kruskal import KruskalRankSumTest3Classes
from numpy import min, unique
import time
import itertools
from multiprocessing import Pool, Lock, cpu_count
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from numpy import mean, std, median
from sklearn import metrics
from datetime import datetime
import gc


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

            del x_train
            del x_test
            del y_train
            del y_test
            del std_scale

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
        del x_
        del classifier

        return mean(acc_list)



def key(genes):
    return sum([2**i for i in genes])


def evaluate(args):
    genes, n, x, y, classifiers_class, min_break_accuracy, k = args[0], args[1], \
                                                              args[2], args[3], args[4], args[5], args[6]
    acc_list = []
    for classifier_name in classifiers_class.keys():
        acc_list.append(CrossValidation.run(genes, n, KFold(len(dataset.matrix), n_folds=k, shuffle=True), x, y, classifiers_class[classifier_name], min_break_accuracy))
    gc.collect()

    return {"group":genes, "accs":acc_list}


def split(arr, count):
     return [arr[i::count] for i in range(count)]


def evaluate_genes(dataset, min_accuracy, n, k, max_group_size, classifiers_class):

    # breaks a k-fold if a minimum acc is not reached...
    # supose 5 folds, considering 4 folds with 100% acc, and we want min_accuracy of 95%
    # the fold must have at least 5*0.95 - 4 = 75% of acc... otherwise the average of acc of all folds would never reach 95% of acc
    min_break_accuracy = (k*min_accuracy) - (k-1)  #may be negative.. so any acc is considered, including zero, that is... all loops of k-fold will be executed


    time_now = str(datetime.now()).replace("-", "_").replace(" ", "__").replace(":", "_").replace(".", "_")

    # fixed folds to calculate accuracy and compare the results between different filters
    # if folds were shuffled every time k-folds runs, each time a 4-gene-set would give a different number of
    # lists of genes with accuracy > min_accuracy


    with open('groups_with_high_accuracy_'+time_now+'.csv', 'a') as f:

        header = "N"
        for classifier_name in classifiers_class.keys():
            header = header + ",acc_" + classifier_name
        header = header + ",groups"
        f.write(header+"\n")

        results = dict()


        genes_index = [i for i in range(len(dataset.genes))]

        # calculate N value using number of samples and number of classes
        x = dataset.matrix  # data matrix
        y = factorize(dataset.labels)[0]  # classes/labels of each sample from matrix x

        last_individual_time = 0   # log the time

              # if a group is > min_accuracy in ANY classifier (k-fold) it will be stored in this dictionary:
        high_acc_groups = {}

        print classifiers_class.keys()

        for sub_list_size in range(2, max_group_size):
            print "Testing lists of", sub_list_size, "proteins."

            all_possible_groups = itertools.combinations(genes_index, sub_list_size)
            lists = [(subgenes, n, x, y, classifiers_class, min_break_accuracy, k)
                     for subgenes in all_possible_groups]
            n_lists = len(lists)
            print "There are ", n_lists, "lists to be tested. Estimated time to complete:",\
                n_lists*last_individual_time/60, "minutes."

            pool = Pool(processes=cpu_count())

            start = time.time()

            n_splits = int(len(lists)/1000*(3/(sub_list_size*1.0)))
            if n_splits == 0:
                n_splits = 1

            splited_lists = split(lists, n_splits)
            del lists
            gc.collect()

            while len(splited_lists) > 0:
                lists_part = splited_lists[len(splited_lists)-1] # [splited_lists)-1]  because in the end of the loop this element is removed from the list,
                #so the position len(splited_lists)-1 will be associated to another element

                acc_list_part = pool.map(evaluate, lists_part)

                for i in range(len(acc_list_part)):
                    if max(acc_list_part[i]["accs"]) > min_accuracy:
                        result = acc_list_part[i]
                        group = [dataset.genes[i] for i in result["group"]]
                        accs = result["accs"]
                        line = str(len(group))
                        for acc in accs:
                            line = line + "," + str(acc)
                        for protein in group:
                            line = line + "," + protein
                        f.write(line+"\n")

                del splited_lists[len(splited_lists)-1]
                del acc_list_part
                gc.collect()

            delta = time.time()-start
            last_individual_time = delta/float(n_lists)
            print "The pool of processes took ", delta/60, "minutes to finish the job.\n"

            pool.close()
            pool.join()
        f.close()




if __name__ == '__main__':

    start = time.time()

    global x, y, n, k, ns, nk, max_len, min_acc, classifier_class, min_break_accuracy

    dataset = Dataset("../../dataset/current/train.csv", scale=True)

    n_classes = len(unique(dataset.labels))
    print "Number of classes: ", n_classes

    # classifier_class = svm.LinearSVC

    from sklearn.naive_bayes import GaussianNB
    #classifier_class = GaussianNB

    from sklearn.neighbors.nearest_centroid import NearestCentroid
    #classifier_class = NearestCentroid

    from sklearn import tree
    #classifier_class = tree.DecisionTreeClassifier

    from sklearn.ensemble import RandomForestClassifier
    # classifier_class = RandomForestClassifier

    classifiers_class = {"svm":svm.LinearSVC, "tree":tree.DecisionTreeClassifier, "nsc":NearestCentroid, "naive_bayes":GaussianNB}#, "random_forest":RandomForestClassifier}
    min_accuracy = 0.929999

    n = 1  # n repetitions of k-fold cross validation
    k = 5  # k-fold cross validations
    max_group_size = 3 # the script will test groups of size 2, 3, ..., max_group_size-1 = [2, max_group_size)

    evaluate_genes(dataset, min_accuracy, n, k, max_group_size, classifiers_class)

    print "\n\n Time to complete the algorithm", (time.time() - start)/60, "minutes."



