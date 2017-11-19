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
    genes, n, folds, x, y, classifier_class, min_break_accuracy = args[0], args[1], \
                                                              args[2], args[3], args[4], args[5], args[6]
    return CrossValidation.run(genes, n, folds, x, y, classifier_class, min_break_accuracy)


def split(arr, count):
     return [arr[i::count] for i in range(count)]


def evaluate_genes(dataset, p_values, accuracies, genes_by_filter, n, k, classifier_class, min_break_accuracy):

    time_now = str(datetime.now()).replace("-", "_").replace(" ", "__").replace(":", "_").replace(".", "_")

    # fixed folds to calculate accuracy and compare the results between different filters
    # if folds were shuffled every time k-folds runs, each time a 4-gene-set would give a different number of
    # lists of genes with accuracy > min_accuracy
    folds = KFold(len(dataset.matrix), n_folds=k, shuffle=False)

    min_of_accuracies = min(accuracies)

    with open('results_'+time_now+'.csv', 'a') as f:
        header = "filter,p-value<,sublist-size,total_genes"
        for accuracy in accuracies:
            header += ",acc>" + str(accuracy)
        f.write(header+"\n")

        results = dict()
        for genes_dict in genes_by_filter:
            genes = genes_dict["genes"]  # attributes to be considered from x
            filter_name = genes_dict["filter"]
            p_v_filter = genes_dict["p-value"]

            results_by_pvalue = dict()
            for max_p_value in p_values:

                print "\n"
                print "Testing filter:", filter_name, ", p-value:", max_p_value

                filtered_genes = [genes[i] for i in range(len(p_v_filter)) if p_v_filter[i] < max_p_value]

                sub_dataset = dataset.get_sub_dataset(filtered_genes)  # construct a new dataset with only filtered genes

                genes_index = [i for i in range(len(sub_dataset.genes))]

                number_of_genes = len(genes_index)
                print "Number of genes:", number_of_genes

                # calculate N value using number of samples and number of classes
                x = sub_dataset.matrix  # data matrix
                y = factorize(sub_dataset.labels)[0]  # classes/labels of each sample from matrix x
                nk = len(unique(y))  # number of classes
                ns = len(x)  # number of samples
                # max number of genes to train a classifier, with ns samples and nk classes, taking no dimensionality curse.
                # max_len = (ns/(5*nk))  # -1???   (ns/nk)/breadth > 5 according to Wang
                max_len = 3  # temporary ignoring the line above  TODO remover essa linha

                last_individual_time = 0   # log the time

                high_acc = {}
                for sub_list_size in range(2, max_len+1):
                    print "Testing lists of", sub_list_size, "genes."

                    lists = [(subgenes, n, folds, x, y, classifier_class, min_break_accuracy)
                             for subgenes in itertools.combinations(genes_index, sub_list_size)]
                    n_lists = len(lists)
                    print "There are ", n_lists, "lists to be tested. Estimated time to complete:",\
                        n_lists*last_individual_time/60, "minutes."

                    pool = Pool(processes=cpu_count()-1)

                    start = time.time()

                    n_splits = len(lists)/10000
                    if n_splits == 0:
                        n_splits = 1

                    acc_list = []
                    splited_lists = split(lists, n_splits)
                    del lists
                    gc.collect()

                    while len(splited_lists) > 0:
                        lists_part = splited_lists[0]
                        acc_list_part = pool.map(evaluate, lists_part)
                        acc_list += [acc_list_part[i] for i
                                     in range(len(acc_list_part)) if acc_list_part[i] > min_of_accuracies]
                        del splited_lists[0]
                    gc.collect()

                    counter = {}
                    line = filter_name + ',' + str(max_p_value) + ',' + str(sub_list_size) + ',' + str(number_of_genes)

                    for accuracy in accuracies:
                        count = 0
                        for pv in acc_list:
                            if pv > accuracy:
                                count += 1
                        counter[accuracy] = count
                        line += ',' + str(count)
                        print "Accuracy", accuracy, " implies in ", count, "sets of genes."

                    f.write(line+"\n")

                    high_acc[sub_list_size] = counter
                    delta = time.time()-start
                    last_individual_time = delta/float(n_lists)
                    print "The pool of processes took ", delta/60, "minutes to finish the job.\n"

                    pool.close()
                    pool.join()
                results_by_pvalue[max_p_value] = high_acc
            results[filter_name] = results_by_pvalue
        f.close()
        return results

if __name__ == '__main__':
    global x, y, n, k, ns, nk, max_len, min_acc, classifier_class, min_break_accuracy

    dataset = Dataset("../../dataset/current/train.csv", scale=True)

    n_classes = len(unique(dataset.labels))
    print "Number of classes: ", n_classes

    # classifier_class = svm.LinearSVC

    #from sklearn.naive_bayes import GaussianNB
    #classifier_class = GaussianNB

    from sklearn.neighbors.nearest_centroid import NearestCentroid
    classifier_class = NearestCentroid

    #from sklearn import tree
    #classifier_class = tree.DecisionTreeClassifier

    # from sklearn.ensemble import RandomForestClassifier
    # classifier_class = RandomForestClassifier

    p_values = [0.01, 0.05, 0.1, 0.5, 1.1]
    accuracies = [0.99, 0.97, 0.95, 0.90, 0.85, 0.8]

    n = 1  # n repetitions of k-fold cross validation
    k = 9  # k-fold cross validations

    min_break_accuracy = 0.4  # if any loop of a k-fold returns acc < min_break_accuracy
    # it will break computation for the correspondent subset of genes

    if n_classes == 2:
        wil = WilcoxonRankSumTest(dataset)
        wil_z, wil_p = wil.run()

        print "Number of calculated p-values", len(wil_p)

        # wil_genes = [dataset.genes[i] for i in range(len(wil_p)) if wil_p[i] < 0.05]
        wilcoxon = {"filter": "Wilcoxon",
                    "genes": dataset.genes,
                    "p-value": wil_p}

        genes_by_filter = [wilcoxon]

        start = time.time()
        evaluate_genes(dataset, p_values, accuracies, genes_by_filter, n, k, classifier_class, min_break_accuracy)

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
    elif n_classes == 3:
        krus = KruskalRankSumTest3Classes(dataset)
        krus_h, krus_p = krus.run()

        print "Number of calculated p-values", len(krus_p)

        kruskal = {"filter": "Kruskal",
                    "genes": dataset.genes,
                    "p-value": krus_p}


        genes_by_filter = [kruskal]

        start = time.time()
        evaluate_genes(dataset, p_values, accuracies, genes_by_filter, n, k, classifier_class, min_break_accuracy)

        print ""
        print "\n\n Time to complete the algorithm", (time.time() - start)/60, "minutes."




