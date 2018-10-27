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
    def kfold_cross_validation(classifier_class, x, y, folds, min_break_accuracy):
        """ Executes one k-fold cross-validation

        :param classifier:
        :param x: the numeric matrix of samples, lines as samples, columns are genes (attributes).
        :param y: the classes of each sample
        :param k: the number os folds (k-folds)
        :param min_break_accuracy: if any accuracy calculated is < min_break_accuracy, the algorithm stops
        :return: mean of k-fold accuracies
        """
        classifier = classifier_class()

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


def evaluate_genes(dataset, genes_rank_result, n, k):

    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors.nearest_centroid import NearestCentroid
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    classifiers = {"SVM": svm.LinearSVC,"GaussianNaiveBayes": GaussianNB, "NSC":NearestCentroid, "DecisionTree":tree.DecisionTreeClassifier, "RandomForest":RandomForestClassifier}

    time_now = str(datetime.now()).replace("-", "_").replace(" ", "__").replace(":", "_").replace(".", "_")

    genes = genes_rank_result["genes"]  # attributes to be considered from x
    filter_name = genes_rank_result["filter"]
    p_v_filter = genes_rank_result["p-value"]

    header = "p-valor < P, N proteins"
    classifier_names = classifiers.keys()
    for classifier_name in classifier_names:
        for rep in range(n):
            header = header + "," + classifier_name + "_rep-" + str(rep)

    p = 0.01

    with open('results_'+time_now+'.csv', 'a') as f:

        f.write(header+"\n")

        while (p < 1.01):
            filtered_genes = [genes[i] for i in range(len(p_v_filter)) if p_v_filter[i] < p]
            sub_dataset = dataset.get_sub_dataset(filtered_genes)  # construct a new dataset with only filtered genes

            x = sub_dataset.matrix  # data matrix
            y = factorize(sub_dataset.labels)[0]  # classes/labels of each sample from matrix x
            nk = len(unique(y))  # number of classes
            ns = len(x)  # number of samples

            line = str(p) + "," + str(len(filtered_genes))
            for classifier_name in classifier_names:
                for rep in range(n):
                    folds = KFold(len(dataset.matrix), n_folds=k, shuffle=True)
                    #print "P-value: "+str(p)+". Computing repetition " + str(rep) + " of classifier "+classifier_name+ ".\n"
                    kfold_acc = CrossValidation.kfold_cross_validation(classifiers[classifier_name], x, y, folds, min_break_accuracy=1.1)
                    line = line + ","+str(kfold_acc)
            p = p + 0.01
            f.write(line+"\n")
        f.close()


if __name__ == '__main__':
    global x, y, n, k, ns, nk, max_len, min_acc, classifier_class, min_break_accuracy

    dataset = Dataset("../../dataset/current/train.csv", scale=True)

    n_classes = len(unique(dataset.labels))
    print "Number of classes: ", n_classes

    n = 100  # n repetitions of k-fold cross validation
    k = 9 # k-fold cross validations

    if n_classes == 3:
        krus = KruskalRankSumTest3Classes(dataset)
        krus_h, krus_p = krus.run()

        print "Number of calculated p-values", len(krus_p)

        kruskal = {"filter": "Kruskal",
                    "genes": dataset.genes,
                    "p-value": krus_p}

        with open('kruskal_rank.csv', 'a') as f:
            for i in range(len(kruskal["genes"])):
                f.write(kruskal["genes"][i] +","+str(kruskal["p-value"][i])+"\n" )
            f.close()

        start = time.time()
        evaluate_genes(dataset, kruskal, n, k)

        print ""
        print "\n\n Time to complete the algorithm", (time.time() - start)/60, "minutes."




