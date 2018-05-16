# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20

@author: Henry
"""
import csv
import gc
import itertools
import math
import time
from datetime import datetime
from multiprocessing import Lock, Pool, cpu_count

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from numpy import mean, median, min, std, unique
from pandas import factorize
from pylab import interp, savefig
from sklearn import manifold, metrics, preprocessing, svm, tree
from sklearn.cross_validation import KFold
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import (ElasticNet, LogisticRegression,
                                  MultiTaskElasticNet,
                                  PassiveAggressiveClassifier, Perceptron,
                                  SGDClassifier)
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm import SVC

from dataset import Dataset
from possibleEdge import PossibleEdge
from signature import Signature
from ttest import TTest
from wilcoxon import WilcoxonRankSumTest

#from wilcoxon import WilcoxonRankSumTest
#from kruskal import KruskalRankSumTest3Classes

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

def buildClassifier(name):
    classifier = None
    if name == "svm" or name == "linear-svm":
        classifier = svm.LinearSVC()
        predict_proba = False
        decision_function = True
    elif name == "tree":
        classifier =  tree.DecisionTreeClassifier()
        predict_proba = True
        decision_function = False
    elif name == "nsc":
        classifier = NearestCentroid()
        predict_proba = False
        decision_function = False    
    elif name == "naive_bayes":
        classifier = GaussianNB()
        predict_proba = True
        decision_function = False
    elif name == "glm":
        classifier = LogisticRegression()
        predict_proba = True
        decision_function = False        
    elif name == "sgdc":
        classifier = SGDClassifier()
        predict_proba = False # ?
        decision_function = False # ?        
    elif name == "mtElasticNet":
        classifier = MultiTaskElasticNet()
        predict_proba = False  #?
        decision_function = False #?         
    elif name == "elasticNet":
        classifier = ElasticNet()
        predict_proba = False  #?
        decision_function = False #?       
    elif name == "perceptron":
        classifier = Perceptron()
        predict_proba = False  #?
        decision_function = False #?    
    elif name == "randForest":
        classifier = RandomForestClassifier()
        predict_proba = True
        decision_function = False  
    elif name == "svm-rbf":
        classifier = SVC()
        predict_proba = False
        decision_function = True        
    elif name == "gbm": 
        classifier = GradientBoostingClassifier()
        predict_proba = True
        decision_function = False

        # "bag_nsc_eucl", "bag_kneighbor", "bag_tree", "bag_svm_rbf"
        #     if name in has_predict_proba:
        #     classifier = None
        # if name == "bag_nsc_eucl":
        #     classifier = BaggingClassifier(NearestCentroid(),random_state=7)#,n_estimators=20
        # elif name == "bag_kneighbor":
        #     classifier = BaggingClassifier(KNeighborsClassifier())
        # elif name == "bag_tree":
        #     classifier = BaggingClassifier(tree.DecisionTreeClassifier(),random_state=7)
        # elif name == "bag_svm_rbf":
        #     classifier = BaggingClassifier(SVC(),random_state=7)
        # else:
        #     classifier = classifiers_class[name]()
                
    return (classifier, predict_proba, decision_function)        


def evaluate(args):
    # classifiers_class = {"svm":svm.LinearSVC, "tree":tree.DecisionTreeClassifier, "nsc":NearestCentroid, "naive_bayes":GaussianNB, "glm": LogisticRegression, "sgdc":SGDClassifier,"mtElasticNet":MultiTaskElasticNet,"elasticNet":ElasticNet,"perceptron":Perceptron, "randForest":RandomForestClassifier, "svm-rbf":SVC, "gbm": GradientBoostingClassifier}

    # has_predict_proba = ["randForest","bag_nsc_eucl", "bag_kneighbor", "bag_tree", "bag_svm_rbf"]
                        # g = args[0]          (g, outer_datasets, name, k)
    outer_datasets, name, k = args[1], args[2], args[3]#, args[4]#, args[5]
    t = []
    p = []
    fpr_l = []
    tpr_l = []
    y_folds = []
    probs_folds = []
    accs = []
    aucs = []
    f1s = []
    precisions = []
    recalls = []
    specificities = []
    i = 0  
    processed_tpr = False      
    for train_dataset, test_dataset in outer_datasets: #bootstrap? (100 iterations...)       
        
        x_train = train_dataset.matrix[:, args[0]]  # data matrix
        y_train = factorize(train_dataset.labels)[0]  # classes/labels of each sample from matrix x

        x_test = test_dataset.matrix[:, args[0]]
        y_test = factorize(test_dataset.labels)[0]
          
        classifier, predict_proba, decision_function = buildClassifier(name)

        std_scale = preprocessing.StandardScaler().fit(x_train)
        classifier.fit(std_scale.transform(x_train), y_train)
        t.extend(y_test)
        p.extend(classifier.predict(std_scale.transform(x_test)))

        if predict_proba:   
            independent_probs = classifier.predict_proba(std_scale.transform(x_test))[:,1]
            probs_folds.extend(independent_probs)
            y_folds.extend(y_test)
        elif decision_function:            
            independent_probs = classifier.decision_function(std_scale.transform(x_test))
            probs_folds.extend(independent_probs)
            y_folds.extend(y_test)                    

        if i == k-1:
            i = 0

            acc = metrics.accuracy_score(t,p)
            f1 = metrics.f1_score(t,p)
            precision = metrics.precision_score(t,p)
            recall = metrics.recall_score(t,p)
            tn, fp, fn, tp = confusion_matrix(t, p).ravel()
            specificity = tn*1.0 / (tn+fp)

            accs.append(acc)
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            specificities.append(specificity)            
            t = []
            p = []

            if predict_proba or decision_function:                
                fpr, tpr, _ = roc_curve(y_folds, probs_folds)                
                fpr_l.append(fpr)
                tpr_l.append(tpr)
                aucs.append(auc(fpr, tpr))            
                y_folds = []
                probs_folds = []              
        else:
            i += 1

        if predict_proba or decision_function:
            processed_tpr = True

    return {'g':args[0], 'accs':accs, 'f1s': f1s, 'precisions': precisions,'recalls':recalls , 'specificities':specificities ,'fpr_l':fpr_l, 'tpr_l':tpr_l, 'p_tpr':processed_tpr, 'aucs':aucs}     



def split(arr, count):
     return [arr[i::count] for i in range(count)]


def multidimensional_scaling(data, labels, title, filename, colors, n_iterations):
        
    c=colors

    fig = plt.figure(figsize=(12, 11))
    ax = plt.axes()
    mds = manifold.MDS(2, max_iter=n_iterations, n_init=1)
    Y = mds.fit_transform(data)       
    plt.scatter(Y[:, 0], Y[:, 1], c=c, cmap=plt.cm.Spectral)
    plt.title(title+" (MDS)")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    savefig(path_results+filename+"_MDS"+'.png')
    plt.close()

    fig = plt.figure(figsize=(12, 11))
    ax = plt.axes()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, n_iter=n_iterations)
    Y = tsne.fit_transform(data)    
    # ax = fig.add_subplot(2, 5, 10)
    plt.scatter(Y[:, 0], Y[:, 1], c=c, cmap=plt.cm.Spectral)
    plt.title(title+ " (t-SNE)")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    savefig(path_results+filename+"_t_sne"+'.png')    
    plt.close()

if __name__ == '__main__':
    
    path_results = "./results/proteins/"
    path_dataset = "./dataset/proteins/"

    start = time.time()

    global x, y, n, k, ns, nk, max_len, min_acc, classifier_class, min_break_accuracy, dataset

    dataset = Dataset(path_dataset+"independent_train.txt", scale=False, normalize=False, sep='\t')

    dataset_test = Dataset(path_dataset+"independent_test.txt", scale=False, normalize=False, sep='\t')
    #dataset_test = dataset
    
    smote_enn = False
    smote_tomek = False
    smote = False
    
    newdataset1 = None
    if smote_enn:
        newdataset1 = dataset.getSmoteEnn(invert=True)
    elif smote_tomek:        
        newdataset1 = dataset.getSmoteTomek(invert=True)
    elif smote:
        newdataset1 = dataset.getSmote(invert=True)

    if newdataset1:
        print "------------------------------------------------------"
        print "Number of samples: "+ str(len(newdataset1.labels))
        print newdataset1.labels
        print "------------------------------------------------------"

    colors = ['red' if l == 'N+' else 'blue' for l in dataset.labels]
    multidimensional_scaling(data=dataset.get_scaled_data(), labels=dataset.labels, title="Original training set", filename="original_training", colors=colors, n_iterations=10000)

    #{pep8,pep12}    
    signatures = []
    if not 'proteins' in path_results:
        signatures = [["Pep8","Pep12","Pep9"]]

    for signature in signatures:
        multidimensional_scaling(data=dataset.get_sub_dataset(signature).get_scaled_data(), labels=dataset.labels, title="Original training set "+str(signature), filename="original_training_"+str(signature), colors=colors, n_iterations=10000)
        
    print("Testing SmoteEnn", dataset.getSmoteEnn(invert=True).labels)
    print("Testing SmoteTomek", dataset.getSmoteTomek(invert=True).labels)
    
    max_group_size = -1 # make max_group_size == len(dataset.genes)
    #max_group_size = 3 # limit the size of signature (max_group_size)
    filter = True
    filter_name = "wilcoxon"
    cutoff = 0.10

    # Over + Under resampling


    independent_test_only = False
    save_ROC_graphics = False
    
    k = 10  # k-fold cross validations - outer    
    n_repeats = 100
    min_acc_ROC = 0

    n_cpu = cpu_count()       
    n_splits = n_cpu

    selected_genes = []
    selected_genes_true = False
    if selected_genes_true:
        if 'proteins' in path_results:        
            selected_genes = ["LKHA4"]
        else:
            selected_genes = ["Pep12", "Pep13", "Pep14"]


    fdr_test = True
    if fdr_test:
        from sklearn.feature_selection import SelectFdr, chi2
        from scipy.stats import ranksums

        with open(path_results+'FDR.txt', 'w') as f:

            # ANOVA F-value
            data = {}
            dataset_fdr = dataset#.getSmoteEnn(invert=True)
            print("size after...",len(dataset.matrix))
            for i in range(len(dataset_fdr.genes)):
                data[dataset_fdr.genes[i]] = {}

            fdr = SelectFdr(alpha=0.05)            
            fdr.fit(X=dataset_fdr.matrix, y=dataset_fdr.labels)
            v_selected = fdr.get_support(indices=False)

            for i in range(len(v_selected)):
                data[dataset_fdr.genes[i]]["ANOVA F-value"] = v_selected[i]
            
            fdr = SelectFdr(score_func=chi2,alpha=0.05)            
            fdr.fit(X=dataset_fdr.matrix, y=dataset_fdr.labels)
            v_selected = fdr.get_support(indices=False)

            for i in range(len(v_selected)):
                data[dataset_fdr.genes[i]]["Chi-squared"] = v_selected[i]

            print(data)
            exit()


    if len(selected_genes) > 0:
        dataset = dataset.get_sub_dataset(selected_genes)
        dataset_test = dataset_test.get_sub_dataset(selected_genes)

    if filter:
        if filter_name == "wilcoxon":
            # Filtering train and test Datasets
            wil = WilcoxonRankSumTest(dataset)
            wil_z, wil_p = wil.run()            
            with open(path_results+'combinations/wilcoxon_test.csv', 'w') as f:
                f.write("gene,p-value\n")
                for i in range(len(dataset.genes)):
                    f.write(dataset.genes[i]+","+str(wil_p[i])+"\n")        

            dataset = dataset.get_sub_dataset([dataset.genes[i] for i in range(len(dataset.genes)) if wil_p[i]<cutoff])

            dataset_test = dataset_test.get_sub_dataset([dataset_test.genes[i] for i in range(len(dataset_test.genes)) if wil_p[i]<cutoff])

        elif filter_name == "ttest":
            # Filtering train and test Datasets
            ttest = TTest(dataset)
            ttest_t, ttest_p = ttest.run()            
            with open(path_results+'combinations/t_test.csv', 'w') as f:
                f.write("gene,p-value\n")
                for i in range(len(dataset.genes)):
                    f.write(dataset.genes[i]+","+str(ttest_p[i])+"\n")        

            dataset = dataset.get_sub_dataset([dataset.genes[i] for i in range(len(dataset.genes)) if ttest_p[i]<cutoff])

            dataset_test = dataset_test.get_sub_dataset([dataset_test.genes[i] for i in range(len(dataset_test.genes)) if ttest_p[i]<cutoff])
                



    # if smote_enn or smote_tomek or smote:
    #     print("SMOTE______TOMEK")

    #     colors = ['red' if l == 'N+' else 'blue' for l in dataset.labels]

    #     multidimensional_scaling(data=dataset.get_scaled_data(), labels=dataset.labels, title="Resampled training set", filename="resampled_training", colors=colors, n_iterations=10000)

    #     for signature in signatures:
    #         multidimensional_scaling(data=dataset.get_sub_dataset(signature).get_scaled_data(), labels=dataset.labels, title="Resampled training set "+str(signature), filename="resampled_training_"+str(signature), colors=colors, n_iterations=10000)

    print "Genes with Wilcox < ", str(cutoff),": ", dataset.genes

    n_classes = len(unique(dataset.labels))
    print "Number of classes: ", str(n_classes)
    
    print("Number of samples",len(dataset.matrix))
    print("Classes",dataset.labels)
    print("Variables", dataset.genes)
    
    # classifiers_class = {
    #                     "gbm": GradientBoostingClassifier,
    #                     "svm":svm.LinearSVC, 
    #                     "tree":tree.DecisionTreeClassifier, 
    #                     "nsc":NearestCentroid, 
    #                     "naive_bayes":GaussianNB, 
    #                     "glm": LogisticRegression, 
    #                     "sgdc":SGDClassifier,
    #                     "mtElasticNet":MultiTaskElasticNet,
    #                     "elasticNet":ElasticNet,
    #                     "perceptron":Perceptron, 
    #                     "randForest":RandomForestClassifier, 
    #                     "svm-rbf":SVC
    #                     }

    # classifiers that will be considered "bag_nsc_corr",  "nsc_corr",
    #"bag_nsc_corr", "bag_nsc_eucl", "nsc_eucl", "nsc_corr", "bag_kneighbor", "bag_tree", "bag_svm_rbf"
    #classifiers_names = ["nsc","tree","glm","randForest","svm-rbf","svm","naive_bayes","perceptron"]
    classifiers_names = ["randForest"]
    #"nsc","bag_nsc_eucl", "tree"]#,"bag_tree"]#,"bag_nsc_eucl","bag_svm_rbf"]#,"nsc"]#"svm-rbf", "tree",  "bag_svm_rbf", "bag_tree", "bag_kneighbor"]# ["svm","svm-rbf","tree","nsc","naive_bayes","glm","sgdc","perceptron", "randForest"] #["svm","tree","nsc","naive_bayes"]

    view_classifiers_names = {
                            "svm-linear":"Linear SVM",
                            "svm-rbf":"RBF SVM",
                            "tree": "Decision Tree",
                            "nsc": "Nearest Centroid",
                            "glm":"Logistic Regression", 
                            "naive_bayes":"Gaussian Naive Bayes",
                            "bag_nsc_eucl": "NSC Ensemble",
                            "bag_tree": "Decision Tree Ensemble",
                            "bag_svm_rbf": "RBF SVM Ensemble",
                            "gbm": "Gradient Boosting Classifier",
                            "randForest": "Random Forest"
                            }    


    # ARG
    if max_group_size == -1:
        max_group_size = len(dataset.genes)
    print "max groups size: ", max_group_size


    accuracy_list = []

    all_signatures = {}
    ci = 1
    for name in classifiers_names:
        all_signatures[name] = {"idx":ci,"signatures":[]}
        ci+=1

    # BOOTSTRAP OR K-FOLD?    
    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats,random_state=25684)    
    gen = rskf.split(dataset.matrix, dataset.labels)
    outer_folds = []
    for ti, tk in gen:
        outer_folds.append([ti,tk])

    outer_datasets = []
    n_samples = []
    for train_index, test_index in outer_folds: #bootstrap? (100 iterations...)
        newdataset1 = dataset.get_sub_dataset_by_samples(train_index)        
        test_dataset1 = dataset.get_sub_dataset_by_samples(test_index)        

        # http://contrib.scikit-learn.org/imbalanced-learn/stable/index.html
        if smote_enn:
            newdataset1 = newdataset1.getSmoteEnn(invert=True)
        elif smote_tomek:        
            newdataset1 = newdataset1.getSmoteTomek(invert=True)
        elif smote:
            newdataset1 = newdataset1.getSmote(invert=True)                
        outer_datasets.append([newdataset1,test_dataset1])
        n_samples.append(len(newdataset1.labels))

    n_samples_min =np.min(n_samples)
    n_samples_max = np.max(n_samples)
    n_samples_mean = np.mean(n_samples)
    n_samples_std = np.std(n_samples)

    pool = Pool(processes=n_cpu)        
    start = time.time()
    maxAcc = 0.0    
    maxF1 = 0.0
    n_samples = None
    with open(path_results+'combinations/predictions_all_classifiers_n_signatures.csv', 'w')as f:            
        for name in classifiers_names:
            
            line = ""
            if independent_test_only:
                line = "Classifier, N, Accuracy Independent Test, AUC Independent Test, F1 Independent Test, Sensitivity Independent Test, Specificity Independent Test, Precision Independent Test, Signature\n"
            else:
                line = "Classifier, N, Accuracy Mean, Accuracy SD, AUC Mean, AUC SD, F1, F1 SD, Sensitivity Mean, Sensitivity SD, Specificity Mean, Specificity SD, Precision Mean, Precision SD, Accuracy Independent Test, AUC Independent Test, F1 Independent Test, Sensitivity Independent Test, Specificity Independent Test, Precision Independent Test, Resampling N, Resampling N SD, Resampling Min, Resampling Max, Signature\n"

            f.write(line)
            print "Classifier name", name

            
            for sub_list_size in range(1, max_group_size+1):
                genes_freq =[0 for i in range(len(dataset.genes))]
                            
                print  sub_list_size, "proteins."

                genes_index = [i for i in range(len(dataset.genes))]

                all_possible_groups = itertools.combinations(genes_index, sub_list_size)
                n_possible_groups = nCr(len(genes_index),sub_list_size)       
            
                print "Combination of ",sub_list_size,". There are ", n_possible_groups, "lists to be tested."
                
                hasnext = True
                executed = 0
                group = None
                group_indexes = None
                groups_found_count = 0
                while(hasnext):
                    current_args = []
                    if independent_test_only:
                        try:
                            g = next(all_possible_groups)                            
                            group = [dataset.genes[i] for i in g]
                            group_indexes = g   
                        except:
                            hasnext = False
                            break
                        
                        x_train = dataset.matrix[:, group_indexes]  # data matrix
                        y_train = factorize(dataset.labels)[0]  # classes/labels of each sample from matrix x

                        x_test = dataset_test.matrix[:, group_indexes]
                        y_test = factorize(dataset_test.labels)[0]

                        classifier, predict_proba, decision_function = buildClassifier(name=name)
                        
                        std_scale = preprocessing.StandardScaler().fit(x_train)
                        classifier.fit(std_scale.transform(x_train), y_train)                       
                        independent_prediction = classifier.predict(std_scale.transform(x_test))                        
                        acc_independent = metrics.accuracy_score(y_test,independent_prediction)
                        f1_independent = metrics.f1_score(y_test,independent_prediction)
                        precision_independent = metrics.precision_score(y_test,independent_prediction)
                        recall_independent = metrics.recall_score(y_test,independent_prediction)
                        
                        tn, fp, fn, tp = confusion_matrix(y_test, independent_prediction).ravel()

                        specificity_independent = tn*1.0 / (tn+fp)                        

                        auc_independent = -1.0
                        if acc_independent >= min_acc_ROC and (predict_proba or decision_function):
                            independent_probs = None
                            if predict_proba:
                                independent_probs = classifier.predict_proba(std_scale.transform(x_test))[:,1]
                            else:
                                independent_probs = classifier.decision_function(std_scale.transform(x_test))
                                
                            auc_independent = roc_auc_score(y_test, independent_probs)
                            fpr, tpr, _ = roc_curve(y_test, independent_probs) 
                            
                            base_fpr = np.linspace(0, 1, 101)
                            tpr = interp(base_fpr, fpr, tpr)
                            tpr[0] = 0.0                        
                            roc_auc = auc(base_fpr, tpr)

                            if save_ROC_graphics:
                                plt.figure(figsize=(12, 11))
                                plt.title(view_classifiers_names[name]+' ROC curve for '+str(group))
                                plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Random', alpha=.8)
                                plt.plot(base_fpr, tpr, 'b', label="ROC (AUC = %0.3f)"%(roc_auc))
                                plt.legend(loc='lower right')
                                # plt.plot([0,1],[0,1],'r--')                            
                                plt.xlim([-0.01,1.01])
                                plt.ylim([-0.01,1.01])  #sensitivity vs 1-Specificity.
                                plt.ylabel('Sensitivity')
                                plt.xlabel('1 - Specificity')
                                # plt.tight_layout()
                                savefig(path_results+'roc/'+'roc_independent_'+str(group)+'_________'+name+'_'+'.png')
                                plt.close()
                    
                            if independent_test_only:
                                line = name + "," +\
                                        str(len(group))+","+\
                                        str(acc_independent)+","+\
                                        str(auc_independent)+","+\
                                        str(f1_independent)+","+\
                                        str(recall_independent)+","+\
                                        str(specificity_independent)+","+\
                                        str(precision_independent)                       
                                        
                            #"n, f1, acc, recall, precision, signature\n"                               
                            for g in group:
                                line = line + "," + str(g)

                            f.write(line+"\n")

                    else:
                        for rep in range(n_splits):
                            try:
                                g = next(all_possible_groups)                                
                                #dataset, outer_folds, classifier_name        
                                current_args.append((g, outer_datasets, name, k)) 
                            except:
                                hasnext = False
                                break
                        
                        gc.collect()
                        result_part = pool.map(evaluate, current_args)                        
                        gc.collect()

                        for i in range(len(result_part)):            
                            result = result_part[i]

                            group = [dataset.genes[i] for i in result["g"]]
                            group_indexes = result['g']
                            # truth = result['t']
                            # predicted = result['p']
                            processed_tpr = result['p_tpr']
                            
                            acc = np.mean(result['accs']) #metrics.accuracy_score(truth,predicted)
                            acc_std = np.std(result['accs'])                        

                            f1 = np.mean(result['f1s']) #metrics.f1_score(truth,predicted)
                            f1_std = np.std(result['f1s'])
                            
                            if acc > maxAcc:
                                maxAcc = acc 
                            if f1 > maxF1:
                                maxF1 = f1      

                            precision = np.mean(result['precisions']) #metrics.precision_score(truth,predicted)
                            precision_std = np.std(result['precisions'])

                            recall = np.mean(result['recalls']) #metrics.recall_score(truth,predicted)
                            recall_std = np.std(result['recalls'])

                            specificity = np.mean(result['specificities'])
                            specificity_std = np.std(result['specificities'])

                            auc_std = np.std(result['aucs'])

                            auc_cv_mean = -1.0
                            if acc > min_acc_ROC and processed_tpr:
                                fpr_l = result['fpr_l']
                                tpr_l = result['tpr_l']
                                base_fpr = np.linspace(0, 1, 101)
                                tprs = []  
                                gray_line = None
                                                    
                                for i in range(len(fpr_l)):                                
                                    tpr = tpr_l[i]
                                    fpr = fpr_l[i]                                    
                                    tpr = interp(base_fpr, fpr, tpr)
                                    tpr[0] = 0.0
                                    tprs.append(tpr)

                                tprs = np.array(tprs)
                                mean_tprs = tprs.mean(axis=0)
                                std_tprs = tprs.std(axis=0)
                                tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
                                tprs_lower = np.maximum(mean_tprs - std_tprs, 0)#mean_tprs - std    
                                mean_roc_auc = auc(base_fpr, mean_tprs)    

                                if save_ROC_graphics:
                                    plt.figure(figsize=(12, 11))                       
                                    plt.title(view_classifiers_names[name]+' ROC curve for '+str(group))
                                    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Random', alpha=.8)
                                    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.')
                                    blue_line, = plt.plot(base_fpr, mean_tprs, 'b',label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_roc_auc, auc_std),
                                    lw=2, alpha=.7)                        
                                    plt.legend(loc='lower right')                                       
                                    plt.xlim([-0.01,1.01])
                                    plt.ylim([-0.01,1.01])  #sensitivity vs 1-Specificity.
                                    plt.ylabel('Sensitivity')
                                    plt.xlabel('1 - Specificity')
                                    # plt.tight_layout()
                                    savefig(path_results+'roc/'+'cv_roc_CONCAT_'+str(group)+'_________'+name+'_'+'.png')
                                    plt.close()

                                auc_cv_mean = mean_roc_auc
                                            
                            x_train = dataset.matrix[:, group_indexes]  # data matrix
                            y_train = factorize(dataset.labels)[0]  # classes/labels of each sample from matrix x

                            x_test = dataset_test.matrix[:, group_indexes]
                            y_test = factorize(dataset_test.labels)[0]  

                            classifier, predict_proba, decision_function = buildClassifier(name=name)
                            
                            std_scale = preprocessing.StandardScaler().fit(x_train)
                            classifier.fit(std_scale.transform(x_train), y_train)                       
                            independent_prediction = classifier.predict(std_scale.transform(x_test))                        
                            acc_independent = metrics.accuracy_score(y_test,independent_prediction)
                            f1_independent = metrics.f1_score(y_test,independent_prediction)
                            precision_independent = metrics.precision_score(y_test,independent_prediction)
                            recall_independent = metrics.recall_score(y_test,independent_prediction)
                            
                            tn, fp, fn, tp = confusion_matrix(y_test, independent_prediction).ravel()

                            specificity_independent = tn*1.0 / (tn+fp)                        

                            auc_independent = -1.0
                            if acc_independent >= min_acc_ROC and (predict_proba or decision_function):
                                independent_probs = None
                                if predict_proba:
                                    independent_probs = classifier.predict_proba(std_scale.transform(x_test))[:,1]
                                else:
                                    independent_probs = classifier.decision_function(std_scale.transform(x_test))
                                    
                                auc_independent = roc_auc_score(y_test, independent_probs)
                                fpr, tpr, _ = roc_curve(y_test, independent_probs) 
                                
                                base_fpr = np.linspace(0, 1, 101)
                                tpr = interp(base_fpr, fpr, tpr)
                                tpr[0] = 0.0
                                roc_auc = auc(base_fpr, tpr)

                                if save_ROC_graphics:
                                    plt.figure(figsize=(12, 11))
                                    plt.title(view_classifiers_names[name]+' ROC curve for '+str(group))
                                    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Random', alpha=.8)
                                    plt.plot(base_fpr, tpr, 'b', label="ROC (AUC = %0.3f)"%(roc_auc))
                                    plt.legend(loc='lower right')
                                    # plt.plot([0,1],[0,1],'r--')                            
                                    plt.xlim([-0.01,1.01])
                                    plt.ylim([-0.01,1.01])  #sensitivity vs 1-Specificity.
                                    plt.ylabel('Sensitivity')
                                    plt.xlabel('1 - Specificity')
                                    # plt.tight_layout()
                                    savefig(path_results+'roc/'+'roc_independent_'+str(group)+'_________'+name+'_'+'.png')
                                    plt.close()
                                                       
                                line = name + "," +\
                                            str(len(group))+","+\
                                            str(acc)+","+\
                                            str(acc_std)+","+\
                                            str(auc_cv_mean)+","+\
                                            str(auc_std)+","+\
                                            str(f1)+","+\
                                            str(f1_std)+","+\
                                            str(recall)+","+\
                                            str(recall_std)+","+\
                                            str(specificity)+","+\
                                            str(specificity_std)+","+\
                                            str(precision)+","+\
                                            str(precision_std)+","+\
                                            str(acc_independent)+","+\
                                            str(auc_independent)+","+\
                                            str(f1_independent)+","+\
                                            str(recall_independent)+","+\
                                            str(specificity_independent)+","+\
                                            str(precision_independent)+","+\
                                            str(n_samples_mean)+","+\
                                            str(n_samples_std)+","+\
                                            str(n_samples_min)+","+\
                                            str(n_samples_max)                                                                                    
                                for g in group:
                                    line = line + "," + str(g)
                                f.write(line+"\n") 
                    f.flush()
                    gc.collect()
                    executed = executed + len(current_args)
                    print "Restam ", str(n_possible_groups - executed), "grupos.\n"
        f.close()
  
    delta = time.time()-start
    last_individual_time = delta/float(n_possible_groups)
    del all_possible_groups
    print "The pool of processes took ", delta/60, "minutes to finish the job.\n"
    print "\n\nAcurácia máxima: ", maxAcc,"\n\n"
    print "\n\nF1 máximo: ",maxF1,"\n\n"
    pool.close()
    pool.join()
    pool.terminate()
    gc.collect()

    exit()
