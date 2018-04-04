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
from ttest import TTest
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

from sklearn.ensemble import BaggingClassifier 
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pylab import interp, savefig

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

def buildClassifier(name=name):
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

    dataset, outer_folds, name, k = args[1], args[2], args[3], args[4]#, args[5]
    t = []
    p = []
    fpr_l = []
    tpr_l = []
    y_folds = []
    probs_folds = [] 
    accs = []
    f1s = []
    precisions = []
    recalls = []
    i = 0  
    processed_tpr = False
    for train_index, test_index in outer_folds: #bootstrap? (100 iterations...)
        newdataset1 = dataset.get_sub_dataset_by_samples(train_index)        
        test_dataset1 = dataset.get_sub_dataset_by_samples(test_index)

        x_train = newdataset1.matrix[:, args[0]]  # data matrix
        y_train = factorize(newdataset1.labels)[0]  # classes/labels of each sample from matrix x

        x_test = test_dataset1.matrix[:, args[0]]
        y_test = factorize(test_dataset1.labels)[0]
          
        classifier, predict_proba, decision_function = buildClassifier(name)

        std_scale = preprocessing.StandardScaler().fit(x_train)
        classifier.fit(std_scale.transform(x_train), y_train)
        t.extend(y_test)
        p.extend(classifier.predict(std_scale.transform(x_test)))

        if predict_proba:   
            independent_probs = classifier.predict_proba(std_scale.transform(x_test))[:,1]
            probs_folds.extend(independent_probs)
            y_folds.extend(y_test)
        

        if i == k-1:
            i = 0

            acc = metrics.accuracy_score(t,p)
            f1 = metrics.f1_score(t,p)
            precision = metrics.precision_score(t,p)
            recall = metrics.recall_score(t,p)
            accs.append(acc)
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            t = []
            p = []

            if predict_proba:                
                fpr, tpr, _ = roc_curve(y_folds, probs_folds)                
                fpr_l.append(fpr)
                tpr_l.append(tpr)            
                y_folds = []
                probs_folds = []              
        else:
            i += 1

        if predict_proba or decision_function:
            processed_tpr = True

    return {'g':args[0], 'accs':accs, 'f1s': f1s, 'precisions': precisions,'recalls':recalls ,'fpr_l':fpr_l, 'tpr_l':tpr_l, 'p_tpr':processed_tpr}     



def split(arr, count):
     return [arr[i::count] for i in range(count)]


if __name__ == '__main__':
    
    path_results = "./results/"
    path_dataset = "./dataset/"

    start = time.time()

    global x, y, n, k, ns, nk, max_len, min_acc, classifier_class, min_break_accuracy, dataset

    dataset = Dataset(path_dataset+"independent_train.txt", scale=False, normalize=False, sep='\t')

    dataset_test = Dataset(path_dataset+"independent_test.txt", scale=False, normalize=False, sep='\t')

    filter = True
    filter_name = "wilcoxon"
    cutoff = 0.1

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

            dataset_test = dataset.get_sub_dataset([dataset_test.genes[i] for i in range(len(dataset_test.genes)) if wil_p[i]<cutoff])
        elif filter_name == "ttest":
            # Filtering train and test Datasets
            ttest = TTest(dataset)
            ttest_t, ttest_p = ttest.run()            
            with open(path_results+'combinations/t_test.csv', 'w') as f:
                f.write("gene,p-value\n")
                for i in range(len(dataset.genes)):
                    f.write(dataset.genes[i]+","+str(ttest_p[i])+"\n")        

            dataset = dataset.get_sub_dataset([dataset.genes[i] for i in range(len(dataset.genes)) if ttest_p[i]<cutoff])

            dataset_test = dataset.get_sub_dataset([dataset_test.genes[i] for i in range(len(dataset_test.genes)) if ttest_p[i]<cutoff])
                


    print "Genes with Wilcox < ", str(cutoff),": ", dataset.genes

    n_classes = len(unique(dataset.labels))
    print "Number of classes: ", str(n_classes)

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
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC


    # classifiers that will be considered "bag_nsc_corr",  "nsc_corr",
    #"bag_nsc_corr", "bag_nsc_eucl", "nsc_eucl", "nsc_corr", "bag_kneighbor", "bag_tree", "bag_svm_rbf"
    classifiers_names = ["nsc","tree","glm","randForest","svm-rbf","svm","naive_bayes","perceptron"]
    #"nsc","bag_nsc_eucl", "tree"]#,"bag_tree"]#,"bag_nsc_eucl","bag_svm_rbf"]#,"nsc"]#"svm-rbf", "tree",  "bag_svm_rbf", "bag_tree", "bag_kneighbor"]# ["svm","svm-rbf","tree","nsc","naive_bayes","glm","sgdc","perceptron", "randForest"] #["svm","tree","nsc","naive_bayes"]
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

    min_accuracy = 0
    static_min_accuracy = 0
    #min_accuracy = 0.8
    
    k = 10  # k-fold cross validations - outer    
    n_repeats = 100
    max_group_size = len(dataset.genes)

    print "mas groups size: ", max_group_size
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
    outer_folds = []
    for ti, tk in gen:
        outer_folds.append([ti,tk])
        
    for ti ,tk in outer_folds:
        print "Folds:", ti, tk,"\n"
    
    #n_splits = int(n_possible_groups/40)#1000*(3/(sub_list_size*1.0)))
    n_cpu = cpu_count()
    
    n_splits = n_cpu*3
    # if (n_splits > n_possible_groups/2):
    #     if n_possible_groups > 300:
    #         n_splits = n_possible_groups/2.5

    pool = Pool(processes=n_cpu)
    
    start = time.time()
    maxAcc = 0.0    
    maxF1 = 0.0
    with open(path_results+'combinations/predictions_all_classifiers_n_signatures.csv', 'w') as f:
        for name in classifiers_names:
            line = "classifier, n, f1, f1 sd, acc, acc sd, recall, precision, auc_cv_mean, f1_independent, acc_independent, recall_independent, precision_independent, auc_independent, signature\n"
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
                groups_found_count = 0
                while(hasnext):
                    current_args = []
                    for rep in range(n_splits):
                        try:
                            g = next(all_possible_groups)               
                            #dataset, outer_folds, classifier_name        
                            current_args.append((g, dataset, outer_folds, name, k)) 
                        except:
                            hasnext = False
                            break
                    gc.collect()
                
                    result_part = pool.map(evaluate, current_args)

                    gc.collect()
                    
                    for i in range(len(result_part)):            
                        result = result_part[i]

                        group = [dataset.genes[i] for i in result["g"]]
                        # truth = result['t']
                        # predicted = result['p']
                        processed_tpr = result['p_tpr']
                        auc_cv_mean = -1.0
                        if processed_tpr:
                            fpr_l = result['fpr_l']
                            tpr_l = result['tpr_l']
                          
                            plt.figure(figsize=(12, 11))                       
                            plt.title(view_classifiers_names[name]+' ROC curve for '+str(group))    
                            
                            base_fpr = np.linspace(0, 1, 101)
                            tprs = []  
                            gray_line = None                          
                            for i in range(len(fpr_l)):                                
                                tpr = tpr_l[i]
                                fpr = fpr_l[i]
                                gray_line, = plt.plot(fpr, tpr, 'gray', alpha=0.11)

                                tpr = interp(base_fpr, fpr, tpr)
                                tpr[0] = 0.0
                                tprs.append(tpr)

                            tprs = np.array(tprs)
                            mean_tprs = tprs.mean(axis=0)
                            std = tprs.std(axis=0)
                            tprs_upper = np.minimum(mean_tprs + std, 1)
                            tprs_lower = mean_tprs - std        

                            # mean_fprs = tprs.mean(axis=1)     

                            roc_auc = auc(base_fpr, mean_tprs)                                      

                            # gray_line = mlines.Line2D([], [], color='gray', alpha=0.5, label="Rep-i ROC")
                            
                            blue_line, = plt.plot(base_fpr, mean_tprs, 'b')#, label="Mean ROC (AUC = %0.3f)"%(roc_auc))
                            label="Mean ROC (AUC = %0.3f)"%(roc_auc)
                            plt.rc('font', **{'family':'serif','serif':['Palatino']})
                            
                            plt.legend((gray_line, blue_line),("Rep-i ROC", label), loc=4)

                            plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
                            plt.legend(loc='lower right')
                            plt.plot([0,1],[0,1],'r--')
                            plt.xlim([-0.01,1.01])
                            plt.ylim([-0.01,1.01])  #sensitivity vs 1-Specificity.
                            plt.ylabel('Sensitivity')
                            plt.xlabel('1 - Specificity')
                            # plt.tight_layout()
                            savefig(path_results+'roc/'+'cv_roc_CONCAT_'+str(group)+'_________'+name+'_'+'.png')
                            plt.close()

                            auc_cv_mean = roc_auc


                        acc = np.mean(result['accs']) #metrics.accuracy_score(truth,predicted)
                        acc_std = np.std(result['accs']) 

                        f1 = np.mean(result['f1s']) #metrics.f1_score(truth,predicted)
                        f1_std = np.std(result['f1s'])

                        precision = np.mean(result['precisions']) #metrics.precision_score(truth,predicted)
                        recall = np.mean(result['recalls']) #metrics.recall_score(truth,predicted)


                        x_train = dataset.matrix[:, result["g"]]  # data matrix
                        y_train = factorize(dataset.labels)[0]  # classes/labels of each sample from matrix x

                        x_test = dataset_test.matrix[:, result["g"]]
                        y_test = factorize(dataset_test.labels)[0]

                        classifier, predict_proba, decision_function = buildClassifier(name=name)
                        
                        std_scale = preprocessing.StandardScaler().fit(x_train)
                        classifier.fit(std_scale.transform(x_train), y_train)                       
                        independent_prediction = classifier.predict(std_scale.transform(x_test))                        
                        acc_independent = metrics.accuracy_score(y_test,independent_prediction)
                        f1_independent = metrics.f1_score(y_test,independent_prediction)
                        precision_independent = metrics.precision_score(y_test,independent_prediction)
                        recall_independent = metrics.recall_score(y_test,independent_prediction)

                        auc_independent = -1.0
                        if predict_proba:
                            independent_probs = classifier.predict_proba(std_scale.transform(x_test))[:,1]
                            auc_independent = roc_auc_score(y_test, independent_probs)
                            fpr, tpr, _ = roc_curve(y_test, independent_probs) 

                            plt.figure(figsize=(12, 11))
                            plt.title(view_classifiers_names[name]+' ROC curve for '+str(group))
                            
                            base_fpr = np.linspace(0, 1, 101)
                            tpr = interp(base_fpr, fpr, tpr)
                            tpr[0] = 0.0

                            roc_auc = auc(base_fpr, tpr)
                            plt.plot(base_fpr, tpr, 'b', label="ROC (AUC = %0.3f)"%(roc_auc))
                            plt.legend(loc='lower right')
                            plt.plot([0,1],[0,1],'r--')
                            plt.xlim([-0.01,1.01])
                            plt.ylim([-0.01,1.01])  #sensitivity vs 1-Specificity.
                            plt.ylabel('Sensitivity')
                            plt.xlabel('1 - Specificity')
                            # plt.tight_layout()
                            savefig(path_results+'roc/'+'roc_independent_'+str(group)+'_________'+name+'_'+'.png')
                            plt.close()




                        if acc > maxAcc:
                            maxAcc = acc 
                        if f1 > maxF1:
                            maxF1 = f1                       

                        line =        name + "," +\
                                      str(len(group))+","+\
                                      str(f1)+","+\
                                      str(f1_std)+","+\
                                      str(acc)+","+\
                                      str(acc_std)+","+\
                                      str(recall)+","+\
                                      str(precision)+","+\
                                      str(auc_cv_mean)+","+\
                                      str(f1_independent)+","+\
                                      str(acc_independent)+","+\
                                      str(recall_independent)+","+\
                                      str(precision_independent)+","+\
                                      str(auc_independent)
                        #"n, f1, acc, recall, precision, signature\n"                               
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
