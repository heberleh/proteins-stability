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
    classifiers_class = {"svm":svm.LinearSVC, "tree":tree.DecisionTreeClassifier, "nsc":NearestCentroid, "naive_bayes":GaussianNB, "glm": LogisticRegression, "sgdc":SGDClassifier,"mtElasticNet":MultiTaskElasticNet,"elasticNet":ElasticNet,"perceptron":Perceptron, "randForest":RandomForestClassifier, "svm-rbf":SVC}

    dataset, outer_folds, classifier_name = args[1], args[2], args[3]#, args[4], args[5]
    t = []
    p = []
    for train_index, test_index in outer_folds: #bootstrap? (100 iterations...)
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
    return {'g':args[0], 't':t, 'p':p}
    

def split(arr, count):
     return [arr[i::count] for i in range(count)]


if __name__ == '__main__':

    start = time.time()

    global x, y, n, k, ns, nk, max_len, min_acc, classifier_class, min_break_accuracy, dataset

    dataset = Dataset("./dataset/independent_train.txt", scale=False, normalize=False, sep='\t')

    dataset_test = Dataset("./dataset/independent_test.txt", scale=False, normalize=False, sep='\t')

    filter = True
    filter_name = "wilcoxon"
    cutoff = 0.1

    if filter:
        if filter_name == "wilcoxon":
            # Filtering train and test Datasets
            wil = WilcoxonRankSumTest(dataset)
            wil_z, wil_p = wil.run()            
            with open('./results/combinations/wilcoxon_test.csv', 'w') as f:
                f.write("gene,p-value\n")
                for i in range(len(dataset.genes)):
                    f.write(dataset.genes[i]+","+str(wil_p[i])+"\n")        

            dataset = dataset.get_sub_dataset([dataset.genes[i] for i in range(len(dataset.genes)) if wil_p[i]<cutoff])

            dataset_test = dataset.get_sub_dataset([dataset_test.genes[i] for i in range(len(dataset_test.genes)) if wil_p[i]<cutoff])
        elif filter_name == "ttest":
            # Filtering train and test Datasets
            ttest = TTest(dataset)
            ttest_t, ttest_p = ttest.run()            
            with open('./results/combinations/t_test.csv', 'w') as f:
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
    from sklearn.svm import SVC

    # classifiers that will be considered
    classifiers_names = ["svm","svm-rbf","tree","nsc","naive_bayes","glm","sgdc","perceptron", "randForest"] #["svm","tree","nsc","naive_bayes"]
    classifiers_class = {"svm":svm.LinearSVC, "tree":tree.DecisionTreeClassifier, "nsc":NearestCentroid, "naive_bayes":GaussianNB, "glm": LogisticRegression, "sgdc":SGDClassifier,"mtElasticNet":MultiTaskElasticNet,"elasticNet":ElasticNet,"perceptron":Perceptron, "randForest":RandomForestClassifier, "svm-rbf":SVC}

    min_accuracy = 0
    static_min_accuracy = 0
    #min_accuracy = 0.8
    
    k = 8  # k-fold cross validations - outer    
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
    
    n_splits = n_cpu*10
    # if (n_splits > n_possible_groups/2):
    #     if n_possible_groups > 300:
    #         n_splits = n_possible_groups/2.5

    pool = Pool(processes=n_cpu)
    
    start = time.time()
    maxAcc = 0.0    
    maxF1 = 0.0
    with open('./results/combinations/predictions_all_classifiers_n_signatures.csv', 'w') as f:
        for name in classifiers_names:
            line = "classifier, n, f1, acc, recall, precision, f1_independent, acc_independent, recall_independent, precision_independent, signature\n"
            f.write(line)               
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
                            current_args.append((g, dataset, outer_folds, name)) 
                        except:
                            hasnext = False
                            break
                    gc.collect()
                
                    result_part = pool.map(evaluate, current_args)

                    gc.collect()
                    for i in range(len(result_part)):            
                        result = result_part[i]

                        group = [dataset.genes[i] for i in result["g"]]
                        truth = result['t']
                        predicted = result['p']

                        acc = metrics.accuracy_score(truth,predicted)
                        f1 = metrics.f1_score(truth,predicted)
                        precision = metrics.precision_score(truth,predicted)
                        recall = metrics.recall_score(truth,predicted)


                        x_train = dataset.matrix[:, result["g"]]  # data matrix
                        y_train = factorize(dataset.labels)[0]  # classes/labels of each sample from matrix x

                        x_test = dataset_test.matrix[:, result["g"]]
                        y_test = factorize(dataset_test.labels)[0]

                        classifier = classifiers_class[name]()
                        std_scale = preprocessing.StandardScaler().fit(x_train)
                        classifier.fit(std_scale.transform(x_train), y_train)                       
                        independent_prediction = classifier.predict(std_scale.transform(x_test))

                        acc_independent = metrics.accuracy_score(y_test,independent_prediction)
                        f1_independent = metrics.f1_score(y_test,independent_prediction)
                        precision_independent = metrics.precision_score(y_test,independent_prediction)
                        recall_independent = metrics.recall_score(y_test,independent_prediction)

                        if acc > maxAcc:
                            maxAcc = acc 
                        if f1 > maxF1:
                            maxF1 = f1                       

                        line =        name + "," +\
                                      str(len(group))+","+\
                                      str(f1)+","+\
                                      str(acc)+","+\
                                      str(recall)+","+\
                                      str(precision)+","+\
                                      str(f1_independent)+","+\
                                      str(acc_independent)+","+\
                                      str(recall_independent)+","+\
                                      str(precision_independent)
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

    # *N,acc_tree,groups
    # 1,0.833333333333,CON__P00761
    # 1,0.833333333333,P02769_BSA        
    independent_genes1_with_high_acc = set()        

    for name in classifiers_names:
        maxAcc = static_min_accuracy
        signatures_data = all_signatures[name]            
        signatures = signatures_data["signatures"]

        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            reader.next()
            for row in reader:            
                if float(row[signatures_data["idx"]]) > maxAcc:
                    maxAcc = float(row[signatures_data["idx"]])
            csv_file.close()
        
        print "New max acc for classifier ", name, ":", maxAcc           

        # for each signature...   
        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            reader.next()
            print "Signatures: "            
            for row in reader:  
                acc = row[signatures_data["idx"]]              
                genes = []
                for col in range(len(classifiers_names)+1,len(row)):
                    genes.append(row[col])                       
                signature = Signature(genes)
                            
                genes_index = []
                for i in range(len(dataset.genes)):
                    for gene in genes:
                        if dataset.genes[i] == gene:
                            genes_index.append(i)
                
                x_tr = x_train[:,genes_index]  # data matrix
                y_tr = y_train
                x_te = x_test[:,genes_index]
                y_te = y_test

                classifier = classifiers_class[name]()
                # standardize attributes to mean 0 and desv 1 (z-score)
                std_scale = preprocessing.StandardScaler().fit(x_tr)
                classifier.fit(std_scale.transform(x_tr), y_tr)
                # accuracy = metrics.accuracy_score(y_te, classifier.predict(std_scale.transform(x_te)))
                predicted = classifier.predict(std_scale.transform(x_te))

                found = False
                for sig in signatures:
                    if sig == signature:
                        sig.add_pair_truth_prediced(y_te,predicted)
                        found = True
                        break
                if not found:
                    signature.add_pair_truth_prediced(y_te,predicted)                        
                    signatures.append(signature)
        
    
        # if len(all_signature_genes) > 0:
        #     genes_index = []
        #     for i in range(len(dataset.genes)):
        #         for gene in all_signature_genes:
        #             if dataset.genes[i] == gene:
        #                 genes_index.append(i)

        #     x_tr = x_train[:,genes_index]  # data matrix
        #     y_tr = y_train
        #     x_te = x_test[:,genes_index]
        #     y_te = y_test

        #     classifier = tree.DecisionTreeClassifier()
        #     # standardize attributes to mean 0 and desv 1 (z-score)
        #     std_scale = preprocessing.StandardScaler().fit(x_tr)
        #     classifier.fit(std_scale.transform(x_tr), y_tr)
        #     accuracy = metrics.accuracy_score(y_te, classifier.predict(std_scale.transform(x_te)))  
            
        #     with open('./results/combinations/kfold_test_with_all_signature_genes.csv', 'a') as f:
        #         f.write(str(len(all_signature_genes))+","+str(accuracy))
        #         for gene in all_signature_genes:
        #             f.write(","+gene)
        #         f.write("\n")
        #         f.close()
        csv_file.close()
        
    for name in classifiers_names:
        with open('./results/combinations/all_sig_dcv_acc_'+name+'.csv', 'w') as f:
            signatures = all_signatures[name]["signatures"]  
            min_acc = 1
            max_acc = 0


            f.write("max acc,"+min_acc+"\n")
            f.write("min acc,"+max_acc+"\n")
            f.write("size, avg dcv acc, std dcv acc, signature\n")
                    
            for signature in signatures:
                f.write(str(signature.get_n())+","
                        +str(np.mean(signature.get_independent_test_accs()))+","
                        +str(np.std(signature.get_independent_test_accs()))                        
                        )
                for gene in signature.genes:
                    f.write(","+gene)
                f.write("\n")
            f.close()
    



    
    # print "\n\n Time to complete the algorithm", (time.time() - start)/60, "minutes."

    # with open('./results/combinations/signatures_combinations.csv', 'w') as f:
    #     f.write("n,avg_acc,stddev_acc,ind_avg_acc,ind_stddev_acc,weight,signature\n")       
    #     for signature in all_signatures:
    #         print "accs", signature.get_independent_test_accs()
    #         mean_acc=0.0
    #         std_acc=0.0
    #         mean_independent_acc=0.0
    #         std_independent_acc=0.0
    #         if signature.get_weight() == 1: 
    #             mean_acc= signature.get_accs()[0]
    #             std_acc=0
    #             mean_independent_acc=signature.get_independent_test_accs()[0]
    #             std_independent_acc=0
    #         else:
    #             mean_acc=np.mean(signature.get_accs())
    #             std_acc=np.std(signature.get_accs())
    #             mean_independent_acc=np.mean(signature.get_independent_test_accs())
    #             std_independent_acc=np.std(signature.get_independent_test_accs()) 

    #         f.write(
    #             str(signature.get_n())+","+
    #             str(mean_acc)+","+
    #             str(std_acc)+","+
    #             str(mean_independent_acc)+","+
    #             str(std_independent_acc)+","+
    #             str(signature.get_weight()))

    #         for gene in signature.genes:
    #             f.write(","+gene)
    #         f.write("\n")
    #     f.close()
    
    
    # intersection_of_all_independent_proteins = kfold_independent_proteins[0]    
    # union_of_all_independent_proteins = set()
    # for key in kfold_independent_proteins:        

    #     union_of_all_independent_proteins = union_of_all_independent_proteins | kfold_independent_proteins[key]

    #     intersection_of_all_independent_proteins = intersection_of_all_independent_proteins & kfold_independent_proteins[key]


    # intersection_of_all_signature_proteins = kfold_signature_proteins[0]    
    # union_of_all_signature_proteins = set()
    # for key in kfold_signature_proteins:        

    #     union_of_all_signature_proteins = union_of_all_signature_proteins | kfold_signature_proteins[key]

    #     intersection_of_all_signature_proteins = intersection_of_all_signature_proteins & kfold_signature_proteins[key]


    # with open('./results/combinations/independent_test_set.csv', 'w') as f:
    #     f.write("type,acc,n,proteins\n")  
    
    #     # test independent dataset
        
    #     #__________________________________________________
    #     # with intersection of "most independent proteins"
    #     if len(intersection_of_all_independent_proteins) > 0:
    #         genes_index = []
    #         for i in range(len(dataset.genes)):
    #             for gene in intersection_of_all_independent_proteins:
    #                 if dataset.genes[i] == gene:
    #                     genes_index.append(i)

    #         x_tr = x_train[:,genes_index]  # data matrix
    #         y_tr = y_train
    #         x_te = x_test[:,genes_index]
    #         y_te = y_test

    #         classifier = tree.DecisionTreeClassifier()
    #         # standardize attributes to mean 0 and desv 1 (z-score)
    #         std_scale = preprocessing.StandardScaler().fit(x_tr)
    #         classifier.fit(std_scale.transform(x_tr), y_tr)
    #         accuracy = metrics.accuracy_score(y_te, classifier.predict(std_scale.transform(x_te)))
            
    #         f.write("intersection_of_all_independent_proteins,"+str(accuracy)+","
    #         +str(len(intersection_of_all_independent_proteins)))
    #         for gene in intersection_of_all_independent_proteins:
    #             f.write(","+gene)
    #         f.write("\n")
        
        
    #     #__________________________________________________
    #     # with union of "most independent proteins"
    #     if len(union_of_all_independent_proteins) > 0:
    #         genes_index = []
    #         for i in range(len(dataset.genes)):
    #             for gene in union_of_all_independent_proteins:
    #                 if dataset.genes[i] == gene:
    #                     genes_index.append(i)

    #         x_tr = x_train[:,genes_index]  # data matrix
    #         y_tr = y_train
    #         x_te = x_test[:,genes_index]
    #         y_te = y_test

    #         classifier = tree.DecisionTreeClassifier()
    #         # standardize attributes to mean 0 and desv 1 (z-score)
    #         std_scale = preprocessing.StandardScaler().fit(x_tr)
    #         classifier.fit(std_scale.transform(x_tr), y_tr)
    #         accuracy = metrics.accuracy_score(y_te, classifier.predict(std_scale.transform(x_te)))

    #         f.write("union_of_all_independent_proteins,"+str(accuracy)+","
    #         +str(len(union_of_all_independent_proteins)))
    #         for gene in union_of_all_independent_proteins:
    #             f.write(","+gene)
    #         f.write("\n")

    #     #__________________________________________________
    #     # with intersection of "union of signature proteins"
    #     if len(intersection_of_all_signature_proteins) > 0:
    #         genes_index = []
    #         for i in range(len(dataset.genes)):
    #             for gene in intersection_of_all_signature_proteins:
    #                 if dataset.genes[i] == gene:
    #                     genes_index.append(i)

    #         x_tr = x_train[:,genes_index]  # data matrix
    #         y_tr = y_train
    #         x_te = x_test[:,genes_index]
    #         y_te = y_test

    #         classifier = tree.DecisionTreeClassifier()
    #         # standardize attributes to mean 0 and desv 1 (z-score)
    #         std_scale = preprocessing.StandardScaler().fit(x_tr)
    #         classifier.fit(std_scale.transform(x_tr), y_tr)
    #         accuracy = metrics.accuracy_score(y_te, classifier.predict(std_scale.transform(x_te)))
            
    #         f.write("intersection_of_all_signature_proteins,"+str(accuracy)+","
    #         +str(len(intersection_of_all_signature_proteins)))
    #         for gene in intersection_of_all_signature_proteins:
    #             f.write(","+gene)
    #         f.write("\n")        

    #     #__________________________________________________
    #     # with union of "union of signature proteins"
    #     if len(union_of_all_signature_proteins) > 0:
    #         genes_index = []
    #         for i in range(len(dataset.genes)):
    #             for gene in union_of_all_signature_proteins:
    #                 if dataset.genes[i] == gene:
    #                     genes_index.append(i)

    #         x_tr = x_train[:,genes_index]  # data matrix
    #         y_tr = y_train
    #         x_te = x_test[:,genes_index]
    #         y_te = y_test

    #         classifier = tree.DecisionTreeClassifier()
    #         # standardize attributes to mean 0 and desv 1 (z-score)
    #         std_scale = preprocessing.StandardScaler().fit(x_tr)
    #         classifier.fit(std_scale.transform(x_tr), y_tr)
    #         accuracy = metrics.accuracy_score(y_te, classifier.predict(std_scale.transform(x_te)))
        
    #         f.write("union_of_all_signature_proteins,"+str(accuracy)+","
    #         +str(len(union_of_all_signature_proteins)))
    #         for gene in union_of_all_signature_proteins:
    #             f.write(","+gene)
    #         f.write("\n")

        
        # if len(union_of_all_independent_proteins) > 2:
        #     newdataset5 = newdataset1.get_sub_dataset([gene for gene in union_of_all_independent_proteins])           

        #     filename = evaluate_genes(dataset=newdataset5, min_accuracy=static_min_accuracy, n=n, k=kinner, min_group_size=1, max_group_size=3,classifiers_names=classifiers_names)
            
            
        #     possible_edges = []
        #     with open(filename, 'r') as csv_file:
        #         reader = csv.reader(csv_file, delimiter=",")
        #         reader.next()                
        #         f.write("signature,acc,n,signatures formed by union_of_all_independent_proteins crosvalidated externally\n")            
        #         for row in reader:
        #             f.write(","+str(row[1])+",")
        #             genes = []
        #             for col in range(2,len(row)):
        #                 genes.append(row[col])
        #             f.write(str(len(genes)))
        #             for gene in genes:
        #                 f.write(","+gene)
        #             f.write("\n")

        #             if len(genes) == 3:
        #                 edge0 = PossibleEdge(genes[0],genes[1]) #out 2
        #                 edge1 = PossibleEdge(genes[1],genes[2]) #out 0                        
        #                 edge2 = PossibleEdge(genes[0],genes[2]) #out 1
                        
        #                 if edge0 not in possible_edges:
        #                     possible_edges.append(edge0)
        #                 else:
        #                     possible_edges[possible_edges.index(edge0)].increment_count()

        #                 if edge1 not in possible_edges:
        #                     possible_edges.append(edge1)
        #                 else:
        #                     possible_edges[possible_edges.index(edge1)].increment_count()

        #                 if edge2 not in possible_edges:
        #                     possible_edges.append(edge2)
        #                 else:
        #                     possible_edges[possible_edges.index(edge2)].increment_count()                    

        #     with open('./results/combinations/possible_interactions_union_of_all_independent_proteins.csv', 'w') as f2:
        #         f2.write("source,target,weight\n")
        #         for edge in possible_edges:
        #             f2.write(edge.source+","+edge.target+","+str(edge.count)+"\n")
        #         f2.close
        # f.close()