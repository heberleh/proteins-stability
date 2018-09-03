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


def evaluate_genes(dataset, min_accuracy, n, k, min_group_size, max_group_size, classifiers_names):

    if len(dataset.genes) == 0:
        return False
    elif len(dataset.genes) < max_group_size:
        max_group_size = len(dataset.genes)
    print "\n\n_____________________________________________"
    print "Acurácia mínima:", min_accuracy
    # breaks a k-fold if a minimum acc is not reached...
    # supose 5 folds, considering 4 folds with 100% acc, and we want min_accuracy of 95%
    # the fold must have at least 5*0.95 - 4 = 75% of acc... otherwise the average of acc of all folds would never reach 95% of acc
    min_break_accuracy = (k*min_accuracy) - (k-1)  #may be negative.. so any acc is considered, including zero, that is... all loops of k-fold will be executed

    folds = StratifiedKFold(y=dataset.labels, n_folds=k)    
        

    time_now = str(datetime.now()).replace("-", "_").replace(" ", "__").replace(":", "_").replace(".", "_")

    # fixed folds to calculate accuracy and compare the results between different filters
    # if folds were shuffled every time k-folds runs, each time a 4-gene-set would give a different number of
    # lists of genes with accuracy > min_accuracy
    header = "*N"

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

    groups_file_name = './results/combinations/groups_from_'+str(min_group_size)+'_to_'+str(max_group_size)+'_fold-'+str(fold_i)+'_accuracy_higher_'+str(min_accuracy)+'__'+time_now+'.csv'

    with open(groups_file_name, 'a') as f:
        f.write(header+"\n")
        for sub_list_size in range(min_group_size, max_group_size+1):
            genes_freq =[0 for i in range(len(dataset.genes))]
                        
            print  sub_list_size, "proteins."

            all_possible_groups = itertools.combinations(genes_index, sub_list_size)
            n_possible_groups = nCr(len(genes_index),sub_list_size)

            print "Combination of ",len(genes_index),". There are ", n_possible_groups, "lists to be tested. Estimated time to complete:",\
                n_possible_groups*last_individual_time/60/60, "horas."

            #n_splits = int(n_possible_groups/40)#1000*(3/(sub_list_size*1.0)))
            n_cpu = cpu_count()
            
            n_splits = n_cpu*300
            if (n_splits > n_possible_groups/2):
                if n_possible_groups > 300:
                    n_splits = n_possible_groups/3
            

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
        with open('./results/combinations/genes_freq_size_from_'+str(min_group_size)+'_to_'+str(max_group_size)+'_fold-'+str(fold_i)+'__accuracy_higher_'+str(min_accuracy)+'__'+time_now+'.csv', 'a') as f2:
            f2.write('protein, counting from '+str(len(dataset.genes))+' of being in a group of size from '+str(min_group_size)+' to '+str(max_group_size)+' with acc higher or equal to '+ str(min_accuracy)+'\n')
            for protidx in range(len(genes_freq)):
                f2.write(dataset.genes[protidx] +','+str(genes_freq[protidx])+'\n')
            f2.close()

        delta = time.time()-start
        last_individual_time = delta/float(n_possible_groups)
        del all_possible_groups
        print "The pool of processes took ", delta/60, "minutes to finish the job.\n"
        pool.close()
        pool.join()
        pool.terminate()
        gc.collect()
    return groups_file_name





if __name__ == '__main__':

    start = time.time()

    global x, y, n, k, ns, nk, max_len, min_acc, classifier_class, min_break_accuracy, dataset

    dataset = Dataset("./dataset/train_6_samples_independent.txt", scale=False, normalize=False, sep='\t')

    dataset_test = Dataset("./dataset/test_6_samples_independent.txt", scale=False, normalize=False, sep='\t')


    # Filtering train and test Datasets
    krus = KruskalRankSumTest3Classes(dataset)
    krus_h, krus_p = krus.run()
    cutoff = 0.05

    dataset = dataset.get_sub_dataset([dataset.genes[i] for i in range(len(dataset.genes)) if krus_p[i]<cutoff])

    dataset_test = dataset.get_sub_dataset([dataset_test.genes[i] for i in range(len(dataset_test.genes)) if krus_p[i]<cutoff])

    print "Genes with Kruskal < ", str(cutoff),": ", dataset.genes



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
    static_min_accuracy =  0.8182
    #min_accuracy = 0.8

    n = 1  # n repetitions of k-fold cross validation
    k = 4  # k-fold cross validations - outer
    kinner = 3
    max_group_size = 2


    accuracy_list = []

    all_signatures = []

    # para cada volta i... armazena assinaturas e acurácias
    outer_folds = StratifiedKFold(y=dataset.labels, n_folds=k)

    
    
    with open('./results/combinations/kfold_test_with_all_signature_genes.csv', 'w') as f:
        f.write("N, acc independent, all proteins found in a loop\n")
        f.close()
    
    with open('./results/combinations/kfold_test_with_all_independent_genes.csv', 'w') as f:
        f.write("N, acc independent, all proteins found in a loop\n")
        f.close()

    fold_i = -1


    kfold_independent_proteins = {}
    for i in range(k):
        kfold_independent_proteins[i] = set()

    
    kfold_signature_proteins = {}
    for i in range(k):
        kfold_signature_proteins[i] = set()   

    for train_index, test_index in outer_folds:     

        fold_i += 1

        print "\n\n==============================="
        print "Next train set"

        # encontra assinaturas de tamanho 1
        newdataset1 = dataset.get_sub_dataset_by_samples(train_index)        
        test_dataset1 = dataset.get_sub_dataset_by_samples(test_index)

        x_train = newdataset1.matrix  # data matrix
        y_train = factorize(newdataset1.labels)[0]  # classes/labels of each sample from matrix x

        x_test = test_dataset1.matrix
        y_test = factorize(test_dataset1.labels)[0]
                
        # ----------------------------------
        filename = evaluate_genes(dataset=newdataset1, min_accuracy=static_min_accuracy, n=n, k=kinner, min_group_size=1, max_group_size=1,classifiers_names=classifiers_names)        

        # *N,acc_tree,groups
        # 1,0.833333333333,CON__P00761
        # 1,0.833333333333,P02769_BSA        
        independent_genes1_with_high_acc = set()
        maxAcc = static_min_accuracy


        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            reader.next()
            for row in reader:            
                if float(row[1]) > maxAcc:
                    maxAcc = float(row[1])
            csv_file.close()
        
        print "New max acc ", maxAcc
        
        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            reader.next()
            for row in reader:
                if float(row[1]) == maxAcc:
                    independent_genes1_with_high_acc.add(row[2])
            csv_file.close()
       

        # ----------------------------------
        newdataset2 = newdataset1.get_sub_dataset([gene for gene in newdataset1.genes if (gene not in independent_genes1_with_high_acc)])
       

        filename = evaluate_genes(dataset=newdataset2, min_accuracy=maxAcc, n=n, k=kinner, min_group_size=2, max_group_size=2,classifiers_names=classifiers_names)

        independent_genes2_with_high_acc = set()
        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            reader.next()
            for row in reader:
                if  float(row[1]) > maxAcc:
                    maxAcc = float(row[1])
            csv_file.close()
        
        print "New max acc ", maxAcc

        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            reader.next()
            for row in reader:
                if float(row[1]) == maxAcc:
                    independent_genes2_with_high_acc.add(row[2])
                    independent_genes2_with_high_acc.add(row[3])
            csv_file.close()


        # ----------------------------------
        # encontra assinaturas de tamanho 3 com acc > 0.8182, removendo-se genes que conseguem separar as classses sozinhos ou em duplas (já excluídos os sozinhos) com acc > 0.8182
        
        newdataset3 = newdataset2.get_sub_dataset([gene for gene in newdataset2.genes if  (gene not in independent_genes1_with_high_acc)]) # (gene not in independent_genes2_with_high_acc) and

        filename = evaluate_genes(dataset=newdataset3, min_accuracy=maxAcc+0.01, n=n, k=kinner, min_group_size=3, max_group_size=3,classifiers_names=classifiers_names)

        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            reader.next()
            for row in reader:
                if float(row[1]) > maxAcc:
                    maxAcc = float(row[1])
            csv_file.close()
        
        print "New max acc ", maxAcc

        independent_genes3_with_high_acc = set()
        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            reader.next()
            for row in reader:
                if float(row[1]) == maxAcc:
                    independent_genes3_with_high_acc.add(row[2])
                    independent_genes3_with_high_acc.add(row[3])
                    independent_genes3_with_high_acc.add(row[4])
            csv_file.close()


        # ----------------------------------
        # encontra assinaturas de tamanho 3 com acc > 0.8182, removendo-se genes que conseguem separar as classses sozinhos ou em duplas (já excluídos os sozinhos) com acc > 0.8182
        independent_genes = independent_genes1_with_high_acc | independent_genes2_with_high_acc
        independent_genes = independent_genes | independent_genes3_with_high_acc

        kfold_independent_proteins[fold_i] = kfold_independent_proteins[fold_i] | independent_genes

        print "All IndependentGenes", independent_genes

        if len(independent_genes) > 0:
            genes_index = []
            for i in range(len(dataset.genes)):
                for gene in independent_genes:
                    if dataset.genes[i] == gene:
                        genes_index.append(i)

            x_tr = x_train[:,genes_index]  # data matrix
            y_tr = y_train
            x_te = x_test[:,genes_index]
            y_te = y_test

            classifier = tree.DecisionTreeClassifier()
            # standardize attributes to mean 0 and desv 1 (z-score)
            std_scale = preprocessing.StandardScaler().fit(x_tr)
            classifier.fit(std_scale.transform(x_tr), y_tr)
            accuracy = metrics.accuracy_score(y_te, classifier.predict(std_scale.transform(x_te)))  
            
            with open('./results/combinations/kfold_test_with_all_independent_genes.csv', 'a') as f:
                f.write(str(len(independent_genes))+","+str(accuracy))
                for gene in independent_genes:
                    f.write(","+gene)
                f.write("\n")
                f.close()


#-------------------------------------------------------- try to create minimum signatures with "more independent" proteins...
        newdataset4 = newdataset1
        if len(independent_genes) > 0:
            newdataset4 = newdataset1.get_sub_dataset([gene for gene in independent_genes])

        filename = evaluate_genes(dataset=newdataset4, min_accuracy=maxAcc, n=n, k=kinner, min_group_size=1, max_group_size=3,classifiers_names=classifiers_names)

        all_signature_genes = set()
        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            reader.next()
            print "Signatures: "            
            for row in reader:
                acc = row[1]
                genes = []
                for col in range(2,len(row)):
                    genes.append(row[col])
                    all_signature_genes.add(row[col])
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

                classifier = tree.DecisionTreeClassifier()
                # standardize attributes to mean 0 and desv 1 (z-score)
                std_scale = preprocessing.StandardScaler().fit(x_tr)
                classifier.fit(std_scale.transform(x_tr), y_tr)
                accuracy = metrics.accuracy_score(y_te, classifier.predict(std_scale.transform(x_te)))

                found = False
                for sig in all_signatures:
                    if sig == signature:
                        sig.add_acc(acc)
                        signature.add_independent_test_acc(accuracy)
                        found = True
                        break
                if not found:
                    signature.add_acc(acc)
                    signature.add_independent_test_acc(accuracy)
                    all_signatures.append(signature)
            
        
            if len(all_signature_genes) > 0:
                genes_index = []
                for i in range(len(dataset.genes)):
                    for gene in all_signature_genes:
                        if dataset.genes[i] == gene:
                            genes_index.append(i)

                x_tr = x_train[:,genes_index]  # data matrix
                y_tr = y_train
                x_te = x_test[:,genes_index]
                y_te = y_test

                classifier = tree.DecisionTreeClassifier()
                # standardize attributes to mean 0 and desv 1 (z-score)
                std_scale = preprocessing.StandardScaler().fit(x_tr)
                classifier.fit(std_scale.transform(x_tr), y_tr)
                accuracy = metrics.accuracy_score(y_te, classifier.predict(std_scale.transform(x_te)))  
                
                with open('./results/combinations/kfold_test_with_all_signature_genes.csv', 'a') as f:
                    f.write(str(len(all_signature_genes))+","+str(accuracy))
                    for gene in all_signature_genes:
                        f.write(","+gene)
                    f.write("\n")
                    f.close()
            csv_file.close()
        kfold_signature_proteins[fold_i] = kfold_signature_proteins[fold_i] | all_signature_genes
   
    
    print "\n\n Time to complete the algorithm", (time.time() - start)/60, "minutes."

    with open('./results/combinations/signatures_combinations.csv', 'w') as f:
        f.write("n,avg_acc,stddev_acc,ind_avg_acc,ind_stddev_acc,weight,signature\n")       
        for signature in all_signatures:
            print "accs", signature.get_independent_test_accs()
            mean_acc=0.0
            std_acc=0.0
            mean_independent_acc=0.0
            std_independent_acc=0.0
            if signature.get_weight() == 1: 
                mean_acc= signature.get_accs()[0]
                std_acc=0
                mean_independent_acc=signature.get_independent_test_accs()[0]
                std_independent_acc=0
            else:
                mean_acc=np.mean(signature.get_accs())
                std_acc=np.std(signature.get_accs())
                mean_independent_acc=np.mean(signature.get_independent_test_accs())
                std_independent_acc=np.std(signature.get_independent_test_accs()) 

            f.write(
                str(signature.get_n())+","+
                str(mean_acc)+","+
                str(std_acc)+","+
                str(mean_independent_acc)+","+
                str(std_independent_acc)+","+
                str(signature.get_weight()))

            for gene in signature.genes:
                f.write(","+gene)
            f.write("\n")
        f.close()
    
    
    intersection_of_all_independent_proteins = kfold_independent_proteins[0]    
    union_of_all_independent_proteins = set()
    for key in kfold_independent_proteins:        

        union_of_all_independent_proteins = union_of_all_independent_proteins | kfold_independent_proteins[key]

        intersection_of_all_independent_proteins = intersection_of_all_independent_proteins & kfold_independent_proteins[key]


    intersection_of_all_signature_proteins = kfold_signature_proteins[0]    
    union_of_all_signature_proteins = set()
    for key in kfold_signature_proteins:        

        union_of_all_signature_proteins = union_of_all_signature_proteins | kfold_signature_proteins[key]

        intersection_of_all_signature_proteins = intersection_of_all_signature_proteins & kfold_signature_proteins[key]


    with open('./results/combinations/independent_test_set.csv', 'w') as f:
        f.write("type,acc,n,proteins\n")  
    
        # test independent dataset
        
        #__________________________________________________
        # with intersection of "most independent proteins"
        if len(intersection_of_all_independent_proteins) > 0:
            genes_index = []
            for i in range(len(dataset.genes)):
                for gene in intersection_of_all_independent_proteins:
                    if dataset.genes[i] == gene:
                        genes_index.append(i)

            x_tr = x_train[:,genes_index]  # data matrix
            y_tr = y_train
            x_te = x_test[:,genes_index]
            y_te = y_test

            classifier = tree.DecisionTreeClassifier()
            # standardize attributes to mean 0 and desv 1 (z-score)
            std_scale = preprocessing.StandardScaler().fit(x_tr)
            classifier.fit(std_scale.transform(x_tr), y_tr)
            accuracy = metrics.accuracy_score(y_te, classifier.predict(std_scale.transform(x_te)))
            
            f.write("intersection_of_all_independent_proteins,"+str(accuracy)+","
            +str(len(intersection_of_all_independent_proteins)))
            for gene in intersection_of_all_independent_proteins:
                f.write(","+gene)
            f.write("\n")
        
        
        #__________________________________________________
        # with union of "most independent proteins"
        if len(union_of_all_independent_proteins) > 0:
            genes_index = []
            for i in range(len(dataset.genes)):
                for gene in union_of_all_independent_proteins:
                    if dataset.genes[i] == gene:
                        genes_index.append(i)

            x_tr = x_train[:,genes_index]  # data matrix
            y_tr = y_train
            x_te = x_test[:,genes_index]
            y_te = y_test

            classifier = tree.DecisionTreeClassifier()
            # standardize attributes to mean 0 and desv 1 (z-score)
            std_scale = preprocessing.StandardScaler().fit(x_tr)
            classifier.fit(std_scale.transform(x_tr), y_tr)
            accuracy = metrics.accuracy_score(y_te, classifier.predict(std_scale.transform(x_te)))

            f.write("union_of_all_independent_proteins,"+str(accuracy)+","
            +str(len(union_of_all_independent_proteins)))
            for gene in union_of_all_independent_proteins:
                f.write(","+gene)
            f.write("\n")

        #__________________________________________________
        # with intersection of "union of signature proteins"
        if len(intersection_of_all_signature_proteins) > 0:
            genes_index = []
            for i in range(len(dataset.genes)):
                for gene in intersection_of_all_signature_proteins:
                    if dataset.genes[i] == gene:
                        genes_index.append(i)

            x_tr = x_train[:,genes_index]  # data matrix
            y_tr = y_train
            x_te = x_test[:,genes_index]
            y_te = y_test

            classifier = tree.DecisionTreeClassifier()
            # standardize attributes to mean 0 and desv 1 (z-score)
            std_scale = preprocessing.StandardScaler().fit(x_tr)
            classifier.fit(std_scale.transform(x_tr), y_tr)
            accuracy = metrics.accuracy_score(y_te, classifier.predict(std_scale.transform(x_te)))
            
            f.write("intersection_of_all_signature_proteins,"+str(accuracy)+","
            +str(len(intersection_of_all_signature_proteins)))
            for gene in intersection_of_all_signature_proteins:
                f.write(","+gene)
            f.write("\n")        

        #__________________________________________________
        # with union of "union of signature proteins"
        if len(union_of_all_signature_proteins) > 0:
            genes_index = []
            for i in range(len(dataset.genes)):
                for gene in union_of_all_signature_proteins:
                    if dataset.genes[i] == gene:
                        genes_index.append(i)

            x_tr = x_train[:,genes_index]  # data matrix
            y_tr = y_train
            x_te = x_test[:,genes_index]
            y_te = y_test

            classifier = tree.DecisionTreeClassifier()
            # standardize attributes to mean 0 and desv 1 (z-score)
            std_scale = preprocessing.StandardScaler().fit(x_tr)
            classifier.fit(std_scale.transform(x_tr), y_tr)
            accuracy = metrics.accuracy_score(y_te, classifier.predict(std_scale.transform(x_te)))
        
            f.write("union_of_all_signature_proteins,"+str(accuracy)+","
            +str(len(union_of_all_signature_proteins)))
            for gene in union_of_all_signature_proteins:
                f.write(","+gene)
            f.write("\n")

        
        # if len(intersection_of_all_independent_proteins) > 2:
        #     newdataset5 = newdataset1.get_sub_dataset([gene for gene in intersection_of_all_independent_proteins])           

        #     filename = evaluate_genes(dataset=newdataset5, min_accuracy=static_min_accuracy, n=n, k=kinner, min_group_size=1, max_group_size=3,classifiers_names=classifiers_names)
            
            
        #     possible_edges = []
        #     with open(filename, 'r') as csv_file:
        #         reader = csv.reader(csv_file, delimiter=",")
        #         reader.next()                
        #         f.write("signature,acc,n,signatures formed by intersection_of_all_independent_proteins crosvalidated externally\n")            
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

        #     with open('./results/combinations/possible_interactions_intersection_of_all_independent_proteins.csv', 'w') as f2:
        #         f2.write("source,target,weight\n")
        #         for edge in possible_edges:
        #             f2.write(edge.source+","+edge.target+","+str(edge.count)+"\n")
        #         f2.close()

        if len(union_of_all_independent_proteins) > 2:
            newdataset5 = newdataset1.get_sub_dataset([gene for gene in union_of_all_independent_proteins])           

            filename = evaluate_genes(dataset=newdataset5, min_accuracy=static_min_accuracy, n=n, k=kinner, min_group_size=1, max_group_size=3,classifiers_names=classifiers_names)
            
            
            possible_edges = []
            with open(filename, 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=",")
                reader.next()                
                f.write("signature,acc,n,signatures formed by union_of_all_independent_proteins crosvalidated externally\n")            
                for row in reader:
                    f.write(","+str(row[1])+",")
                    genes = []
                    for col in range(2,len(row)):
                        genes.append(row[col])
                    f.write(str(len(genes)))
                    for gene in genes:
                        f.write(","+gene)
                    f.write("\n")

                    if len(genes) == 3:
                        edge0 = PossibleEdge(genes[0],genes[1]) #out 2
                        edge1 = PossibleEdge(genes[1],genes[2]) #out 0                        
                        edge2 = PossibleEdge(genes[0],genes[2]) #out 1
                        
                        if edge0 not in possible_edges:
                            possible_edges.append(edge0)
                        else:
                            possible_edges[possible_edges.index(edge0)].increment_count()

                        if edge1 not in possible_edges:
                            possible_edges.append(edge1)
                        else:
                            possible_edges[possible_edges.index(edge1)].increment_count()

                        if edge2 not in possible_edges:
                            possible_edges.append(edge2)
                        else:
                            possible_edges[possible_edges.index(edge2)].increment_count()                    

            with open('./results/combinations/possible_interactions_union_of_all_independent_proteins.csv', 'w') as f2:
                f2.write("source,target,weight\n")
                for edge in possible_edges:
                    f2.write(edge.source+","+edge.target+","+str(edge.count)+"\n")
                f2.close
        f.close()