# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 13:22:02 2015

@author: Henry

This script can be adapted to your problem.
It will run the following main procedures:

    1. N repetition of K-fold Cross validation for M classifiers
       types for each rank and defined number of proteins.
    2. A plot for each Classifier type, comparing the mean accuracy and standard deviation.

    3. Calculate a K-fold cross validation for top-i proteins for each rank, for i in [2, max]
    4. Plot the results of 3.

    5. For each rank, for each distance metric, and defined number of proteins,
        project the samples using t-sne technique. Define the silhuette in distance matrix and
        inform it on the plot.

    6. Search the max. accuracy found in 3. and project using the respective rank and top-i proteins,
        for each  distance metric.




Usual parameters to change:
    - when loading data, you can set True or False for std (z-score) - DataSet class
    - when creating the object CrossValidation() you can set:
            - the K of K-fold crossvalidation (look for variable globalK)
            - the N of N repetitions of K-fold (look for variable globalN)
"""


from wilcoxon import WilcoxonRankSumTest
import time
import sys
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances
from sklearn import manifold
import sklearn
import pylab
import numpy as np
from classifier import *
from crossvalidation import CrossValidation
from dataset import Dataset
import csv


def pearson(matrix):
    """
    Computes the pearson correlation(similarity) between all lines of the matrix
    :param matrix: numeric matrix, each line correspond to a sample
    :return: a distance matrix of n samples per n samples
    """

    rows, cols = matrix.shape[0], matrix.shape[1]
    r = np.ones(shape=(rows, rows))
    p = np.ones(shape=(rows, rows))
    matrix = matrix.tolist()
    for i in range(rows):
        for j in range(i+1, rows):
            r_, p_ = pearsonr(matrix[i], matrix[j])
            r[i, j] = r[j, i] = r_
            p[i, j] = p[j, i] = p_
    return r, p


def pearson_distance(matrix):
    """
    Computes the pearson distance between all lines of the matrix
    :param matrix: numeric matrix, each line correspond to a sample
    :return: a distance matrix of n samples per n samples
    """
    r, p = pearson(matrix)
    ones = np.ones(r.shape)
    distances = ones - np.array(r)
    return distances


def pearson_squared_distance(matrix):
    """
    Computes the pearson squared distance between all lines of the matrix
    :param matrix: numeric matrix, each line correspond to a sample
    :return: a distance matrix of n samples per n samples
    """
    r, p = pearson(matrix)
    ones = np.ones(r.shape)
    distances = ones - (np.array(r)**2)
    return distances


def calc_stuff(args):
    """ Computes the k-fold crossvalidation for the multiprocess map function.
    :param args: must be in this order: classifier_type, dataset, list_of_ranking_genes, k, n, scale, normalize
    :return: accuracy dictionary for N repetitions, for each classifier method, for each k-fold
    """
    global list_of_ranking_genes
    global scale
    global normalize
    global globalK
    global globalN
    classifier_type, dataset = args[0], args[1]
    methods_acc = {}


    #print scale, normalize, globalN, globalK, classifier_type
    #print dataset.labels

    for method in list_of_ranking_genes:
        #print method
        start_time = time.time()
        cross_validation = CrossValidation(
                                  classifier_type=classifier_type,
                                  x=dataset.get_sub_dataset(method['genes']).matrix,
                                  y=dataset.labels,
                                  k=globalK, n=globalN,
                                  scale=scale,
                                  normalize=normalize)
        cross_validation.run()
        print "Execution time of", dataset.name,",", classifier_type.name, ",",method['id'], ":",\
            time.time() - start_time, "seconds"
        l = cross_validation.get_list(metric=cross_validation.ACCURACY)
        methods_acc[method['id']] = l
    return methods_acc





# =================== Global variables, change as you want =====================
box_plot_file_name = "./results/crossvalidation/teste_multi_processos_zscore_3rep_10-foldcross.PNG"
globalN = 5
globalK = 10
scale = True # apply z-score to attributes in the cross-validation class?
normalize = False # apply normalization (0-1) to attributes in the cross-validation class?
classifiers_types = [SVM_linear,SVM_poly, SVM_rbf, NSC,  GaussianNaiveBayes]  # DecisionTree, LinearDA,
# RandomForest, AdaBoost]  #MultinomialNaiveBayes, (non-negative...)




# =========== Loading list os biomarkers candidates to test ====================
# load selected genes from files ... TODO
list_of_ranking_genes = []#[svmrfe, ttest, nsc, wilcoxon_genes, little1, little2]
# ==============================================================================


if __name__ == '__main__':  # freeze_support()

    # =================== Loading Datasets =========================================
    # para cada pasta dentro da pasta datasets
    # seleciona nome da pasta como ID (breast_cancer)
    # cria o tipo Dataset (leitura do conjunto de dados) com o arquivo dentro e que começa com a palavra dataset
    # mapeia o conjunto de dados com o ID (nome do arquivo) em um dicionário

    # organização das pastas
    # datasets/breast_cancer/dataset_wang.csv
    # datasets/breast_cancer/selected_genes/svm-rfe_wang.csv
    # datasets/breast_cancer/selected_genes/nsc_wang.csv

    datasets_names = ["./dataset/current/train.txt"]
    datasets = []
    for name in datasets_names:
        data = Dataset(name, scale=False, normalize=False, sep='\t')
        datasets.append(data)
        if (len(data.levels()) == 2):
            wil = WilcoxonRankSumTest(data)
            z,p = wil.run()
            print "\n\n\nThe dataset", name, "has", len([i for i in range(len(p)) if p[i] < 0.05]), \
                "differential expression genes with p < 0.05 for Wilcoxon test.\n\n\n"
    # ==============================================================================


    # ============ Reading rankings =================================================
    rankings = []
    with open("./dataset/current/rankings.txt", 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter='\t')
        for row in reader:
            for i in range(len(row)):
                row[i] = row[i].replace(' ','')
            rankings.append(row)
    rankings = np.array(rankings)
    rankings_names = rankings[0,].tolist()
    rankings_N = rankings[1,].astype(int).tolist()
    rankings_genes = rankings[2:,]

    for i in range(len(rankings_names)):
        r_name = rankings_names[i]
        r_n = rankings_N[i]
        ranking = rankings_genes[:,i].reshape(len(rankings_genes.tolist()))
        ranking_node = dict()
        ranking_node["id"] = r_name
        ranking_node["N"] = r_n
        ranking_node["genes"] = ranking[0:r_n]  #top-n genes
        ranking_node["complete_ranking"] = ranking.tolist()
        list_of_ranking_genes.append(ranking_node)
    # ==============================================================================



    # =========== Computing the tests ==============================================

    global_start_time = time.time()
    acc_list = {}
    for dataset in datasets:
        classifiers_acc = {}
        pool = multiprocessing.Pool()
        args = [(c, dataset) for c in classifiers_types]
        for r in args:
            print r
        out = pool.map(calc_stuff, args)
        for i, classifier_type in enumerate(classifiers_types):
            classifiers_acc[classifier_type.name] = out[i]
        acc_list[dataset.name] = classifiers_acc

        #to-do save each dataset results

    # ==============================================================================
    print "\n\nTime to finish the complete test:", time.time()-global_start_time, "seconds.\n\n"





    # ====================== Creating Box plots =====================================

    # creates matrix for panda box plot
    classifiers_names = []
    for classif in classifiers_types:
        classifiers_names.append(classif.name)

    methods_label = []
    for dataset in datasets:
        current_classifier_res = acc_list[dataset.name]
        label_ready = False
        values_matrix = []
        for classifier_type in classifiers_types:
            current_methods_res = current_classifier_res[classifier_type.name]
            values = []
            for method in list_of_ranking_genes:
                l = current_methods_res[method['id']]
                size = len(l)
                values += l
                if not label_ready:
                    methods_label += size*[method['id']]
            label_ready = True
            values_matrix.append(values)

        box_plot_matrix = np.matrix(values_matrix).transpose()
        y_min = np.min(box_plot_matrix) - 0.02
        df = pd.DataFrame(box_plot_matrix, columns=classifiers_names)
        df['Genes List'] = pd.Series(methods_label)
        pd.options.display.mpl_style = 'default'

        df.boxplot(by='Genes List')

        fig = plt.gcf()
        plt.ylim(y_min, 1.02)
        fig.set_size_inches(15,15)
        plt.savefig(box_plot_file_name,dpi=400)
    # ==============================================================================

    # todo
    # embaralhar as classes
    #

    # todo
    # criar listas de genes aleatórios e de mesmo tamanho das listas encontradas por cada método
    # plotar de cada método a comparação


    # todo
    # gráfico em linha mostrando a acurácia média conforme aumenta o valor de N em cada ranking
    # atribuir valor a variável ja criada list_of_complete_genes_rankings





    # ============== Multidimensional Projection Overview ==========================
    # t-sne projection of samples
    metrics = ["pearson","euclidean", "pearson_squared"]
    for dataset in datasets:
        for method in list_of_ranking_genes:
            for metric in metrics:
                cmatrix = dataset.get_sub_dataset(method['genes']).matrix
                print cmatrix
                print
                print cmatrix[0]
                print
                print cmatrix[0,1]
                distances = []
                #try:
                if metric == "pearson":
                    distances = pearson_distance(cmatrix)
                elif metric == "pearson_squared":
                    distances = pearson_squared_distance(cmatrix)
                else:
                    distances = pairwise_distances(cmatrix.tolist(), metric=metric)

                #print distances

                t_sne = sklearn.manifold.TSNE(n_components=2, perplexity=20, init='random',
                                      metric="precomputed",
                                      random_state=7, n_iter=200, early_exaggeration=6,
                                      learning_rate=1000)
                coordinates = t_sne.fit_transform(distances)

                c = pd.factorize(dataset.labels)[0]
                categories = np.unique(c)

                x = [e[0] for e in coordinates]
                y = [e[1] for e in coordinates]

                fig = pylab.figure(figsize=(20,20))
                ax = fig.add_subplot(111)
                ax.set_title("TSNE projection using "+method["id"]+" selected proteins and "+metric+" distance",fontsize=12)
                ax.grid(True,linestyle='-',color='0.75')
                scatter = ax.scatter(x, y, c=c, marker = 'o',
                                     cmap=plt.get_cmap('Set1', len(categories)),s=200)
                plt.savefig("./results/t-sne_projection/samples_projection_t-sne_with_"+metric+"_dist_and_"+method["id"]+"_selected_proteins.pdf")
                #except:
                #    print "Unexpected error:", sys.exc_info()[0]
                # tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=[ids[i] for i in unknown_index])
                # mpld3.plugins.connect(fig, tooltip)

# global_start_time = time.time()
# acc_list = {}
# for dataset in datasets:
#     classifiers_acc = {}
#     for classifier_type in classifiers_types:
#         methods_acc = {}
#         for method in list_of_ranking_genes:
#             start_time = time.time()
#             crossvalidation = CrossValidation(
#                                        classifier_type=classifier_type,
#                                       x=dataset.get_sub_dataset(method['genes']),
#                                       genes=method['genes'],
#                                       y=dataset.labels,
#                                       k=globalK, n=globalN,
#                                       scale=scale,
#                                       normalize=normalize)
#             crossvalidation.run()
#            # print "Executou com", dataset.name,",", classifier_type.name, ",",method['id'], "em",
# # time.time() - start_time, "segundos"
#             l = crossvalidation.get_list(metric=crossvalidation.ACURACY_MEAN)
#             methods_acc[method['id']] = l
#         classifiers_acc[classifier_type.name] = methods_acc
#     acc_list[dataset.name] = classifiers_acc
#
# print "\n\nTempo para rodar o teste completo", time.time()-global_start_time, "segundos.\n\n"
