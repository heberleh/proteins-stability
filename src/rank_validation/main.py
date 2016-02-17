# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 13:22:02 2015

@author: Henry

This script can be adapted to your problem. It is not ready for multiple datasets. Use only one dataset.

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
from numpy import mean, std
import itertools
import matplotlib.lines as mlines


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
    methods_metrics = {}


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
        methods_metrics[method['id']] = {"accuracy": cross_validation.get_list(metric=cross_validation.ACCURACY),
                                         "precision": cross_validation.get_list(metric=cross_validation.PRECISION),
                                         "f1": cross_validation.get_list(metric=cross_validation.F1),
                                         "recall": cross_validation.get_list(metric=cross_validation.RECALL)}
    return methods_metrics





# =================== Global variables, change as you want =====================
globalN = 5
globalK = 10
scale = True # apply z-score to attributes in the cross-validation class?
normalize = False # apply normalization (0-1) to attributes in the cross-validation class?
classifiers_types = [SVM_linear,SVM_poly, SVM_rbf, NSC, GaussianNaiveBayes, RandomForest, DecisionTree]  # DecisionTree, LinearDA,
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


    #IMPORTANT: this script is not ready for more than 1 dataset.
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
    dataset_res = {}
    for dataset in datasets:
        classifiers_res = {}
        pool = multiprocessing.Pool()
        args = [(c, dataset) for c in classifiers_types]
        for r in args:
            print r
        out = pool.map(calc_stuff, args)
        for i, classifier_type in enumerate(classifiers_types):
            classifiers_res[classifier_type.name] = out[i]
        dataset_res[dataset.name] = classifiers_res

        #to-do save each dataset results

    # ==============================================================================
    print "\n\nTime to finish the complete test:", time.time()-global_start_time, "seconds.\n\n"




    # ===================== Saving classifiers results ==============================
    metrics_list = ["accuracy", "f1", "precision", "recall"]

    for dataset in datasets:
        current_dataset_res = dataset_res[dataset.name]
        header = ["classifier"]
        for method in list_of_ranking_genes:
            header.append(method["id"]+"-mean")
            header.append(method["id"]+"-std")

        for metric in metrics_list:
            matrix = []
            matrix.append(header)
            for classifier_type in classifiers_types:
                c_res = current_dataset_res[classifier_type.name]
                line = [classifier_type.name]
                for method in list_of_ranking_genes:
                    l = c_res[method["id"]][metric]
                    line.append(mean(l))
                    line.append(std(l))
                matrix.append(line)
            matrix = np.array(matrix)
            np.savetxt("./results/crossvalidation/"+metric+"_metric_mean_std_of_"+str(globalN)+"repetitions.csv",np.array(matrix),delimiter=";", fmt="%s")


    # ===============================================================================






    # ====================== Creating Box plots =====================================

    # creates matrix for panda box plot
    metrics_list = ["accuracy", "f1", "precision", "recall"]
    classifiers_names = []
    for classif in classifiers_types:
        classifiers_names.append(classif.name)

    methods_label = []
    for dataset in datasets:
        for metric in metrics_list:
            current_classifier_res = dataset_res[dataset.name]
            label_ready = False
            values_matrix = []
            for classifier_type in classifiers_types:
                current_methods_res = current_classifier_res[classifier_type.name]
                values = []
                for method in list_of_ranking_genes:
                    dict_res = current_methods_res[method['id']]
                    l = dict_res[metric]
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
            plt.savefig("./results/crossvalidation/"+metric+"_mean_std_metric_of_"+str(globalN)+"_repetitions.PNG",dpi=400)
            plt.cla()
            plt.clf()
    # ==============================================================================





    # # ========== Running crossvalidation for top-i proteins ========================
    # localN = 3
    # x_size = 0
    # max_f1_score_N = {}
    # colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    # fill_style = itertools.cycle(itertools.cycle(mlines.Line2D.fillStyles))
    # for classifier_type in classifiers_types:
    #     for method in list_of_ranking_genes:
    #         start_time = time.time()
    #         methods_metrics = {}
    #         x_size = len(method['complete_ranking'])
    #         x_a = range(2,x_size)
    #         y_v = []
    #         for topN in range(2,x_size):
    #             cross_validation = CrossValidation(
    #                                       classifier_type=classifier_type,
    #                                       x=dataset.get_sub_dataset(method['complete_ranking'][0:topN]).matrix,
    #                                       y=dataset.labels,
    #                                       k=globalK, n=localN,
    #                                       scale=scale,
    #                                       normalize=normalize)
    #             cross_validation.run()
    #             y_v.append(mean(cross_validation.get_list(metric=cross_validation.F1)))
    #         print "Execution time of", dataset.name,",", classifier_type.name, ",",method['id'], ":",\
    #                 time.time() - start_time, "seconds"

    #         max_f1_score_N[method["id"]] = y_v.index(np.max(y_v))+2

    #         plt.plot(x_a, y_v, marker='o',markersize=3, markerfacecoloralt='lightgray',linestyle='--', c=colors.next(),fillstyle=fill_style.next(), label=method["id"])

    #     fname = "./results/crossvalidation/"+classifier_type.name + "_f1_mean_in_"+str(localN)+"_repetitions_for_each_ranking.PNG"
    #     plt.xlabel('Top-N proteins')
    #     plt.ylabel('F1 mean')
    #     plt.title('F1 mean values of '+str(localN)+' repetitions for each ranking using '+classifier_type.name+' classifier')
    #     plt.legend()
    #     plt.savefig(fname,dpi=400)
    #     plt.clf()

    # # ==============================================================================







    # # todo
    # # embaralhar as classes
    # #

    # # todo
    # # criar listas de genes aleatórios e de mesmo tamanho das listas encontradas por cada método
    # # plotar de cada método a comparação


    # # todo
    # # gráfico em linha mostrando a acurácia média conforme aumenta o valor de N em cada ranking
    # # atribuir valor a variável ja criada list_of_complete_genes_rankings



    # # ============== Multidimensional Projection Overview ==========================
    # # t-sne projection of samples

    # metrics = ["pearson","euclidean", "pearson_squared"]
    # for dataset in datasets:
    #     for method in list_of_ranking_genes:
    #         for metric in metrics:
    #             max = max_f1_score_N[method["id"]]
    #             cmatrix = dataset.get_sub_dataset(method['complete_ranking'][0:max]).matrix
    #             #print cmatrix
    #             #print
    #             #print cmatrix[0]
    #             #print
    #             #print cmatrix[0,1]
    #             distances = []
    #             #try:
    #             if metric == "pearson":
    #                 distances = pearson_distance(cmatrix)
    #             elif metric == "pearson_squared":
    #                 distances = pearson_squared_distance(cmatrix)
    #             else:
    #                 distances = pairwise_distances(cmatrix.tolist(), metric=metric)

    #             #print distances

    #             t_sne = sklearn.manifold.TSNE(n_components=2, perplexity=20, init='random',
    #                                   metric="precomputed",
    #                                   random_state=7, n_iter=200, early_exaggeration=6,
    #                                   learning_rate=1000)
    #             coordinates = t_sne.fit_transform(distances)

    #             c = pd.factorize(dataset.labels)[0]
    #             categories = np.unique(c)

    #             x = [e[0] for e in coordinates]
    #             y = [e[1] for e in coordinates]


    #             fig = pylab.figure(figsize=(20,20))
    #             ax = fig.add_subplot(111)
    #             ax.set_title("TSNE projection using "+method["id"]+" top-"+str(max)+" proteins and "+metric+" distance",fontsize=12)
    #             ax.grid(True,linestyle='-',color='0.75')
    #             scatter = ax.scatter(x, y, c=c, marker = 'o',
    #                                  cmap=plt.get_cmap('Set1', len(categories)),s=200)
    #             plt.savefig("./results/t-sne_projection/samples_projection_t-sne_with_"+metric+"_dist_and_"+method["id"]+"_top-"+str(max)+"_proteins.pdf")
    #             plt.clf()


    # ============== Multidimensional Projection by best Silhouette score ===================

    # Find the best Silhouette_score of distance matrix to define a the best N value
    metrics = ["euclidean","pearson","pearson_squared"]
    for dataset in datasets:
        for method in list_of_ranking_genes:
            for metric in metrics:
                print "Silhouette score for metric ", metric
                max_silhouette_N = 0
                max_silhouette_score = -1
                selected_dist_matrix = []
                n_silhouette_aux = 0
                distances = []
                for n_silhouette in range(2,len(method['complete_ranking'])):
                    n_silhouette_aux = n_silhouette
                    #print "n_silhouette=", n_silhouette
                    # get submatrix with top-n proteins
                    cmatrix = dataset.get_sub_dataset(method['complete_ranking'][0:n_silhouette]).matrix

                    # computes distance matrix
                    distances = []

                    if metric == "pearson":
                        distances = pearson_distance(cmatrix)
                    elif metric == "pearson_squared":
                        distances = pearson_squared_distance(cmatrix)
                    else:
                        distances = pairwise_distances(cmatrix.tolist(), metric=metric)
                    #print "Calculating silhuette"
                    # computes silhouette score

                    distances = np.array(distances)
                    labels = np.array(dataset.labels)
                    #print distances.shape
                    #print
                    if (not np.any(np.isnan(distances))):
                        sil_score = sklearn.metrics.silhouette_score(distances, labels=labels, metric="precomputed")
                        if sil_score > max_silhouette_score:
                            max_silhouette_N = n_silhouette
                            max_silhouette_score = sil_score
                            selected_dist_matrix = distances
                    else:
                        print "Error: NAN found in distance matrix of size",n_silhouette

                if max_silhouette_N > 0:
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
                    ax.set_title("TSNE projection using "+method["id"]+" top-"+str(max_silhouette_N)+" proteins and "+metric+" distance. Selected by max shilhouette score, equal to "+str(max_silhouette_score),fontsize=12)
                    ax.grid(True,linestyle='-',color='0.75')
                    scatter = ax.scatter(x, y, c=c, marker = 'o',
                                         cmap=plt.get_cmap('Set1', len(categories)),s=200)
                    plt.savefig("./results/t-sne_projection/silhouette_samples_projection_t-sne_with_"+metric+"_dist_and_"+method["id"]+"_top-"+str(max_silhouette_N)+"_proteins_silhouette_"+str(max_silhouette_score)+".pdf")
                    plt.clf()



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
