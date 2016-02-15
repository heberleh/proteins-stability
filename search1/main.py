# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20

@author: Henry
"""

from search import gene_find
from pandas import factorize
from dataset import Dataset
from graph import CoExpressionGraph
from wilcoxon import WilcoxonRankSumTest
from search import init_paths_search
from sklearn import svm
from numpy import max
from numpy import unique
from multiprocessing import Pool, Manager, Lock, log_to_stderr
import logging

import time


if __name__ == '__main__':
    dataset = Dataset("../spectral.csv", scale=True)

    wil = WilcoxonRankSumTest(dataset)
    z, p = wil.run()

    wil_genes = [dataset.genes[i] for i in range(len(p)) if p[i] < 0.05]
    subdataset = dataset.get_sub_dataset(wil_genes)

    coexp = CoExpressionGraph(subdataset.matrix, -1)
    graph = coexp.get_graph()

    top_genes = [i for i in range(len(subdataset.matrix))]

    start = time.time()
    d = gene_find(x=subdataset.matrix,
              y=factorize(subdataset.labels)[0],
              graph=graph,
              top_genes=top_genes,
              classifier_class=svm.LinearSVC,
              n=3,
              k=10)

    print "10 fold, 3 rep, all genes connected, p<0.05, breadth=6"
    print "\n\n Time to finish the algorithm", time.time() - start, "seconds."

    max_acc = max(d.values())
    delta = 0.03
    print
    print
    print "Sets with accuracy [", max_acc-delta, ",", max_acc, "]"
    print
    print
    selected_sublists = []
    for key in d.keys():
        if d[key] > max_acc-delta:
            selected_sublists.append(key)

    import evaluation
    for key in selected_sublists:
        index = evaluation.AccuracyHash.get_genes(key)
        print [subdataset.genes[i] for i in index]



