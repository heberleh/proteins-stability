
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20

@author: Henry
"""

from evaluation import *
from numpy import unique
from multiprocessing import Pool, Manager, Lock, log_to_stderr, cpu_count
import logging


def paths_search(graph, x, y, max_breadth, coming_path, classifier, node, level, n,
                 k, acc_hash, prev_acc, worst, max_worst):

    # print level, max_breadth, node, coming_path
    if level < max_breadth and node not in coming_path:
        path = coming_path[:]  # make a copy/clone of the path, to 'store' more paths. It is the fastest way to clone.
        path.append(node)
        # print "coming path:", coming_path
        # print "path:", path
        # Evaluate this new path
        evaluation = EvaluatedGenesList(x=x,
                                        y=y,
                                        genes=path,
                                        classifier=classifier,
                                        n=n,
                                        k=k,
                                        acc_hash=acc_hash)
        #lock.acquire()
        # print "tried eval to", path, "concluded"
        if evaluation.key in acc_hash.acc:
            # print "eval.key", evaluation.key, " is in acc_hash.acc"
            current_score = acc_hash.acc[evaluation.key]
            # print "current score:", current_score
        else:
            # print "eval.key", evaluation.key, " not in acc_hash.acc"
            current_score = 0
        #lock.release()
        if current_score - prev_acc < 0:  # new score is worst than the last one
            worst += 1
            # print "current score is worst than the last one", prev_acc, current_score
        else:
            # print "current score is better than the last one", prev_acc, current_score
            worst = 0

        # print "worst, max:", worst, max_worst
        if worst < max_worst and level+1 < max_breadth:
            # print "list os neighbors:", graph[node]
            for next_node in graph[node]:
                # print "current neighbor", next_node
                paths_search(graph=graph, x=x, y=y, max_breadth=max_breadth, coming_path=path,
                             classifier=classifier, node=next_node,
                             level=level+1, n=n, k=k, acc_hash=acc_hash, prev_acc=current_score,
                             worst=worst, max_worst=max_worst)


def init_paths_search(args):
    graph = args[0]
    root = args[1]
    shared_list = args[2]
    shared_dict = args[3]
    max_worst = args[4]
    classifier = args[5]
    n = args[6]
    k = args[7]
    x = args[8]
    y = args[9]
    max_breadth = args[10]

    acc_hash = AccuracyHash(shared_dict=shared_dict, min_acc=0.5, shared_list=shared_list)

    paths_search(graph=graph, x=x, y=y, max_breadth=max_breadth, classifier=classifier, coming_path=[],
                        node=root, level=0, n=n, k=k, acc_hash=acc_hash, prev_acc=0,
                        worst=0, max_worst=max_worst)
    return acc_hash


def init(l):
    global lock
    lock = l


def gene_find(x, y, graph, top_genes, classifier_class, n, k):
    """
    :param x: numeric samples matrix
    :param y: numeric classes labels of samples
    :param graph:
    :param top_genes: the root genes of each tree will be constructed
    :param classifier_class: class of the classifier will be used
    :param n: number of repetitions of k-fold cross-validation in each Evaluation
    :param k: for k-fold cross-validation
    :return:
    """
    # calculate N value using number of samples and number of classes

    nk = len(unique(y))
    ns = len(x)

    # max number of genes to train a classifier, with ns samples and nk classes, taking no dimensionality curse.
    max_breadth = (ns/(5*nk))  # -1???   (ns/nk)/breadth > 5 according to Wang


    # for each gene:
    #      call bfs(graph, root, d) in parallel approach
    #      store results

    logger = log_to_stderr()
    logger.setLevel(logging.INFO)

    shared_list = Manager().list()
    shared_dict = Manager().dict()

    l = Lock()

    pool = Pool(processes=cpu_count()/2, initializer=init, initargs=(l,))

    if max_breadth > 5:
        max_worst = max_breadth/2
    else:
        max_worst = 3
    args = [(graph, root, shared_list, shared_dict, max_worst, classifier_class, n, k, x, y, max_breadth) for root in top_genes]

    out = pool.map(init_paths_search, args)
    result = dict(out[0].acc)
    pool.close()
    pool.join()

    return result #return only one copy of dictionary (out is formed by N references to the same shared_dict