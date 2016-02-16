# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20

@author: Henry

"""

from simplified_cross_validation import CrossValidation
from numpy import sum, binary_repr

class AccuracyHash:
    def __init__(self, shared_dict, shared_list, min_acc=0.9):
        self.min_acc = min_acc
        self.acc = shared_dict
        self.shared_list = shared_list

    def store(self, key, score):
        #lock.acquire()
        if key not in self.shared_list:
            self.shared_list.append(key)
            if score >= self.min_acc:
                self.acc[key] = score
        #lock.release()

    def visited(self, key):
        #lock.acquire()
        r = key in self.shared_list
        #lock.release()
        return r

    @staticmethod
    def key(genes):
        return sum([2**i for i in genes])

    @staticmethod
    def get_genes(key):
        binary = binary_repr(key)[::-1]
        genes = []
        for i in range(len(binary)):
            if binary[i] == '1':
                genes.append(i)
        return genes


class EvaluatedGenesList:
    def __init__(self, x, y, genes, classifier, n, k, acc_hash):
        self.key = acc_hash.key(genes)
        if not acc_hash.visited(self.key):
            self.score = CrossValidation(classifier=classifier,
                                         x=x[:, genes],
                                         y=y,
                                         k=k,
                                         n=n).run()
            acc_hash.store(self.key, self.score)
