# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 00:57:04 2015

@author: Henry
"""

from scipy.stats import mstats
import numpy as np


class KruskalRankSumTest3Classes(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def run(self):
        matrix = self.dataset.matrix.transpose() #we want to compare the genes
        p_values = []
        h_statistics = []
        classes = np.unique(self.dataset.labels)
        if len(classes) != 3:
            raise Exception("This implementation is for 3 classes.")

        for line in matrix.tolist():
            #devide gene's values into 2 classes (samples)
            sample1 = [line[i] for i in range(len(line)) if self.dataset.labels[i] == classes[0]]
            sample2 = [line[i] for i in range(len(line)) if self.dataset.labels[i] == classes[1]]
            sample3 = [line[i] for i in range(len(line)) if self.dataset.labels[i] == classes[2]]

            h, p_value = mstats.kruskalwallis(np.array(sample1), np.array(sample2), np.array(sample3))
            p_values.append(p_value)
            h_statistics.append(h)
        return h_statistics, p_values
