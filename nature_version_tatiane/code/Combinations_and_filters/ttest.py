# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 00:57:04 2015

@author: Henry
"""

from scipy import stats
import numpy as np


class TTest(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def run(self):
        matrix = self.dataset.matrix.transpose() #we want to compare the genes
        p_values = []
        t_statistics = []
        classes = np.unique(self.dataset.labels)
        if len(classes) > 2:
            raise Exception("Only 2 classes are permited")

        for line in matrix.tolist():
            #devide gene's values into 2 classes (samples)
            sample1 = [line[i] for i in range(len(line)) if self.dataset.labels[i] == classes[0]]
            sample2 = [line[i] for i in range(len(line)) if self.dataset.labels[i] == classes[1]]

            t_statistic, p_value = stats.ttest_ind(np.array(sample1), np.array(sample2),equal_var = True)#(rvs1, rvs3, equal_var = False)
            p_values.append(p_value)
            t_statistics.append(t_statistic)
        return t_statistics, p_values
