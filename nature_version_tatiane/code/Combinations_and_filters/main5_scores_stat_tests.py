# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20

@author: Henry
"""
import csv
import gc
import itertools
import math
import time
from datetime import datetime
from multiprocessing import Lock, Pool, cpu_count

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from numpy import mean, median, min, std, unique
from pandas import factorize
from pylab import interp, savefig
from sklearn import manifold, metrics, preprocessing, svm, tree
from sklearn.cross_validation import KFold
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import (ElasticNet, LogisticRegression,
                                  MultiTaskElasticNet,
                                  PassiveAggressiveClassifier, Perceptron,
                                  SGDClassifier)
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm import SVC

from dataset import Dataset
from possibleEdge import PossibleEdge
from signature import Signature
from ttest import TTest
from wilcoxon import WilcoxonRankSumTest

from scipy import stats


def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def key(genes):
    return sum([2**i for i in genes])

def split(arr, count):
     return [arr[i::count] for i in range(count)]


if __name__ == '__main__':
    
    path_results = "../../results/"
    path_dataset = "../../dataset/"

    # read data
    filename = path_dataset+"scores_0_6.txt"
    data = []
    alpha = 0.05
    pairs = True

    with open(filename, 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        i = 1        
        first_row = None
        second_row = None
        for row in reader:
            if i % 2 == 0:
                second_row = row
                v1 = []
                v2 = []
                for j in range(1,len(first_row)):
                    v1.append(float(first_row[j]))
                for j in range(1,len(second_row)):
                    v2.append(float(second_row[j]))

                sample = {first_row[0]:v1, second_row[0]:v2}
                data.append(sample)
            else:
                first_row = row
            i += 1
    
    with open(path_results+'scores_0_6_report.txt', 'w') as f:
        report = ""
        table = "samples, test, p-value"
        if pairs:
            table += ", T"
        table+="\n"

        for sample in data:
            f.write("\n\n\n==========================================================\n\n")
            f.write("Report for ")
            c = True
            for key in sample:
                f.write(key) 
                if c:
                    f.write(" against ")
                    c = False       

            f.write("\n\n")            

            ps = []            
            samples = []
            samples_names = ""
            f.write("""Note about the statistical test: Test whether a sample differs from a normal distribution.
This function tests the null hypothesis that a sample comes from a normal distribution. It is based on D’Agostino and Pearson’s test that combines skew and kurtosis to produce an omnibus test of normality.\n\n""")
            c = True
            for key in sample:                
                f.write("Null hypothesis: "+key+" comes from a normal distribution.\n")
                samples_names += key
                if c:
                    samples_names += " against "
                    c = False
                k2, p = stats.normaltest(sample[key])
                ps.append(p)
                if p < alpha:  # null hypothesis: x comes from a normal distribution
                    f.write("The null hypothesis can be rejected.\n\n")
                else:
                    f.write("The null hypothesis cannot be rejected.\n\n")

                samples.append(sample[key])
            f.write("\n")

            p_value = None
            test_name = None
            T = None
            if min(ps) >= alpha and not pairs:
                test_name = "T-test"
                f.write("Selected statistical test: T-test\n")
                f.write("""
Note about the statistical test:
Calculate the T-test for the means of two independent samples of scores.
This is a two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values. This test assumes that the populations have identical variances by default.\n\n""")
                f.write("Means for "+samples_names+": "+"%.2f" % mean(samples[0])+" against "+"%.2f" % mean(samples[1])+"\n")

                t_statistic, p_value = stats.ttest_ind(np.array(samples[0]), np.array(samples[1]),equal_var = True)
            
            elif min(ps) < alpha and not pairs:
                test_name = "Wilcoxon rank-sum test"
                f.write("Selected statistical test: Wilcoxon rank-sum test\n")
                f.write("""
Note about the statistical test:
The Wilcoxon rank-sum test tests the null hypothesis that two sets of measurements are drawn from the same distribution. The alternative hypothesis is that values in one sample are more likely to be larger than the values in the other sample.
This test should be used to compare two samples from continuous distributions. It does not handle ties between measurements in x and y.\n\n""")
                z_statistic, p_value = stats.ranksums(np.array(samples[0]), np.array(samples[1]))

                f.write("Means for "+samples_names+": "+"%.2f" % mean(samples[0])+" against "+"%.2f" % mean(samples[1])+"\n")
            elif min(ps) < alpha and pairs:
                test_name = "Wilcoxon signed-rank test"
                f.write("Selected statistical test: Wilcoxon signed-rank test\n")
                f.write("""
Note about the statistical test:
The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution. In particular, it tests whether the distribution of the differences x - y is symmetric about zero. It is a non-parametric version of the paired T-test.\n\n

Siegel used the symbol T for a value related to, but not the same as, W {\displaystyle W} W. In consequence, the test is sometimes referred to as the Wilcoxon T test, and the test statistic is reported as a value of T.

The original Wilcoxon's proposal used a different statistic. Denoted by Siegel as the T statistic, it is the smaller of the two sums of ranks of given sign; in the example given below, therefore, T would equal 3+4+5+6=18. Low values of T are required for significance. As will be obvious from the example below, T is easier to calculate by hand than W and the test is equivalent to the two-sided test described above; however, the distribution of the statistic under H 0 {\displaystyle H_{0}} H_{0} has to be adjusted.

As demonstrated in the example, when the difference between the groups is zero, the observations are discarded. This is of particular concern if the samples are taken from a discrete distribution. In these scenarios the modification to the Wilcoxon test by Pratt 1959, provides an alternative which incorporates the zero differences.[4][5] This modification is more robust for data on an ordinal scale.[5] (https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)\n\n""")
                T, p_value = stats.wilcoxon(np.array(samples[0]), np.array(samples[1]), zero_method='pratt', correction=False)

                f.write("sum(v1-v2) "+samples_names+": "+"%.4f" % sum(np.array(samples[0])-np.array(samples[1]))+"\n\n")
                f.write("sum(v1): "+"%.4f" % sum(samples[0])+"\n\n")
                f.write("sum(v2): "+"%.4f" % sum(samples[1])+"\n\n")

                f.write("Means for "+samples_names+": "+"%.2f" % mean(samples[0])+" against "+"%.2f" % mean(samples[1])+"\n\n")
                f.write("Medians for "+samples_names+": "+ str(np.array(median(samples[0])))+" against "+ str(median(np.array(samples[1])))+"\n\n")
                f.write("STDs for "+samples_names+": "+"%.2f" % std(samples[0])+" against "+"%.2f" % std(samples[1])+"\n\n")
                f.write("The sum of the ranks of the differences above or below zero, whichever is smaller: ")
                f.write(str(T))
                f.write("\n")

            f.write("The p-value found for " + samples_names + " is "+str(p_value))
            table += samples_names + ", "+ test_name+ ", "+ str(p_value)
            if (pairs):
                table += ", " + str(T)
            table += "\n"
            f.write("\n")
        
        f.write("\n\n\n =============== complete table ============== \n\n")
        f.write(table)

        f.write("\n\n\n =============== considered values ============== \n\n")
        for sample in data:
            for key in sample:
                f.write(key)
                f.write(": ")
                f.write(str(sample[key]))
                f.write("\n")
            f.write("\n")


        f.close()

    # 

    # if p < 0.05:

    # else:
        
    # # Filtering train and test Datasets
    # wil = WilcoxonRankSumTest(dataset)
    # wil_z, wil_p = wil.run()            

    # # Filtering train and test Datasets
    # ttest = TTest(dataset)
    # ttest_t, ttest_p = ttest.run()


    