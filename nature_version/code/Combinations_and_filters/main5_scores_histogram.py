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
            samples_names_list = []
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
                samples_names_list.append(key)
            f.write("\n")

            """
            =============================================================
            Demo of the histogram (hist) function with multiple data sets
            =============================================================

            Plot histogram with multiple sample sets and demonstrate:

                * Use of legend with multiple sample sets
                * Stacked bars
                * Step curve with no fill
                * Data sets of different sample sizes

            Selecting different bin counts and sizes can significantly affect the
            shape of a histogram. The Astropy docs have a great section on how to
            select these parameters:
            http://docs.astropy.org/en/stable/visualization/histogram.html
            """

            import numpy as np
            import matplotlib.pyplot as plt

            n_bins = [0,1,2,3,4,5,6,7]            
            x = []
            # for i in range(len(samples[0])):
            #     x.append([samples[0][i],samples[1][i]])

            x=[samples[0],samples[1]]    
          
            
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), dpi=500)
            ax0, ax1, ax2, ax3 = axes.flatten()

            plt.xticks(n_bins)

            labels = []
            protein = samples_names_list[0]
            #COMB_NDRG1 inner tumor
            for i in range(len(samples_names_list)):
                name = None
                #ITF , inner tumor
                if "invasive" in samples_names_list[i]:
                    name = "ITF"                     
                else:
                    name = "Inner tumor"
                labels.append(name)
                protein = protein.replace("COMB","").replace(" inner tumor","").replace(" invasive tumor","").replace(" invasive stroma","").replace("inner stroma", "").replace("_","").replace(" ","")
                
            colors = ['red', 'tan']#lime
            ax0.hist(x, n_bins,  histtype='bar', align='left', label=labels)
            ax0.legend(prop={'size': 10})
            ax0.set_title(protein +" (bars)")            

            ax1.hist(x, n_bins,  histtype='bar', align='left', label=labels, stacked=True)
            ax1.set_title(protein+' (stacked bars)')
            ax1.legend(prop={'size': 10})

            ax2.hist(x, n_bins, histtype='step', label=labels)
            ax2.set_title(protein+ ' (steps)')
            ax2.legend(prop={'size': 10})

            ax3.hist(x, n_bins, histtype='step', label=labels, stacked=True, fill=False)
            ax3.set_title(protein+' (stacked steps)')
            ax3.legend(prop={'size': 10})

            fig.tight_layout()
            plt.savefig(path_results+protein+'.png', dpi=300)