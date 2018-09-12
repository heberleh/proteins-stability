# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 13:25:12 2015

@author: Henry
"""

from sklearn import preprocessing
from numpy import matrix
import csv
import numpy as np
from pandas import factorize, DataFrame


class Dataset(object):

    def __init__(self, filename=None, scale=False, normalize=False, sep=";"):
        self.genes = None
        self.samples = None
        self.matrix = None
        self.labels = None
        self.__geneIndex = {}


        if filename != None:
            self.name = filename.replace(".csv","")
            self.__load(filename, sep=sep)
        else:
            self.name = ""

        self.scale = scale
        self.normalize = normalize
        self.complete_dataset = None
        if self.normalize:
            self.__normalize()
        if self.scale:
            self.__scale()

    def __load(self, filename, sep=";"):
        self.complete_dataset = []

        with open(filename, 'rb') as csv_file:
            reader = csv.reader(csv_file, delimiter=sep)
            for row in reader:
                for i in range(len(row)):
                    row[i] = row[i].replace(' ','')
                self.complete_dataset.append(row)

        self.complete_dataset = np.matrix(self.complete_dataset)
        self.matrix = np.matrix(self.complete_dataset[2:, 1:]).astype(float).transpose()
        self.setGenes(list(np.array(self.complete_dataset[2:, 0].transpose())[0]))        
        self.samples = list(np.array(self.complete_dataset[0, 1:])[0])
        self.labels = list(np.array(self.complete_dataset[1, 1:])[0])
        print("\n----- dataset reader ---------\n Samples found:" +str(self.samples) +"\n total size: " + str(len(self.samples))+'\n------------------------------\n')
        self.sortSamplesByClassLabel()

    def __normalize(self):
        """ Normalize into range 0-1
        """
        self.matrix = np.matrix(preprocessing.normalize(self.matrix))

    def setGenes(self, genes):
        self.genes = genes
        self.__setGenesIndex()

    def __setGenesIndex(self):
        for index in range(len(self.genes)):
            self.__geneIndex[self.genes[index]] = index

    def get_normalized_data(self):
        return preprocessing.normalize(self.matrix)

    def levels(self):
        return np.unique(self.labels)

    def __scale(self):
        """ Scale using z-score
        """
        self.matrix = np.matrix(preprocessing.scale(self.matrix))

    def get_scaled_data(self):
        return preprocessing.scale(self.matrix)

    def nRow(self):
        return len(np.matrix(self.matrix))

    def get_sub_dataset(self, genes):
        """ Returns a sub dataset containing only the given genes.

        Parameters:
            genes: the genes will appear in the returned dataset.
        """
        indexes = [self.geneIndex(gene) for gene in genes]

        new_dataset = Dataset()
        new_dataset.setGenes(genes)        
        new_dataset.matrix = self.matrix[:, indexes]
        new_dataset.labels = self.labels
        new_dataset.samples = self.samples
        new_dataset.name = self.name+"_modified"
        return new_dataset

    def get_sub_dataset_by_samples(self, samples_indexes):
        """ Returns a sub dataset containing only the given genes.

        Parameters:
            genes: the genes will appear in the returned dataset.
        """        
        indexes = samples_indexes
        new_dataset = Dataset()
        new_dataset.setGenes(self.genes)
        new_dataset.matrix = self.matrix[indexes, :]
        new_dataset.labels = list(np.array(self.labels)[indexes])
        new_dataset.samples = list(np.array(self.samples)[indexes])
        new_dataset.name = self.name+"_modified"
        new_dataset.sortSamplesByClassLabel()
        return new_dataset
    
    def shuffle_labels(self):
        print self.labels
        np.random.shuffle(self.labels)
        print self.labels

    def getMatrixZscoreWithColClassAsDataFrame(self):
        levels = self.levels().tolist()

        m = self.get_scaled_data().tolist()
        print(m)

        for i in range(len(self.labels)):
            lindex = levels.index(self.labels[i])
            m[i].append(lindex)

        col_labels = self.genes + ['class']    
            
        print(np.matrix(m))
        d = DataFrame(np.matrix(m), index=self.samples, columns=col_labels)

        print(d)
        return d

    # todo save dataset
    def save(self, filename):
        pass

    def getMinNumberOfSamplesPerClass(self):
        counts = {}        
        for label in self.labels:
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 0 
        
        return min(counts.values())

    def sortSamplesByClassLabel(self):
        indexes = np.argsort(self.labels)
        self.matrix = self.matrix[indexes, :]
        self.labels = list(np.array(self.labels)[indexes])
        self.samples = list(np.array(self.samples)[indexes])

    def X(self):
        return self.matrix
    
    def Y(self):
        return factorize(self.labels)[0]

    def geneIndex(self, gene_name):        
        return self.__geneIndex[gene_name]

    def correlatedAttributes(self, threshold=0.95):
        df = DataFrame(self.matrix)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        correlated_genes = {}
        for gene in self.genes:
            correlated_genes[gene] = set()

        for i in to_drop:
            for j in upper.columns:
                if upper[i][j] > threshold:     
                    correlated_genes[self.genes[j]].add(self.genes[i])
        
        genes_to_drop = [self.genes[i] for i in to_drop]


        # for cases like  0 (1, 2)      2  (3)
        # where 3 is highly correlated to 2, but is not highly as >threshold to 1 and 0...
        # identify and consider them to be correlated, cause 3 and 2 will be dropped, and 3 needs to be linked to some other variable
        for gene1 in correlated_genes:
            for gene2 in correlated_genes[gene1]:
                if len(correlated_genes[gene2]) > 0:
                    correlated_genes[gene1] = correlated_genes[gene1] | correlated_genes[gene2]

        new_correlated_genes = {}
        for gene in correlated_genes.keys():
            if gene not in genes_to_drop and len(correlated_genes[gene]) > 0:
                new_correlated_genes[gene] = correlated_genes[gene]
        correlated_genes = new_correlated_genes        

        return {'corr_matrix':corr_matrix, 'correlated_genes':correlated_genes, 'genes_to_drop': genes_to_drop}

