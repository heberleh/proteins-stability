

#install scikit-bio and run with python3

from pandas import DataFrame
import pandas as pd
from dataset import Dataset
from sklearn.preprocessing import scale, normalize, robust_scale
import numpy as np
import os

from skbio.tree import nj
from skbio import DistanceMatrix, TreeNode
from kruskal import KruskalRankSumTest3Classes
from wilcoxon import WilcoxonRankSumTest
from statsmodels.stats.multitest import fdrcorrection

def filter_dataset(dataset, cutoff=0.25, fdr=True):
    p_values = []
    scores = []
    if len(dataset.levels()) == 3:
        # Kruskal
        stat_test_name = 'Kruskal Wallis p-values histogram'
        krus = KruskalRankSumTest3Classes(dataset)
        krus_h, krus_p = krus.run()
        p_values = krus_p
    # if 2-class
    elif len(dataset.levels()) == 2:   
        #wilcoxon
        stat_test_name = 'Wilcoxon p-values histogram'                   
        wil = WilcoxonRankSumTest(dataset)
        wil_z, wil_p = wil.run() 
        p_values = wil_p

    if fdr:
        p_values = fdrcorrection(p_values, alpha=fdr, method='indep', is_sorted=False)[1] 

    return dataset.get_sub_dataset([dataset.genes[i] for i in range(len(dataset.genes)) if p_values[i]<cutoff])





# create the tree given a DataFrame[protein]
def create_tree(df, column):
    samples = df.index.tolist()
    
    # idenfity values > 0 (greater than the mean, which is zero after zscore)
    values = []
    valid_samples = []
    for i in range(len(samples)):
        if df[column][samples[i]] > 0:
            values.append(df[column][samples[i]])
            valid_samples.append(samples[i])
    
    size = len(valid_samples)
    print(size)
    if size > 3:
        dist_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i, size):
                dist_matrix[i][j] = np.abs(values[i]-values[j])
                dist_matrix[j][i] = dist_matrix[i][j]
        maxv = dist_matrix.max()
        minv = dist_matrix.min()
        delta = maxv - minv
        if delta == 0.0:
            raise Exception("max-min = 0 in distance matrix")
        
        dist_matrix = (dist_matrix-minv)/delta
        dmat = DistanceMatrix(dist_matrix, valid_samples)    
        return nj(dmat).root_at_midpoint()
    else:        
        return None




current_path = os.getcwd()  

print ("The current working directory is %s" % current_path)  
trees_path = current_path+ '/results/trees'

# Data paths
path1 = os.path.join(current_path, 'dataset/romenia_all_samples_trees.csv')
#path2 = os.path.join(current_path, 'dataset/all_samples_tatiane_trees.csv')
path3 = os.path.join(current_path, 'dataset/prostate_all_samples_trees.csv')

# Read data sets
d1 = filter_dataset(Dataset(path1, scale=False, normalize=False, sep=','), 0.25, fdr=True)
#d2 = filter_dataset(Dataset(path2, scale=False, normalize=False, sep=','), 0.10, fdr=false)
d3 = filter_dataset(Dataset(path3, scale=False, normalize=False, sep=','), 0.25, fdr=True)

# Find what is above Mean in each data set... 
m1 = d1.matrix
m1 = robust_scale(m1)
print(m1)
m1[m1<=0.0] = np.nan
m1 = m1+10

m3 = d3.matrix
m3 = robust_scale(m3)
m3[m3<=0.0] = np.nan
m3 = m3+30

# Join the data set into one matrix
df1 = DataFrame(m1, index=d1.samples, columns=d1.genes)
#df2 = DataFrame(m2, index=d2.samples, columns=d2.genes)
df3 = DataFrame(m3, index=d3.samples, columns=d3.genes)
result = DataFrame()
result = result.append(df1)
#result = result.append(df2)
result = result.append(df3)


result = result.fillna(0.0)
#P31146

# Normalize unit vector l2
matrix = np.matrix(result).astype(float)
#matrix = normalize(matrix)

matrix_df = DataFrame(matrix, index=result.index, columns=result.columns)

matrix_df.to_csv(os.path.join(trees_path, 'merged_matrix.csv'), header=True)

# For each column (gene) create a distance matrix, a tree, and save it in the trees file.

trees_file =  open(os.path.join(trees_path, 'trees.txt'),'w')
protein_attributes_file = open(os.path.join(trees_path, 'proteins.txt'),'w')
count = 0
for protein in matrix_df.columns.tolist():
    tree = create_tree(matrix_df, protein)
    if not tree is None:
        tree_nw = str(tree).replace('root','')
        trees_file.write(tree_nw)
        protein_attributes_file.write(protein+'\n')
    else:
        count+=1
trees_file.close()
print("\n%d/%d trees were < 4." % (count, len(matrix_df.columns.tolist())))










