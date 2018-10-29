

#install scikit-bio and run with python3

from pandas import DataFrame
import pandas as pd
from dataset import Dataset
from sklearn.preprocessing import scale, normalize
import numpy as np
import os

from skbio.tree import nj
from skbio import DistanceMatrix, TreeNode

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
    if size > 2:
        dist_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i, size):
                dist_matrix[i][j] = np.abs(values[i]-values[j])
                dist_matrix[j][i] = dist_matrix[i][j]


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
d1 = Dataset(path1, scale=True, normalize=False, sep=',')
#d2 = Dataset(path2, scale=True, normalize=False, sep=',')
d3 = Dataset(path3, scale=True, normalize=False, sep=',')

# Find what is above Mean in each data set... 
m1 = d1.matrix
m1 = scale(m1)
m1[m1<0.0] = 0.0
m1 = m1 * 1

m3 = d3.matrix
m3 = scale(m3)
m3[m3<0.0] = 0.0
m3 = m3 * 3

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
matrix = normalize(matrix)

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
print("\n%d/%d trees were < 3." % (count, len(matrix_df.columns.tolist())))









