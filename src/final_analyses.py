import pandas as pd
import numpy as np


sig_matrix_path="results/romenia_run_kruskal_fdr25_outerk9_innerk7_18_11_09_13_49_30/0/filter/best_signatures_prot_matrix.csv"
pd_best = pd.read_csv(sig_matrix_path, index_col=0, header=None)
print(pd_best   )
proteins = {}
for i in pd_best.index:
    for protein in pd_best.loc[i]:
        if not pd.isna(protein):
            if protein in proteins:
                proteins[protein] += 1
            else:
                proteins[protein] = 1
print(proteins)
proteins_tuples = [(protein, proteins[protein]) for protein in proteins]

proteins = [tup[0] for tup in sorted(proteins_tuples, key=lambda tup: tup[1], reverse=True)]

matrix = []
for protein in proteins:
    row = []
    for i in pd_best.index:      
        if protein in list(pd_best.loc[i]):
            row.append(1)
        else:
            row.append(0)
    matrix.append(row)
matrix = np.matrix(matrix).transpose().tolist()
matrix=sorted(matrix, key=lambda row: sum(row), reverse=True)
matrix = np.matrix(matrix).transpose()
#matrix.sort(axis=1)
#matrix = np.flip(matrix,1)

df = pd.DataFrame(matrix, columns=pd_best.index, index=proteins)
df.to_csv(sig_matrix_path.replace(".csv","")+"_binary.csv", header=True, index=True)