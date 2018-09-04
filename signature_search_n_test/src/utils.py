
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def saveHistogram(values, filename='histogram.png', dpi=300,  title='Histogram', xlabel='Values', ylabel='Counts', bins=20, rwidth=0.9, color='#607c8e', grid=True, ygrid=True, alpha=0.75, xlim=(0, 1)):
    commutes = pd.Series(np.array(values))

    commutes.plot.hist(grid=grid, bins=bins, rwidth=rwidth,
                    color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ygrid:
        plt.grid(axis='y', alpha=alpha)
    plt.savefig(filename, dpi=dpi)
    plt.xlim(xlim)
    plt.close()


def saveScatterPlots(dataset, filename):
    #g = sns.PairGrid(iris, hue="species")
    #g = sns.PairGrid(dataset, hue="class", height=2.5)
    print(dataset.columns)
    genes = dataset.columns.tolist()
    
    genes.remove('class')
    print(genes)

    g = sns.PairGrid(dataset, hue="class", vars=genes,  hue_kws={"cmap": ["Blues", "Greens", "Reds"]}) #palette="Set2",
    #g.map_diag(plt.scatter)
    g.map_diag(sns.kdeplot)
    g.map_lower(plt.scatter)
    g.map_upper(sns.kdeplot) 
    g.savefig(filename, dpi=300)
    plt.close()

def saveHeatMap(matrix, rows_labels, cols_labels, filename, metric='correlation', xticklabels=False):
    dataset = pd.DataFrame(matrix, index=rows_labels, columns=cols_labels)

    g = sns.clustermap(dataset, metric=metric, xticklabels=xticklabels)
    g.savefig(filename, dpi=300)
    plt.close()