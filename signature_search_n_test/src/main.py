
# =================================
# =========== ARGS ================

# import the necessary packages
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument('--projectName', help='The name of the project folder where results are going to be saved. Example: melanoma_run1', action='store', required=True)

ap.add_argument('--noFilter', help='The algorithm will not filter proteins by Wilcoxon/Kruskal tests.', action='store_true')

ap.add_argument('--onlyFilter', help='The algorithm will only test the dataset composed by filtered proteins.', action='store_true')

ap.add_argument('--saveGraphics', help='Save Heatmaps, Scatterplots and others graphics for visualization of datasets and results.', action='store_true')

ap.add_argument('--fdr', help='Correct the Wilcoxon/Kruskal p-values used for filtering proteins by False Discovery Rate.', action='store_true')

ap.add_argument('--fdrPvalue', help='Minimun p-value used as cutoff after FDR correction.', action='store', type=int, default=0.1)

ap.add_argument('--minFdr', help='Minimun number of proteins to consider the FDR result to filter.', action='store', type=int, default=4)

ap.add_argument('--tTest', help='Uses T-Test instead of Wilcoxon when the dataset has 2 classes.', action='store_true')

ap.add_argument('--onlyStats', help='If set, only the statistics test will be executed.', action='store_true')

ap.add_argument('--pValue', help='Proteins are discarted if their p-values from Kruskal/Wilcoxon are greater or equal to the value of this argument (--p_value). Common choices: 0.01, 0.05.', action='store', default=0.05, type=float)

ap.add_argument('--train', help='Path for the train dataset.', action='store', required=True)

ap.add_argument('--test', help='Path for the independent test dataset.', action='store')

default_test_size= 0.08
ap.add_argument('--testSize', help='If --test is not defined, --test_size is used to define the independent test set size. If --test_size is set to 0.0, then the independent test is not performed; that is, only the CV is performed to evaluate the selected signatures.', action='store', type=float, default=-1)

ap.add_argument('--nSSearch', help='Set the maximum number of proteins to search for small signatures formed by all prot. combinations (signature size).', action='store', type=int, default=3)

ap.add_argument('--nSmall', help='Set the number of proteins considered small. If the total number of proteins in a dataset is smaller or equal than NSMALL, it will compute all combinations of proteins to form signatures. Otherwise, it will consider NSSEARCH to compute only combinations of size up to the value set for this parameter.', action='store', type=int, default=10)

ap.add_argument('--topN', help='Create all combinations of top-N signatures from the average of ranks.', action='store', type=int, default=10)

ap.add_argument('--deltaRankCutoff', help='The percentage of difference from the maximum score value that is used as cutoff univariate ranks. The scores are normalized between 0 and 1. So, if the maximum value is 0.9, and deltaRankCutoff is set to 0.05, the cutoff value is 0.85. Proteins with score >= 0.85 are selected to form signatures by top-N proteins.', action='store', type=float, default=0.10)

ap.add_argument('--k', help='The value of K for all k-fold cross-validations.', action='store', type=int, default=10)

ap.add_argument('--limitSignatureSize', help='Limit the size of signatures created by rank to the number of samples. For instance, when selecting top-N proteins using 30 samples, the maximum value of N is 30.', action='store_true')

args = vars(ap.parse_args())
print(args) # Values are saved in the report.txt file













# ======== MAIN =========

from random import shuffle
from dataset import Dataset
import datetime
from time import gmtime, strftime
import os
from statsmodels.stats.multitest import fdrcorrection
from kruskal import KruskalRankSumTest3Classes
from wilcoxon import WilcoxonRankSumTest
from ttest import TTest
from utils import saveHistogram, saveHeatMap, saveScatterPlots

# import pandas as pd
# import numpy as np

# detect the current working directory and print it
current_path = os.getcwd()  
print ("The current working directory is %s" % current_path)  

dt = strftime("%d-%m-%y %H:%M:%S",gmtime())
new_dir = current_path+'/results/'+ str(args['projectName']) +' ('+dt+')'
try:  
    os.mkdir(new_dir)
except OSError:  
    print ("Creation of the directory %s failed" % new_dir)
else:  
    print ("Successfully created the directory %s " % new_dir)
results_path = new_dir + '/'


train_dataset_path = args['train']
dataset = Dataset(train_dataset_path, scale=False, normalize=False, sep=',')

geneIndex = {}
i = 0
for gene in dataset.genes:
    geneIndex[gene] = i
    i += 1
geneNames = [gene for gene in dataset.genes]

test_dataset = None
train_dataset = None
test_dataset_path = args['test']

report =  open(results_path+'report.txt','w')
report.write('============= REPORT =============\n\n')
report.write('Arguments used in this project: %s\n\n' % str(args))

saveGraphics = args['saveGraphics']


if test_dataset_path != None:
    try:
        # TODO Test this parameter -> reading test dataset from file
        test_dataset = Dataset(test_dataset_path, scale=False, normalize=False, sep=',')
        train_dataset = dataset
    except expression as identifier:
        print('The path to test dataset (path:'+ test_dataset_path + ') is incorrect or the file is not defined.')
        exit()
else:
    print('The parameter --test was not defined. The system will separate the independent test set based on --test_size.')
    test_size_scale = 0
    if args['testSize'] == -1:
        print('The parameter --test_size was not set. The system will use the default value ' + str(default_test_size) + '.')
        test_size_scale = default_test_size
    else:
        test_size_scale = args['testSize']

    total_size = len(dataset.samples)
    test_size = int(round(total_size * test_size_scale))
    train_size = total_size - test_size
    print('\n')
    report.write('Number of samples in the train dataset: %d\n' % train_size)
    report.write('Number of samples in the independent test dataset: %d \n\n' % test_size)

    samples_index = range(total_size)
    shuffle(samples_index)

    train_dataset = dataset.get_sub_dataset_by_samples(samples_index[test_size:])
    test_dataset = dataset.get_sub_dataset_by_samples(samples_index[:test_size])

    report.write('Train samples: %s\n\n' % str(train_dataset.samples))
    report.write('Test samples: %s\n\n' % str(test_dataset.samples))

    if saveGraphics:
        saveHeatMap(train_dataset.matrix, train_dataset.samples, train_dataset.genes, results_path+'heatmap_train_dataset_correlation', metric='correlation')

        saveHeatMap(train_dataset.get_scaled_data(), train_dataset.samples, train_dataset.genes, results_path+'heatmap_train_dataset_correlation_zscore', metric='correlation')   

        saveHeatMap(train_dataset.matrix, train_dataset.samples, train_dataset.genes, results_path+'heatmap_train_dataset_euclidean', metric='euclidean')

        saveHeatMap(train_dataset.get_scaled_data(), train_dataset.samples, train_dataset.genes, results_path+'heatmap_train_dataset_euclidean_zscore', metric='euclidean')    


if args['noFilter']:
    filter_options = ['noFilter']
elif args['onlyFilter']:
    filter_options = ['filter']
else:
    filter_options = ['noFilter','filter']


global_results_path = results_path
for option in filter_options:
    report.write('\n\n============================='+option+'=============================\n')

    results_path = global_results_path+option+'/'
    try:  
        os.mkdir(results_path)
    except OSError:  
        print ("Creation of the directory %s failed" % results_path)
        exit()
    else:  
        print ("Successfully created the directory %s " % results_path)        

    if option == 'filter':
        stat_test_name = ''
        cutoff = args['pValue']
        p_values = []
        # if multiclass
        # TODO if number of levels > 3, it requires more general kruskal implementation - for any number of classes
        if len(train_dataset.levels()) == 3:
            # Kruskal
            stat_test_name = 'Kruskal Wallis p-values histogram'
            krus = KruskalRankSumTest3Classes(train_dataset)
            krus_h, krus_p = krus.run()
            p_values = krus_p
            with open(results_path+'kruskal.csv', 'w') as f:
                f.write("gene,p-value\n")
                for i in range(len(train_dataset.genes)):
                    f.write(train_dataset.genes[i]+","+str(krus_p[i])+"\n")    

        # if 2-class
        elif len(train_dataset.levels()) == 2:   

            if args['t_test']:
                #t-test
                stat_test_name = 'T-test p-values histogram'
                ttest = TTest(train_dataset)
                ttest_t, ttest_p = ttest.run() 
                p_values = ttest_p

                with open(results_path+'t_test.csv', 'w') as f:
                    f.write("gene,p-value\n")
                    for i in range(len(train_dataset.genes)):
                        f.write(train_dataset.genes[i]+","+str(ttest_p[i])+"\n")           
            else:
                #wilcoxon
                stat_test_name = 'Wilcoxon p-values histogram'
                            # Filtering train and test Datasets
                wil = WilcoxonRankSumTest(train_dataset)
                wil_z, wil_p = wil.run() 
                p_values = wil_p      

                with open(results_path+'wilcoxon_test.csv', 'w') as f:
                    f.write("gene,p-value\n")
                    for i in range(len(train_dataset.genes)):
                        f.write(train_dataset.genes[i]+","+str(wil_p[i])+"\n")        


        # print p-value histogram
        #saveHistogram(filename=results_path+'p_values_histogram_from_filter_no_correction', values=p_values, title=stat_test_name)
        if saveGraphics:
            saveHistogram(filename=results_path+'histogram_p_values.png', values=p_values, title=stat_test_name, xlabel='p-values', ylabel='counts', bins=20, rwidth=0.9, color='#607c8e', grid=False, ygrid=True, alpha=0.75)

        p_values_corrected = None
        filtered = None
        fdr_cutoff = args['fdrPvalue']
        if args['fdr']:
            #correct p-values
            print('\nP-values before correction: %s \n\n' % str(p_values))
            p_values_corrected = fdrcorrection(p_values, alpha=0.25, method='indep', is_sorted=False)[1]
            print('P-values after correction: %s\n\n' % str(p_values_corrected))
            filtered = [train_dataset.genes[i] for i in range(len(train_dataset.genes)) if p_values_corrected[i]<fdr_cutoff]

            with open(results_path+'corrected_pvalues_fdr.csv', 'w') as f:
                f.write("gene,p-value\n")
                for i in range(len(train_dataset.genes)):
                    f.write(train_dataset.genes[i]+","+str(p_values_corrected[i])+"\n")   

            if saveGraphics:
                saveHistogram(filename=results_path+'histogram_p_values_corrected_fdr.png', values=p_values_corrected, title=stat_test_name+' (corrected)', xlabel='p-values', ylabel='counts', bins=20, rwidth=0.9, color='#607c8e', grid=False, ygrid=True, alpha=0.75)            
            
            print('\nSelected proteins after FDR correction: %s\n\n' % str(filtered))

            report.write('P-values were corrected by FDR and these are the remaining proteins with p-value < %f: %s\n' % (fdr_cutoff, str(filtered)))
            if len(filtered) < 2:
                print('Interrupting the algorithm because the number of selected proteins after FDR filter is < 2')
                exit()
            


        # ========= save graphics  - without correction of p-value ===========
        if saveGraphics:
            train_dataset_temp = train_dataset.get_sub_dataset([train_dataset.genes[i] for i in range(len(train_dataset.genes)) if p_values[i]<cutoff])
        
            saveHeatMap(train_dataset_temp.matrix, train_dataset_temp.samples, train_dataset_temp.genes, results_path+'heatmap_train_dataset_correlation_filtered', metric='correlation', xticklabels=True)

            saveHeatMap(train_dataset_temp.get_scaled_data(), train_dataset_temp.samples, train_dataset_temp.genes, results_path+'heatmap_train_dataset_correlation_filtered_zscore', metric='correlation', xticklabels=True) 

            saveHeatMap(train_dataset_temp.matrix, train_dataset_temp.samples, train_dataset_temp.genes, results_path+'heatmap_train_dataset_euclidean_filtered', metric='euclidean', xticklabels=True)

            saveHeatMap(train_dataset_temp.get_scaled_data(), train_dataset_temp.samples, train_dataset_temp.genes, results_path+'heatmap_train_dataset_euclidean_filtered_zscore', metric='euclidean', xticklabels=True) 

            saveScatterPlots(train_dataset_temp.getMatrixZscoreWithColClassAsDataFrame(), results_path+'scatterplots_filtered_zscore')   
        # ========= end graphics ===========



        # aply filter and create a new train/test data set
        current_cutoff = cutoff
        if args['fdr']:
            if len(filtered) >= args['minFdr']:
                p_values = p_values_corrected
                current_cutoff = fdr_cutoff
            else:
                print('The number of filtered proteins is smaller than minFdr. The program was interrupted.')
                exit()
        

        train_dataset = train_dataset.get_sub_dataset([train_dataset.genes[i] for i in range(len(train_dataset.genes)) if p_values[i]<current_cutoff])

        test_dataset = test_dataset.get_sub_dataset([test_dataset.genes[i] for i in range(len(test_dataset.genes)) if p_values[i]<current_cutoff])
        



        # ========= save graphics ===========
        if saveGraphics:
            saveHeatMap(train_dataset.matrix, train_dataset.samples, train_dataset.genes, results_path+'heatmap_train_dataset_correlation_filtered_fdr', metric='correlation', xticklabels=True)

            saveHeatMap(train_dataset.get_scaled_data(), train_dataset.samples, train_dataset.genes, results_path+'heatmap_train_dataset_correlation_filtered_fdr_zscore', metric='correlation', xticklabels=True)

            saveHeatMap(train_dataset.matrix, train_dataset.samples, train_dataset.genes, results_path+'heatmap_train_dataset_euclidean_filtered_fdr', metric='euclidean', xticklabels=True)

            saveHeatMap(train_dataset.get_scaled_data(), train_dataset.samples, train_dataset.genes, results_path+'heatmap_train_dataset_euclidean_filtered_fdr_zscore', metric='euclidean', xticklabels=True)

        #saveScatterPlots(train_dataset.getMatrixZscoreWithColClassAsDataFrame(), results_path+'scatterplots_filtered_fdr_zscore')
        # ========= end graphics ===========




    if args['onlyStats']:
        print('\n The parameter onlyStats was set and the statistical results are ready.\nThe algorithm stops here.')
        report.close()
        exit()




    #================== signatures by ranking =========================
    # for each ranking method, create signatures by selecting top-N proteins
    # if a signature exists, add the method to its methods list
    # sum the ranks position for each protein
    # SAVE ALL THE RANKS

    from pandas import factorize    
    import numpy as np
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    from utils import saveRank
    from signature import Signature

    signatures_data = dict()
    proteins_ranks = []

    k = args['k']
    if k > train_dataset.getMinNumberOfSamplesPerClass():
        print("\n-------------!!!-------------\n")
        print("Error: The defined K for K-fold Cross-validation is greater than the number of members in each class.\n")
        print("Please consider using a smaller number for the parameter --k\n")
        print("Current K: %d\n" % k)
        print("Max value for K: %d\n" % train_dataset.getMinNumberOfSamplesPerClass())
        exit()

    deltaScore = float(args['deltaRankCutoff'])

    limitSigSize = args['limitSignatureSize']
    maxNumberOfProteins = len(train_dataset.genes)
    if limitSigSize:
        maxNumberOfProteins = len(train_dataset.samples)

    # Type 1: Model Based Ranking
    # Type 2: Regularization (L1 and L2)
    # Type 3: Univariate Feature Selection
    # Type 4: Recursive Feature Elimination
    # Type 5: Random Forest Feature Importance
    # Type 6: Stability Selection

    # ---------------------- Model Based Ranks -------------------------
   
    # Random Forests
    clf = RandomForestClassifier()
    scores = []
    for i in range(len(train_dataset.genes)):
        x_train = train_dataset.matrix[:, i] # data matrix
        y_train = factorize(train_dataset.labels)[0]  # classes/labels of each sample from 
        scores_cv = cross_val_score(clf, x_train, y_train, cv=k)        
        score = np.mean(scores_cv)
        scores.append((score, geneIndex[train_dataset.genes[i]], train_dataset.genes[i]))

    scores = sorted(scores, reverse = True)

    saveRank(scores, results_path+'rank_t1_uni_rf_mean_accuracy.csv')

    maxScore = max(scores,key=lambda item:item[0])[0]
    cutoffScore = maxScore-deltaScore
    genes_to_show = [item for item in scores if item[0] > cutoffScore]

    genes_for_signature = []
    for i in range(1,maxNumberOfProteins+1):
        genes_indexes = [item[1] for item in scores[0:i]]
        sig = Signature(genes_indexes)
        if sig in signatures_data:
            signatures_data[sig]['methods'].add('model_based_rank_random_forest')
        else:
            sig_data = {'methods':set()}
            sig_data['methods'].add('model_based_rank_random_forest')
            signatures_data[sig] = sig_data

    report.write('-- Model Based Rank - Random Forests --\n')
    report.write('Max Mean Accuracy by individual protein: %f\n' %maxScore)
    report.write('%d proteins have mean accuracy > %f (max-deltaRankCutoff): %s\n' %(len(genes_to_show), cutoffScore, str(genes_to_show)))
    

    report.write('\n\n')

    # beta-binomial

    #-----------------------------------------------------------------------------
    # kruskal/t-test/wilcoxon
    scores = []
    name = None
    test = None
    if len(train_dataset.levels()) == 3:
        # Kruskal
        stat_test_name = 'Kruskal Wallis p-values histogram'
        test = KruskalRankSumTest3Classes(train_dataset)
        name = 'kruskal'
    # if 2-class
    elif len(train_dataset.levels()) == 2:   

        if args['t_test']:
            #t-test
            stat_test_name = 'T-test p-values histogram'
            test = TTest(train_dataset)
            name = 'ttest'             
        else:
            #wilcoxon
            stat_test_name = 'Wilcoxon p-values histogram'     
            test = WilcoxonRankSumTest(train_dataset)
            name = 'wilcox'           
    h, p = test.run()    
    for i in range(len(train_dataset.genes)):
        score = (p[i],  geneIndex[train_dataset.genes[i]],  train_dataset.genes[i])
        scores.append(score)

    scores = sorted(scores, reverse = False)

    saveRank(scores, results_path+'rank_t1_uni_'+name+'_pvalue.csv')

    minScore = min(scores,key=lambda item:item[0])[0]
    cutoffScore = minScore+deltaScore
    genes_to_show = [item for item in scores if item[0] < cutoffScore]

    genes_for_signature = []
    for i in range(1,maxNumberOfProteins+1):
        genes_indexes = [item[1] for item in scores[0:i]]
        sig = Signature(genes_indexes)        
        if sig in signatures_data:
            signatures_data[sig]['methods'].add('model_based_rank_'+name)
        else:
            sig_data = {'methods':set()}
            sig_data['methods'].add('model_based_rank_'+name)
            signatures_data[sig] = sig_data

    report.write('-- Model Based Rank - '+name+' --\n')
    report.write('Minimum p-value: %f\n' %minScore)
    report.write('%d proteins have p-value < %f (min+deltaRankCutoff): %s\n' %(len(genes_to_show), cutoffScore, str(genes_to_show)))
    #-----------------------------------------------------------------------------








    # ------------------------------------------------------------------












    # rank by mean-rank position (from above methods)

    # create all combinations considering args['top-n'] proteins
    max_n = args['topN']

    # if total number of proteins is <= args['n_small']
        #compute all possible combinations

    # if total number of proteins is > args['n_small']
        # small signatures from all proteins
    max_sig_size = args['nSSearch']



    print('There are %d signatures that are going to be tested:' % len(signatures_data.keys()))
    
    for signature in signatures_data:        
        print('%d - %s: %s' % (len(signatures_data[signature]['methods']), signature.toGeneNamesList(geneNames), str(signatures_data[signature]['methods']) ))

report.close()


#  ================ END MAIN ===============

