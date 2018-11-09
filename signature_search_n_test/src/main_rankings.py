
# import the necessary packages
import argparse
import os
# =================================
# =========== ARGS ================
import warnings
from datetime import datetime
from random import shuffle
from time import gmtime, strftime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame, factorize
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_selection import (RFE, SelectKBest, chi2,
                                       mutual_info_classif)
from sklearn.linear_model import Lars, LogisticRegression
from sklearn.metrics import (cohen_kappa_score, f1_score, make_scorer,
                             matthews_corrcoef)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.multitest import fdrcorrection

from dataset import Dataset
from kruskal import KruskalRankSumTest3Classes
from recursiveFeatureAddition import RFA
from ttest import TTest
from utils import (getMaxNumberOfProteins, normalizeScores, saveBoxplots,
                   saveHeatMap, saveHeatMapScores, saveHistogram, saveRank,
                   saveScatterPlots)
from wilcoxon import WilcoxonRankSumTest

## agg backend is used to create plot as a .png file
mpl.use('agg')


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument('--projectName', help='The name of the project folder where results are going to be saved. Example: melanoma_run1', action='store', required=True)

ap.add_argument('--noFilter', help='The algorithm will not filter proteins by Wilcoxon/Kruskal tests.', action='store_true')

ap.add_argument('--onlyFilter', help='The algorithm will only test the dataset composed by filtered proteins.', action='store_true')

ap.add_argument('--saveGraphics', help='Save Heatmaps, Scatterplots and others graphics for visualization of datasets and results.', action='store_true')

ap.add_argument('--fdr', help='Correct the Wilcoxon/Kruskal p-values used for filtering proteins by False Discovery Rate.', action='store_true')

ap.add_argument('--fdrPvalue', help='Minimun p-value used as cutoff after FDR correction.', action='store', type=float, default=0.05)

ap.add_argument('--minFdr', help='Minimun number of proteins to consider the FDR result to filter.', action='store', type=int, default=4)

ap.add_argument('--tTest', help='Uses T-Test instead of Wilcoxon when the dataset has 2 classes.', action='store_true')

ap.add_argument('--onlyStats', help='If set, only the statistics test will be executed.', action='store_true')

ap.add_argument('--pValue', help='Proteins are discarted if their p-values from Kruskal/Wilcoxon are greater or equal to the value of this argument (--p_value). Common choices: 0.01, 0.05.', action='store', default=0.05, type=float)

ap.add_argument('--train', help='Path for the train dataset.', action='store', required=True)

ap.add_argument('--test', help='Path for the independent test dataset.', action='store')

default_test_size= 0.15
ap.add_argument('--testSize', help='If --test is not defined, --test_size is used to define the independent test set size. Default=-1, use the entire dataset.', action='store', type=float, default=-1)

ap.add_argument('--outerK', help='The value of K for all outer k-fold cross-validations.  Default=-1, do not double-cross-validate', action='store', type=int, default=-1)

ap.add_argument('--innerK', help='The value of K for all inner k-fold cross-validations.', action='store', type=int, default=10)

ap.add_argument('--nJobs', help='Number of parallel jobs when fitting models.', action='store', type=int, default=1)

ap.add_argument('--ignoreWarnings', help='Stop printing Warnings from models.', action='store_true')

ap.add_argument('--correlation', help='Filter variables highly correlated.', action='store_true')

ap.add_argument('--corrThreshold', help='Cuttoff for correlation.', action='store', type=float, default=0.95)

ap.add_argument('--smote', help='Balance the dataset with oversampling.', action='store_true')

args = vars(ap.parse_args())
print(args) # Values are saved in the report.txt file




# ======== MAIN =========


# detect the current working directory and print it
current_path = os.getcwd()  

print ("The current working directory is %s" % current_path)  

starting_time  = datetime.now()

dt = datetime.now().strftime('%y_%m_%d_%H_%M_%S')

new_dir = current_path+'/results/'+ str(args['projectName']) +'_'+dt
try:  
    os.mkdir(new_dir)
except OSError:  
    print ("Creation of the directory %s failed" % new_dir)
else:  
    print ("Successfully created the directory %s " % new_dir)
results_path = new_dir + '/'


train_dataset_path = args['train']
dataset = Dataset(train_dataset_path, scale=False, normalize=False, sep=',')

scoreEstimator = None



def save_ratio(dataset, filename, normalize=False):
    matrix = []
    matrix_csv = []
    x = dataset.X()
    cluster_col = True
    if normalize:
        scaler = MinMaxScaler((0.00000001,1))
        scaler.fit(x)
        x = scaler.transform(x)
    if len(dataset.levels()) == 3:
              
        for i in range(len(dataset.genes)):
            row = []
            # 0 x 1
            inds0 = dataset.Y() == 0
            inds1 = dataset.Y() == 1
            x0 = np.mean(x[inds0,i])
            x1 = np.mean(x[inds1,i])
            if np.abs(x0) == 0:
                x0 = 0.0000001
            if np.abs(x1) == 0:
                x1 = 0.0000001
            row.append(np.log2(x0) - np.log2(x1))

            # 0 x 2
            inds0 = dataset.Y() == 0
            inds2 = dataset.Y() == 2
            x0 = np.mean(x[inds0,i])
            x1 = np.mean(x[inds2,i])
            if np.abs(x0) == 0:
                x0 = 0.0000001
            if np.abs(x1) == 0:
                x1 = 0.0000001
            row.append(np.log2(x0) - np.log2(x1))

            # 1 x 2
            inds1 = dataset.Y() == 1
            inds2 = dataset.Y() == 2
            x0 = np.mean(x[inds1,i])
            x1 = np.mean(x[inds2,i])
            if np.abs(x0) == 0:
                x0 = 0.0000001
            if np.abs(x1) == 0:
                x1 = 0.0000001
            row.append(np.log2(x0) - np.log2(x1))                      
            matrix.append(row)            
            
        l = dataset.levels()
        df = DataFrame(matrix, index=dataset.genes, columns=np.array([l[0]+"/"+l[1], l[0]+"/"+l[2], l[1]+"/"+l[2]]))
    elif len(dataset.levels()) == 2:                  
        for i in range(len(dataset.genes)):
            row = []
            # 0 x 1
            inds0 = dataset.Y() == 0
            inds1 = dataset.Y() == 1
            x0 = np.mean(x[inds0,i])
            x1 = np.mean(x[inds1,i])
            if np.abs(x0) == 0:
                x0 = 0.0000001
            if np.abs(x1) == 0:
                x1 = 0.0000001
            row.append(np.log2(x0) - np.log2(x1))
            row.append(np.log2(x1) - np.log2(x0))
            matrix.append(row)

        l = dataset.levels()
        df = DataFrame(matrix, index=dataset.genes, columns=np.array([l[0]+"/"+l[1], l[1]+"/"+l[0]]))
    else:
        raise Exception("Number of classes > 3 or < 2 is not supported.") 

    saveHeatMapScores(matrix, dataset.genes, df.columns, filename=filename, metric='euclidean')
    df.to_csv(filename+'.csv', header=True)

    matrix = np.matrix(matrix)
    matrix[matrix > 1] = 1
    matrix[matrix <-1] = -1
    saveHeatMapScores(matrix, dataset.genes, df.columns, filename=filename+"_limited_by_1", metric='euclidean')    



saveHeatMap(dataset.get_scaled_data(), dataset.samples, dataset.genes, dataset.Y(),  results_path+'heatmap_dataset_correlation_zscore', metric='correlation', classes_labels=dataset.levels())   

saveHeatMap(dataset.X(), dataset.samples, dataset.genes, dataset.Y(),  results_path+'heatmap_dataset_correlation', metric='correlation', classes_labels=dataset.levels())   

saveHeatMap(dataset.get_scaled_data(), dataset.samples, dataset.genes, dataset.Y(),  results_path+'heatmap_dataset_euclidean_zscore', metric='euclidean', classes_labels=dataset.levels())   

saveHeatMap(dataset.X(), dataset.samples, dataset.genes, dataset.Y(),  results_path+'heatmap_dataset_euclidean', metric='euclidean', classes_labels=dataset.levels())   



matthews_scorer = make_scorer(matthews_corrcoef)
kappa_scorer = make_scorer(cohen_kappa_score)

uni, counts = np.unique(dataset.Y(), return_counts=True)

from sklearn.metrics import f1_score
if len(dataset.levels()) < 4:
    scoreEstimator = make_scorer(f1_score, average='weighted')    
    scoreEstimatorInfo = """Weighted F1"""
else:
    print('\nDataset with more than 3 classes is not supported.\n\n')
    exit()

#pos_label=positive_class_index

#===================================
geneNames = [gene for gene in dataset.genes]
test_dataset = None
train_dataset = None
test_dataset_path = args['test']
k_outer = args['outerK']
double_cross_validation = False

score_matrices = []
freq_prot_list = []

if test_dataset_path != None:
    try:
        # TODO Test this parameter -> reading test dataset from file
        test_dataset = Dataset(test_dataset_path, scale=False, normalize=False, sep=',')
        train_dataset = dataset
    except expression as identifier:
        print('The path to test dataset (path:'+ test_dataset_path + ') is incorrect or the file is not defined.')
        exit()
else:
    print('The parameter --test was not defined.')
    test_size_scale = 0 
    if k_outer != -1:
        print("K-outer was set, so the script will run DCV.")
        double_cross_validation = True     
    elif args['testSize'] == -1:                  
        print('The parameter --test_size was not set. The system will use the default value ' + str(default_test_size) + '.')
        test_size_scale = default_test_size
    else:
        test_size_scale = args['testSize']

        if test_size_scale == 0:
            #copy dataset, train and test are going to be the same.
            print("Using test == train set == all samples.")
            train_dataset = dataset.get_sub_dataset_by_samples(range(len(dataset.samples)))
            test_dataset = dataset.get_sub_dataset_by_samples(range(len(dataset.samples))) 
        else:
            if k_outer == -1:
                print("Using test-size "+str(test_size_scale))
                total_size = len(dataset.samples)    

                x_indexes = []
                for i in range(total_size):
                    x_indexes.append([i])    
                X_train, X_test, y_train, y_test = train_test_split(x_indexes, dataset.Y(), test_size=test_size_scale, shuffle=True, stratify=dataset.Y())

                train_indexes = [v[0] for v in X_train]
                test_indexes = [v[0] for v in X_test]

                train_dataset = dataset.get_sub_dataset_by_samples(train_indexes)
                test_dataset = dataset.get_sub_dataset_by_samples(test_indexes)           

folder_count = 0
datasets_indexes = [(None, None)]
if double_cross_validation:
    skf = StratifiedKFold(n_splits=k_outer)
    datasets_indexes = skf.split(dataset.X(), dataset.Y())  


try:
    filename = results_path+'ratio_train_samples'
    save_ratio(dataset, filename)

    filename = results_path+'ratio_train_samples_normalized'
    save_ratio(dataset, filename, normalize=True)
except:
    print(dataset.X())



results_path_parent = results_path
for train_index, test_index in datasets_indexes:

    if double_cross_validation:
        train_dataset = dataset.get_sub_dataset_by_samples(train_index)
        test_dataset = dataset.get_sub_dataset_by_samples(test_index)
   

    results_path = results_path_parent
    new_dir = results_path + str(folder_count)
    folder_count += 1
    try:  
        os.mkdir(new_dir)
    except OSError:  
        print ("Creation of the directory %s failed" % new_dir)
    else:  
        print ("Successfully created the directory %s " % new_dir)
    results_path = new_dir + '/'



    report =  open(results_path+'report.txt','w')
    report.write('============= REPORT =============\n\n')
    report.write('Arguments used in this project: %s\n\n' % str(args))

    saveGraphics = args['saveGraphics']
    nJobs = args['nJobs']

    report.write('\n\n Score used for rankings with Classifiers is: %s\n' % scoreEstimator)
    report.write("Info about score: %s: \n\n" % scoreEstimatorInfo)

    report.write('Number of samples in the original dataset: %d\n' % len(dataset.samples))
    report.write('\n\nNumber of features in the original dataset: %d\n\n' % len(dataset.genes))
    report.flush()

    if saveGraphics:
        saveHeatMap(train_dataset.get_scaled_data(), train_dataset.samples, train_dataset.genes, train_dataset.Y(), results_path+'heatmap_train_dataset_correlation_zscore', metric='correlation', classes_labels=train_dataset.levels())   

        saveHeatMap(train_dataset.get_scaled_data(), train_dataset.samples, train_dataset.genes, train_dataset.Y(), results_path+'heatmap_train_dataset_euclidean_zscore', metric='euclidean', classes_labels=train_dataset.levels())    


    
    report.write('Number of samples in the train dataset: %d\n' % len(train_dataset.samples))
    report.write('Number of samples in the independent test dataset: %d \n\n' % len(test_dataset.samples))

    report.write('Train samples: %s\n\n' % str(train_dataset.samples))
    report.write('Test samples: %s\n\n' % str(test_dataset.samples))

    report.write('Number of classes: %d\n' % len(train_dataset.levels()))
    report.write('Classes: %s' % str(train_dataset.levels()))

    report.flush()

    if args['noFilter']:
        filter_options = ['noFilter']
    elif args['onlyFilter']:
        filter_options = ['filter']
    else:
        print("You need to set the parameter --noFilder or the parameter --onlyFilter.")
        exit()
    #else: #! ignoring the possibility of computing both in 1 run due to complexity
    #   filter_options = ['filter', 'noFilter']


    complete_train = train_dataset
    complete_test = test_dataset

    n_estimators = 83

    global_results_path = results_path
    for option in filter_options:
        report.write('\n\n============================='+option+'=============================\n')

        #! if train_dataset is filtered, when runing without filter or future configurations it is necessary to reset the train_dataset:
        train_dataset = complete_train
        test_dataset = complete_test

        results_path = global_results_path+option+'/'
        try:  
            os.mkdir(results_path)
        except OSError:  
            print ("Creation of the directory %s failed" % results_path)
            exit()
        else:  
            print ("Successfully created the directory %s " % results_path)       

        results_path_rank = results_path+'rank/'
        try:  
            os.mkdir(results_path_rank)
        except OSError:  
            print ("Creation of the directory %s failed" % results_path)
            exit()
        else:  
            print ("Successfully created the directory %s " % results_path_rank)    


        print('Computing statistical tests.\n')
        stat_test_name = ''
        cutoff = args['pValue']
        p_values = []
        # if multiclass
        # TODO if number of levels > 3, it requires more general kruskal implementation - for any number of classes
        scores = []
        if len(train_dataset.levels()) == 3:
            # Kruskal
            stat_test_name = 'Kruskal Wallis p-values histogram'
            krus = KruskalRankSumTest3Classes(train_dataset)
            krus_h, krus_p = krus.run()
            p_values = krus_p
            method = 'kruskal.csv'
            for i in range(len(train_dataset.genes)):
                score = p_values[i]
                scores.append((abs(score), train_dataset.geneIndex(train_dataset.genes[i]), train_dataset.genes[i]))            
            scores = sorted(scores, reverse = False, key=lambda tup: tup[0])
            saveRank(scores, results_path+method)

        # if 2-class
        elif len(train_dataset.levels()) == 2:   

            if args['tTest']:
                #t-test
                stat_test_name = 'T-test p-values histogram'
                ttest = TTest(train_dataset)
                ttest_t, ttest_p = ttest.run() 
                p_values = ttest_p
                method = 'tTest.csv'
                for i in range(len(train_dataset.genes)):
                    score = p_values[i]
                    scores.append((abs(score), train_dataset.geneIndex(train_dataset.genes[i]), train_dataset.genes[i]))            
                scores = sorted(scores, reverse = False, key=lambda tup: tup[0])
                saveRank(scores, results_path+method)  
            
            else:
                #wilcoxon
                stat_test_name = 'Wilcoxon p-values histogram'
                            # Filtering train and test Datasets
                wil = WilcoxonRankSumTest(train_dataset)
                wil_z, wil_p = wil.run() 
                p_values = wil_p
                method = 'wilcoxon.csv'
                for i in range(len(train_dataset.genes)):
                    score = p_values[i]
                    scores.append((abs(score), train_dataset.geneIndex(train_dataset.genes[i]), train_dataset.genes[i]))            
                scores = sorted(scores, reverse = False, key=lambda tup: tup[0])
                saveRank(scores, results_path+method)  
    

        # ========= save graphics  - without correction of p-value ===========
        # print p-value histogram
        #saveHistogram(filename=results_path+'p_values_histogram_from_filter_no_correction', values=p_values, title=stat_test_name)
        if saveGraphics:
            print('Saving graphics and p-values Histogram.\n')
            saveHistogram(filename=results_path+'histogram_p_values.png', values=p_values, title=stat_test_name, xlabel='p-values', ylabel='counts', bins=40, rwidth=0.9, color='#607c8e', grid=False, ygrid=True, alpha=0.75)

            train_dataset_temp = train_dataset.get_sub_dataset([train_dataset.genes[i] for i in range(len(train_dataset.genes)) if p_values[i]<cutoff])

            saveHeatMap(train_dataset_temp.get_scaled_data(), train_dataset_temp.samples, train_dataset_temp.genes, train_dataset_temp.Y(), results_path+'heatmap_train_dataset_correlation_filtered_zscore', metric='correlation', xticklabels=True, classes_labels=train_dataset_temp.levels()) 

            saveHeatMap(train_dataset_temp.get_scaled_data(), train_dataset_temp.samples, train_dataset_temp.genes, train_dataset_temp.Y(), results_path+'heatmap_train_dataset_euclidean_filtered_zscore', metric='euclidean', xticklabels=True, classes_labels=train_dataset_temp.levels()) 

            #saveScatterPlots(train_dataset_temp.getMatrixZscoreWithColClassAsDataFrame(), results_path+'scatterplots_filtered_zscore')   
        # ========= end graphics ===========                



        # =============== P-VALUE CORRECTION =============
        p_values_corrected = None
        filtered = None
        fdr_cutoff = args['fdrPvalue']
        print('Computing corrected p-values by FDR.\n')
        #correct p-values
        #print('\nP-values before correction: %s \n\n' % str(p_values))
        p_values_corrected = fdrcorrection(p_values, alpha=0.05, method='indep', is_sorted=False)[1]
        #print('P-values after correction: %s\n\n' % str(p_values_corrected))
        filtered = [train_dataset.genes[i] for i in range(len(train_dataset.genes)) if p_values_corrected[i]<fdr_cutoff]

        method = 'corrected_fdr.csv'
        scores = []
        for i in range(len(train_dataset.genes)):
            score = p_values_corrected[i]
            scores.append((abs(score), train_dataset.geneIndex(train_dataset.genes[i]), train_dataset.genes[i]))            
        scores = sorted(scores, reverse = False, key=lambda tup: tup[0])
        saveRank(scores, results_path+method)  

        print('\nSelected proteins IF FDR correction: %s\n\n' % str(filtered))

        report.write('P-values were corrected by FDR and these are the remaining proteins with p-value < %f IF "--fdr" is set: %s\n' % (fdr_cutoff, str(filtered)))
        report.flush()
        # ========= save graphics after FDR correction ===========
        if saveGraphics:
            saveHistogram(filename=results_path+'histogram_p_values_corrected_fdr.png', values=p_values_corrected, title=stat_test_name+' (corrected)', xlabel='p-values', ylabel='counts', bins=40, rwidth=0.9, color='#607c8e', grid=False, ygrid=True, alpha=0.75) 

            train_dataset_temp = train_dataset.get_sub_dataset([train_dataset.genes[i] for i in range(len(train_dataset.genes)) if p_values_corrected[i]<fdr_cutoff])

            saveHeatMap(train_dataset_temp.get_scaled_data(), train_dataset_temp.samples, train_dataset_temp.genes, train_dataset_temp.Y(), results_path+'heatmap_train_dataset_correlation_filtered_fdr_zscore', metric='correlation', xticklabels=True, classes_labels=train_dataset_temp.levels())

            saveHeatMap(train_dataset_temp.get_scaled_data(), train_dataset_temp.samples, train_dataset_temp.genes, train_dataset_temp.Y(), results_path+'heatmap_train_dataset_euclidean_filtered_fdr_zscore', metric='euclidean', xticklabels=True, classes_labels=train_dataset_temp.levels())
        # ================== end p-value correction ======================



        if option == 'filter':
            print('The option filter is set. Filtering by p-value.\n')
            # aply filter and create a new train/test data set
            current_cutoff = cutoff
            if args['fdr']:
                if len(filtered) >= args['minFdr']:
                    p_values = p_values_corrected
                    current_cutoff = fdr_cutoff
                else:
                    print('The number of filtered proteins is smaller than minFdr. The program was interrupted.')
                    exit()
                if len(filtered) < 2:
                    print('Interrupting the algorithm because the number of selected proteins after FDR filter is < 2')
                    exit()        
            # filter dataset by p-value (corrected or not, depending on the parameter --fdr)
            train_dataset = train_dataset.get_sub_dataset([train_dataset.genes[i] for i in range(len(train_dataset.genes)) if p_values[i]<current_cutoff])

            test_dataset = test_dataset.get_sub_dataset([test_dataset.genes[i] for i in range(len(test_dataset.genes)) if p_values[i]<current_cutoff])



        try:
            filename = results_path+'ratio_train_samples'
            save_ratio(train_dataset, filename)

            filename = results_path+'ratio_train_samples_normalized'
            save_ratio(train_dataset, filename, normalize=True)
        except:
            print(train_dataset.X())

        # =================== filter by CORRELATION ========================
        # remove/store the correlated features, letting only one of each correlated group in the dataset
        correlation = args['correlation']
        corr_threshold = args['corrThreshold']
        correlated_genes = []
        if correlation:       
            print('Filtering by Correlation.\n') 
            result = train_dataset.correlatedAttributes(threshold=corr_threshold)
            corr_matrix = abs(result['corr_matrix'])
            corr_pvalues = result['p_values_matrix']
            corr_pvalues_corrected = result['p_values_corrected_matrix']
            correlated_genes = result['correlated_genes']
            genes_to_drop = result['genes_to_drop']
            
            saveHeatMapScores(np.matrix(corr_matrix).astype(float), train_dataset.genes, train_dataset.genes, filename=results_path+'heatmap_abs_euclidean.png', metric='euclidean')
            
            saveHeatMapScores(np.matrix(corr_matrix).astype(float), train_dataset.genes, train_dataset.genes, filename=results_path+'heatmap_abs_correlation.png', metric='correlation')

            new_matrix = []
            for row in corr_matrix:           
                new_row = []
                for value in row:                                
                    if value < corr_threshold:
                        new_row.append(0.0)
                    else:
                        new_row.append(value)
                new_matrix.append(new_row)

            saveHeatMapScores(np.matrix(new_matrix).astype(float), train_dataset.genes, train_dataset.genes, results_path+'heatmap_abs_correlation_cuttoff_threshold.png', metric='euclidean')

            report.write('\nThe following Genes are correlated to another and will not be considered in the Machine Learning steps: %s\n\nCorrelated Genes:\n' % str(genes_to_drop))
            for gene in correlated_genes:
                report.write('%s:     %s\n' % (gene, str(list(correlated_genes[gene]))))

            report.write('\nCorrelations:\n\n')
            for gene1 in correlated_genes:
                for gene2 in correlated_genes[gene1]:
                    it, jt = train_dataset.geneIndex(gene1), train_dataset.geneIndex(gene2)
                    i = min([it,jt])
                    j = max([it,jt])
                    report.write(gene1 + ' * ' + gene2 + '= corr: ' +str(np.round(corr_matrix[i,j], decimals=3)) +  ', pvalue: ' + str(np.round(corr_pvalues[i,j],decimals=5)) + ', fdr: '+ str(np.round(corr_pvalues_corrected[i,j], decimals=5))+'\n')
            report.write('\n\n')
            report.flush()

            corr_genes_indexes = set()        
            for gene1 in correlated_genes:
                corr_genes_indexes.add(train_dataset.geneIndex(gene1))
                for gene2 in correlated_genes[gene1]:
                    corr_genes_indexes.add(train_dataset.geneIndex(gene2))

            corr_genes_indexes = list(corr_genes_indexes)
            corr_genes_names = np.array(train_dataset.genes)[corr_genes_indexes]
            correlated_correlation_matrix = np.matrix(corr_matrix).astype(float)
            ixgrid = np.ix_(corr_genes_indexes, corr_genes_indexes)
            correlated_correlation_matrix= correlated_correlation_matrix[ixgrid]

            if len(corr_genes_indexes) > 1:
                saveHeatMapScores(abs(correlated_correlation_matrix), corr_genes_names, corr_genes_names, filename=results_path+'heatmap_correlated_genes.png', metric='euclidean')

            # drop the correlated proteins

            train_dataset.saveFile(results_path+'train_including_correlated_genes.csv')
            test_dataset.saveFile(results_path+'test_including_correlated_genes.csv')
            report.write("\nNumber of proteins before removing correlated: %d" % len(train_dataset.genes))
            genes = [gene for gene in train_dataset.genes]
            for gene in genes_to_drop:
                genes.remove(gene)            
            train_dataset = train_dataset.get_sub_dataset(genes)
            test_dataset = test_dataset.get_sub_dataset(genes)
            report.write("\nNumber of proteins after removing correlated: %d" % len(train_dataset.genes))
            # ========= end graphics ===========

        if args['onlyStats']:
            print('\n The parameter onlyStats was set and the statistical results are ready.\nThe algorithm stops here.')
            report.close()
            exit()



        # ======================================================================
        # ================== RANKING / ATTRIBUTE SCORES ========================
        # ======================================================================

        test_dataset.saveFile(filename=results_path+'dataset_test_without_correlated_genes.csv')
        train_dataset.saveFile(filename=results_path+'dataset_train_without_correlated_genes.csv')
        if args['smote']:
            report.write("\n\n Number of samples before SMOTE: %d" % len(train_dataset.samples))
            train_dataset = train_dataset.getSmote(invert=True)
            train_dataset.saveFile(filename=results_path+'dataset_train_from_scrip_SMOTEs.csv')
            report.write("\n\n Number of samples after SMOTE: %d" % len(train_dataset.samples))

        report.write('Number of samples used in the ML: %d\n' % len(train_dataset.samples))
        report.write('Number of samples in the independent test dataset used in the ML: %d \n\n' % len(test_dataset.samples))
        report.write('\n\nNumber of features to be ranked: %d\n\n' % len(train_dataset.genes))
        report.flush()

            
        proteins_ranks = []
        ranks = {}

        k = args['innerK']
        if k > train_dataset.getMinNumberOfSamplesPerClass():
            print("\n-------------!!!-------------\n")
            print("Error: The defined K for K-fold Cross-validation is greater than the number of members in each class.\n")
            print("Please consider using a smaller number for the parameter --k\n")
            print("Current K: %d\n" % k)
            print("Max value for K: %d\n" % train_dataset.getMinNumberOfSamplesPerClass())
            exit()


        type1, type2, type3, type4, type5, type6, type7 = True, True, True, True, True, True, True
        # True, True, True, True, True, True, True
        #type1, type2, type3, type4, type5, type6, type7 = False, False, True, False, False, False, False
        #type1, type2, type3, type4, type5, type6, type7 = False, False, False, True, False, False, False


        # Benchmark of best parameter C for L1 and L2
        param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100]}
        pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'l1'))     
        grid = GridSearchCV(pipe, param_grid, cv = k)
        grid.fit(train_dataset.X(), train_dataset.Y())
        LassoBestC = grid.best_params_['logisticregression__C']


        pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'l2'))     
        grid = GridSearchCV(pipe, param_grid, cv = k)
        grid.fit(train_dataset.X(), train_dataset.Y())
        RidgeBestC = grid.best_params_['logisticregression__C']  

        estimators = [  #lambda_name='model__C',lambda_grid=np.logspace(-5, -1, 50)
            {'name': 'Linear SVM', 'model': LinearSVC(), 'lambda_name':'model__C', 'lambda_grid': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])}, # has decision_function

            {'name': 'Decision Tree', 'model': DecisionTreeClassifier(), 'lambda_name':'model__max_depth', 'lambda_grid': np.arange(1, 20)}, #'max_depth': np.arange(3, 10) #has predict_proba

            {'name': 'Random Forest', 'model': RandomForestClassifier(n_estimators=n_estimators,n_jobs=1), 'lambda_name':'model__max_features', 'lambda_grid': np.array([0.5, 0.75, 1.0])}, # predict_proba(X)

            #{'name': 'Ada Boost Decision Trees', 'model': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=n_estimators), 'lambda_name':'model__learning_rate', 'lambda_grid': np.array([0.01, 0.1, 0.3, 0.6, 1.0])}, # predict_proba(X)

            #{'name': 'Gradient Boosting', 'model': GradientBoostingClassifier(n_estimators=n_estimators, loss="deviance" ), 'lambda_name':'model__learning_rate', 'lambda_grid': np.array([0.01, 0.1, 0.3, 0.6, 1.0])}, #predict_proba(X)

            {'name': 'Lasso', 'model': LogisticRegression(penalty='l1', C=LassoBestC), 'lambda_name':'model__C', 'lambda_grid': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])}, #predict_proba(X)

            {'name': 'Ridge', 'model': LogisticRegression(penalty='l2', C=RidgeBestC), 'lambda_name':'model__C', 'lambda_grid': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])}, #predict_proba(X)

            {'name': 'Linear Discriminant Analysis', 'model': LinearDiscriminantAnalysis(), 'lambda_name':'model__n_components', 'lambda_grid': None} #! need to be set after filtering
            # predict_proba(X)
        ]


        def getScore(traindata, pipe, estimator_name, i):
            if estimator_name in ['Lasso', 'Ridge', 'Linear SVM', 'Linear Discriminant Analysis']:
                if len(traindata.levels()) == 2:                
                    return pipe.coef_[0][i]
                else:
                    return np.mean(pipe.coef_[:,i])
            else:
                return pipe.feature_importances_[i]


        # Type 1: Univariate Feature Selection (Statistics)
        # Type 2: Model Based Ranking
        # Type 3: Attributes' Weights
        # Type 4: Recursive Feature Elimination
        # Type 5: Stability Selection
        # Type 6: Decrease of Accuracy

        import re
        def reportTop10Proteins(report, scores, correlated, header):
            #! scores must be sorted
            top_scores = scores[0:10]
            genes = [item[2] for item in top_scores]        
            scores_values = [item[0] for item in top_scores]

            correlated_genes = set()
            for gene in genes:
                if gene in correlated:
                    for corr in correlated[gene]:
                        correlated_genes.add(corr)

            genes_str = re.sub(r'\[|\]|\(|\)|set','',str(genes))
            correlated_genes_str = re.sub(r'\[|\]|\(|\)|set','',str(correlated_genes))
            report.write(header+'\n')
            report.write('Min/Max scores of top-10 proteins: %s\n' % (str(np.min(scores_values))+'/'+str(np.max(scores_values))))
            report.write('Top-10 proteins:\n %s\n' % genes_str)
            report.write('Correlated genes: %s\n\n' % correlated_genes_str)
            report.flush()

        def sortSaveNormalizeAndSave(scores, path, filename, inverse=False):
            reverse = True
            if inverse:
                reverse = False

            scores = sorted(scores, reverse = reverse, key=lambda tup: tup[0])        
            #saveRank(scores, path + filename + '.csv')

            scores = normalizeScores(scores) 

            if inverse:        
                scores = [(1-score[0], score[1], score[2]) for score in scores]             
                
            #saveRank(scores, path + filename + '_normalizedMinMax.csv' )

            return scores



        # 
        # ---------------------- Type 1 - Univariate Ranks -------------------------
        if type1:
            print('\nExecuting Type 1\n')

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

                if args['tTest']:
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
                score = (p[i],  train_dataset.geneIndex(train_dataset.genes[i]),  train_dataset.genes[i])
                scores.append(score)

            filename = 'rank_t1_uni_'+name
            method_name = 't1_uni_'+name
            header = '-- Type 1 - Univariate by Statistical-test P-value - '+name+' --'
            #! inverse must be True, so that the function will do [1-p-value] to be the score
            scores = sortSaveNormalizeAndSave(scores, results_path_rank, filename, inverse=True)
            ranks[method_name] = scores     
            reportTop10Proteins(report, scores, correlated_genes, header)  
            #-----------------------------------------------------------------------------


            #---------- Chi-Squared ----------
            test = SelectKBest(score_func=chi2, k='all')
            x_train = train_dataset.X()

            y_train = factorize(train_dataset.labels)[0] 
            test.fit(x_train, y_train)
            
            scores = []
            for i in range(len(train_dataset.genes)):
                score = test.scores_[i]
                scores.append((score, train_dataset.geneIndex(train_dataset.genes[i]), train_dataset.genes[i]))

            filename = 'rank_t1_uni_chi_squared'
            method_name = 't1_uni_chi_squared'
            header = '-- Type 1 - Univariate by Statistical-test - Chi-Squared --'
            scores = sortSaveNormalizeAndSave(scores, results_path_rank, filename)
            ranks[method_name] = scores     
            reportTop10Proteins(report, scores, correlated_genes, header)   



            #---------- Mutual Information ----------
            test = SelectKBest(score_func = mutual_info_classif, k='all')
            x_train = train_dataset.matrix
            y_train = factorize(train_dataset.labels)[0] 
            test.fit(x_train, y_train)
            
            scores = []
            for i in range(len(train_dataset.genes)):
                score = test.scores_[i]
                scores.append((score, train_dataset.geneIndex(train_dataset.genes[i]), train_dataset.genes[i]))

            filename = 'rank_t1_uni_mutual_inf'
            method_name = 't1_uni_mutual_inf'
            header = '-- Type 1 - Univariate by Statistical-test - Mutual Information --'
            scores = sortSaveNormalizeAndSave(scores, results_path_rank, filename)
            ranks[method_name] = scores     
            reportTop10Proteins(report, scores, correlated_genes, header)    


            #---------- ANOVA F-value ----------
            test = SelectKBest(score_func = mutual_info_classif, k='all')
            x_train = train_dataset.matrix
            y_train = factorize(train_dataset.labels)[0] 
            test.fit(x_train, y_train)
            
            scores = []
            for i in range(len(train_dataset.genes)):
                score = test.scores_[i]
                scores.append((score, train_dataset.geneIndex(train_dataset.genes[i]), train_dataset.genes[i]))
            
            filename = 'rank_t1_uni_anova_fvalue'
            method_name = 't1_uni_anova_fvalue'
            header = '-- Type 1 - Univariate by Statistical-test - ANOVA F-value --'
            scores = sortSaveNormalizeAndSave(scores, results_path_rank, filename)
            ranks[method_name] = scores     
            reportTop10Proteins(report, scores, correlated_genes, header)   




        # ---------------------- Type 2 - Model Based Ranks -------------------------
        if type2:

            def singularAttributeScore(estimator, train_data):                
                name = estimator['name']
                method = name.lower().replace(" ","_")

                scores = []
                for i in range(len(train_dataset.genes)):
                    x_train = train_dataset.X()[:, i] # data matrix
                    y_train = train_dataset.Y()  # classes/labels of each sample from 

                    base_estimator = Pipeline([('scaler', StandardScaler()),
                                                 ('model', clone(estimator['model']))])
                    score = np.mean(cross_val_score(base_estimator, x_train, y_train, cv = k, scoring=scoreEstimator, n_jobs=nJobs))
                    scores.append((abs(score), train_dataset.geneIndex(train_dataset.genes[i]), train_dataset.genes[i]))

                filename = 'rank_t2_model_based_'+method
                method_name = 't2_model_based_rank_'+method
                header = '-- Type 2 - Model Based (per attribute) - '+name+' --'
                scores = sortSaveNormalizeAndSave(scores, results_path_rank, filename)
                ranks[method_name] = scores     
                reportTop10Proteins(report, scores, correlated_genes, header)  

        for estimator in estimators:
            print('Model based (1 per feature): %s' % estimator['name'])
            singularAttributeScore(estimator, train_dataset)                  
        

        # ---------------------- Type 3 - Rank based on attribute Weights -------------------------
        if type3:
            print('\nExecuting Type 3\n')
            def regularizationRank(estimator, traindata):
                model = clone(estimator['model'])
                name = estimator['name']
                method = name.lower().replace(" ","_")

                pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])           
                pipe.fit(traindata.X(), traindata.Y())
                scores = []
                for i in range(len(train_dataset.genes)):
                    score = getScore(traindata, pipe.named_steps['clf'], name, i)
                    scores.append((abs(score), train_dataset.geneIndex(train_dataset.genes[i]), train_dataset.genes[i]))       

                filename = 'rank_t3_weights_'+method
                method_name = 't3_weights_rank_'+method
                header = '-- Type 3 - Attributes\' Weights - '+name+' --'
                scores = sortSaveNormalizeAndSave(scores, results_path_rank, filename)
                ranks[method_name] = scores     
                reportTop10Proteins(report, scores, correlated_genes, header)  


            for estimator in estimators:
                print('Attribute Weights: %s' % estimator['name'])
                regularizationRank(estimator, train_dataset)
            


        #---------- Type 4 - Recursive Feature Elimination (RFE) ----------
        if type4:
            print('\nExecuting Type 4\n')
            def rfeRank(estimator, traindata):        
                model = clone(estimator['model'])
                name = estimator['name']
                method = name.lower().replace(" ","_")
                        
                rfe = RFE(model, n_features_to_select=1)
                rfe.fit(StandardScaler().fit_transform(traindata.X()),traindata.Y())
                scores = []        
                for i in range(len(traindata.genes)):
                    score = float(rfe.ranking_[i])
                    scores.append((abs(score), train_dataset.geneIndex(traindata.genes[i]), traindata.genes[i]))          

                filename = 'rank_t4_rfe_'+method
                method_name = 't4_rfe_'+method
                header = '-- Type 4 - Recursive Feature Elimination - '+name+' --'
                scores = sortSaveNormalizeAndSave(scores, results_path_rank, filename, inverse=True)
                ranks[method_name] = scores     
                reportTop10Proteins(report, scores, correlated_genes, header)               

            for estimator in estimators:
                print('RFE: '+estimator['name'])
                rfeRank(estimator, train_dataset)
            print("\n")

        
        #---------- Type 5 - Stability Selection ----------
        if type5:
            print('\nExecuting Type 5\n')
            from stability_selection import StabilitySelection  
            from sklearn.model_selection import StratifiedShuffleSplit      

            def stratified_subsampling(y, n_subsamples, random_state=7):

                if n_subsamples < 2*len(np.unique(y)):
                    n_subsamples = 2*len(np.unique(y))
                    if n_subsamples > len(y):
                        raise Exception('Number of sample is too small to run stability selection.')

                test_size = len(y)-n_subsamples
                if test_size < len(np.unique(y)):
                    test_size = len(np.unique(y))
                test_size = test_size/float(len(y))
                sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                x_indexes = []
                for i in range(len(y)):
                    x_indexes.append([i])                
                for train_indexes, test_indexes in sss.split(x_indexes, y):                     
                    return sorted(train_indexes)


            def stabilitySelectionScores(estimator, traindata): 
                if estimator['name'] == 'Linear Discriminant Analysis':
                    estimator['lambda_grid'] = np.arange(2, len(traindata.genes), 3)
                base_estimator = Pipeline([('scaler', StandardScaler()), ('model', clone(estimator['model']))])
                name = estimator['name']  
                selector = None
                try:                
                    selector = StabilitySelection(base_estimator=base_estimator, sample_fraction=0.75, lambda_name=estimator['lambda_name'], lambda_grid=estimator['lambda_grid'], n_jobs=nJobs, bootstrap_func=stratified_subsampling).fit(traindata.X(), traindata.Y()) #
                except Exception as e:
                    message = None
                    if hasattr(e, 'message'):
                        message = e.message
                    else:
                        message = str(e)
                    report.write('-- Type 5 -- Stability Selection - '+name+' --\n')
                    report.write('!!! - Error: could not run '+name+' because of the following error: '+ message)
                    print('!!! - Error: could not run '+name+' because of the following error: '+ message+ '\nThe script will continue computing the ranks based on next Estimator.')
                    print("Y data for stability selection: %s" % str( traindata.Y()))
                    return None

                
                scores = []             
                method = name.lower().replace(" ","_")     
                for i in range(len(traindata.genes)):                
                    score = np.mean(selector.stability_scores_[i])
                    scores.append((abs(score), train_dataset.geneIndex(traindata.genes[i]), traindata.genes[i]))
                    
                filename = 'rank_t5_stability_'+method
                method_name = 't5_stability_'+method
                header = '-- Type 5 -- Stability Selection - '+name+' --'
                scores = sortSaveNormalizeAndSave(scores, results_path_rank, filename)
                ranks[method_name] = scores     
                reportTop10Proteins(report, scores, correlated_genes, header)                

            for estimator in estimators:
                print('Stability selection: %s' % estimator['name'])
                stabilitySelectionScores(estimator, train_dataset)

    
        #---------- Type 6 - Mean Decrease Accuracy ----------
        if type6:
            print('\nExecuting Type 6\n')
            def meanDecreaseAccuracyScore(estimator, traindata):
                X = traindata.X()
                y = traindata.Y()
                scores = []
                name = estimator['name']  
                method = name.lower().replace(" ","_")  
                base_estimator = Pipeline([('scaler', StandardScaler()), ('model', clone(estimator['model']))])
                score_normal = np.mean(cross_val_score(base_estimator, X, y, cv = k, scoring=scoreEstimator, n_jobs=nJobs))
                
                for i in range(len(traindata.genes)):
                    X_shuffled = X.copy()
                    scores_shuffle = []
                    for j in range(3):
                        np.random.seed(j*3)
                        np.random.shuffle(X_shuffled[:,i])
                        base_estimator = Pipeline([('scaler', StandardScaler()), ('model', clone(estimator['model']))])
                        score = np.mean(cross_val_score(base_estimator, X_shuffled, y, cv = k, scoring=scoreEstimator, n_jobs=nJobs))
                        scores_shuffle.append(score)               
                    gene_name = traindata.genes[i]
                    scores.append((score_normal - np.mean(scores_shuffle), train_dataset.geneIndex(gene_name), gene_name))                    

                filename = 'rank_t6_decrease_acc_'+method
                method_name = 't6_decrease_acc_'+method
                header = '-- Type 6 --- Mean Decrease Accuracy - '+name+' --'
                scores = sortSaveNormalizeAndSave(scores, results_path_rank, filename)
                ranks[method_name] = scores     
                reportTop10Proteins(report, scores, correlated_genes, header)     
            
            for estimator in estimators:            
                print('Decrease Accuracy: %s' % estimator['name'])
                meanDecreaseAccuracyScore(estimator, train_dataset)



        #---------- Type 7 - Recursive Feature Elimination (RFA) ----------
        if type7:
            print('\nExecuting Type 7\n')
            def rfaRank(estimator, traindata):        
                model = clone(estimator['model'])
                name = estimator['name']
                method = name.lower().replace(" ","_")
                        
                rfa = RFA(model, n_features_to_select=1)
                rfa.fit(StandardScaler().fit_transform(traindata.X()),traindata.Y())
                scores = []        
                for i in range(len(traindata.genes)):
                    score = float(rfa.ranking_[i])
                    scores.append((abs(score), train_dataset.geneIndex(traindata.genes[i]), traindata.genes[i]))          

                filename = 'rank_t7_rfa_'+method
                method_name = 't7_rfa_'+method
                header = '-- Type 7 - Recursive Feature Addition - '+name+' --'
                scores = sortSaveNormalizeAndSave(scores, results_path_rank, filename, inverse=True)
                ranks[method_name] = scores     
                reportTop10Proteins(report, scores, correlated_genes, header)               

            for estimator in estimators:
                
                rfaRank(estimator, train_dataset)
                print('RFA: '+estimator['name'])
                # except Exception:
                #     print("There was a problem performing RFA: "+estimator['name'] +" and it was not computed. Please try again if this rank in particular is important.")
            print("\n")



        # ------------ Saving ranks Tables --------------------
        #! scores lists must be already sorted
        # Protein names are ordered by rank
        matrix = []  
        for name in sorted(ranks.keys()):
            rank = ranks[name]              
            row = [name] + [score[2] for score in sorted(rank, key=lambda tup: tup[0], reverse=True)]
            matrix.append(row)
        matrix = np.matrix(matrix).transpose()    
        filename = results_path+'all_ranks_prot_names.csv'
        np.savetxt(filename, matrix, delimiter=",", fmt='%s')


        # Protein names are listed, values of the matrix are their Scores
        matrix = [['method']+train_dataset.genes]
        for name in sorted(ranks.keys()):
            rank = ranks[name]                  
            order_by_index = sorted(rank, key=lambda tup: tup[1])
            values = [score[0] for score in order_by_index]
            row = [name] + values
            matrix.append(row)        
        matrix = np.matrix(matrix).transpose()   
        # mean_column = ['mean'] + np.array(matrix[1:,1:]).astype(float).mean(axis=1).tolist()
        # std_column = ['std'] +   np.array(matrix[1:,1:]).astype(float).std(axis=1).tolist()
        # new_matrix = np.matrix([mean_column,std_column]).transpose()
        # matrix = np.concatenate((matrix, new_matrix), axis=1)    
        # sorted_matrix = matrix[1:,:]
        # sorted_matrix.sort(axis=-2)
        # matrix = np.concatenate((matrix[0,:],sorted_matrix[::-1]), axis=0) #! Score: the greater the value, the more important is the protein

        filename = results_path+'all_ranks_scores.csv'
        np.savetxt(filename, matrix, delimiter=",", fmt='%s')

        # save Heatmap of scores    
        row_lables = np.squeeze(np.asarray(matrix[1:,0]))
        cols_labels=np.squeeze(np.asarray(matrix[0,1:]))
        num_matrix = matrix[1:,1:].astype(float)

        

        filename = results_path+'all_ranks_scores_heatmap_euclidean.png'
        df_matrix = saveHeatMapScores(num_matrix, rows_labels=row_lables , cols_labels=cols_labels, filename=filename, metric='euclidean')
        score_matrices.append(df_matrix)

        fold_proteins_scores = df_matrix

        filename = results_path+'all_ranks_scores_heatmap_euclidean_cutoff_0.8.png'
        num_matrix_cutoff = num_matrix.copy()
        np.maximum(num_matrix_cutoff, 0.8, num_matrix_cutoff)
        saveHeatMapScores(num_matrix_cutoff, rows_labels=row_lables , cols_labels=cols_labels, filename=filename, metric='euclidean')

        filename = results_path+'all_ranks_scores_heatmap_euclidean_cutoff_0.9.png'
        num_matrix_cutoff = num_matrix.copy()
        np.maximum(num_matrix_cutoff, 0.9, num_matrix_cutoff)
        saveHeatMapScores(num_matrix_cutoff, rows_labels=row_lables , cols_labels=cols_labels, filename=filename, metric='euclidean')

        filename = results_path+'all_ranks_scores_heatmap_euclidean_cutoff_0.7.png'
        num_matrix_cutoff = num_matrix.copy()
        np.maximum(num_matrix_cutoff, 0.7, num_matrix_cutoff)
        saveHeatMapScores(num_matrix_cutoff, rows_labels=row_lables , cols_labels=cols_labels, filename=filename, metric='euclidean')


        #filename = results_path+'all_ranks_scores_heatmap_correlation.png'
        #saveHeatMapScores(num_matrix, rows_labels=row_lables , cols_labels=cols_labels, filename=filename, metric='correlation')

        # Protein names are listed, values of the matrix are their Positions/Ranks
        matrix = [['method']+train_dataset.genes]
        for name in sorted(ranks.keys()):
            rank = ranks[name]    
            new_scores = [(rank[i][0], rank[i][1], rank[i][2], i) for i in range(len(rank))]
            values = [score[3] for score in sorted(new_scores, key=lambda tup: tup[1])]
            row = [name] + values
            matrix.append(row)
        matrix = np.matrix(matrix).transpose()
        # mean_column = ['mean'] + np.array(matrix[1:,1:]).astype(int).mean(axis=1).tolist()
        # std_column = ['std'] +   np.array(matrix[1:,1:]).astype(int).std(axis=1).tolist()
        # new_matrix = np.matrix([mean_column,std_column]).transpose()
        # matrix = np.concatenate((matrix, new_matrix), axis=1)    
        # sorted_matrix = matrix[1:,:]
        # sorted_matrix.sort(axis=-2)
        # matrix = np.concatenate((matrix[0,:],sorted_matrix), axis=0)  #! Position: the lower the value, the more important is the protein

        filename = results_path+'all_ranks_positions.csv'
        np.savetxt(filename, matrix, delimiter=",", fmt='%s')    

        # save Heatmap of positions    
        row_lables = np.squeeze(np.asarray(matrix[1:,0]))
        cols_labels=np.squeeze(np.asarray(matrix[0,1:]))
        num_matrix = matrix[1:,1:].astype(int)


        cmap = "Blues"
        cmap = 'vlag_r'

        filename = results_path+'all_ranks_positions_heatmap_euclidean.png'
        saveHeatMapScores(num_matrix, rows_labels=row_lables , cols_labels=cols_labels, filename=filename, metric='euclidean', colors=cmap)

        if len(train_dataset.genes) > 3:
            filename = results_path+'all_ranks_positions_heatmap_euclidean_cutoff_3.png'
            num_matrix_cutoff = num_matrix.copy()
            np.minimum(num_matrix_cutoff, 3, num_matrix_cutoff)
            saveHeatMapScores(num_matrix_cutoff, rows_labels=row_lables , cols_labels=cols_labels, filename=filename, metric='euclidean', colors=cmap)

        if len(train_dataset.genes) > 5:
            filename = results_path+'all_ranks_positions_heatmap_euclidean_cutoff_5.png'
            num_matrix_cutoff = num_matrix.copy()
            np.minimum(num_matrix_cutoff, 5, num_matrix_cutoff)
            saveHeatMapScores(num_matrix_cutoff, rows_labels=row_lables , cols_labels=cols_labels, filename=filename, metric='euclidean', colors=cmap)

        if len(train_dataset.genes) > 10: 
            filename = results_path+'all_ranks_positions_heatmap_euclidean_cutoff_10.png'
            num_matrix_cutoff = num_matrix.copy()
            np.minimum(num_matrix_cutoff, 10, num_matrix_cutoff)
            saveHeatMapScores(num_matrix_cutoff, rows_labels=row_lables , cols_labels=cols_labels, filename=filename, metric='euclidean', colors=cmap)

        if len(train_dataset.genes) > 15: 
            filename = results_path+'all_ranks_positions_heatmap_euclidean_cutoff_15.png'
            num_matrix_cutoff = num_matrix.copy()
            np.minimum(num_matrix_cutoff, 15, num_matrix_cutoff)
            saveHeatMapScores(num_matrix_cutoff, rows_labels=row_lables , cols_labels=cols_labels, filename=filename, metric='euclidean', colors=cmap)



        freq_prot = {}
        for gene in train_dataset.genes:
            freq_prot[gene] = 0

        for name in sorted(ranks.keys()):
            rank = ranks[name][0:10]
            for score in rank:
                gene = score[2]
                if gene in freq_prot:
                    freq_prot[gene]+=1
        
        freq_prot_list.append(freq_prot)

        matrix = []
        for gene in freq_prot.keys():
            matrix.append([gene, freq_prot[gene]/float(len(ranks.keys()))])

        matrix = sorted(matrix, key=lambda tup: tup[1], reverse=True)
        filename = results_path+'top_10_prot_frequency.csv'
        np.savetxt(filename, matrix, delimiter=",", fmt='%s')


        if correlation:
            filename = results_path+'correlated_genes.csv'
            max_n_correlated_genes = 0
            for gene in correlated_genes:
                if len(correlated_genes[gene]) > max_n_correlated_genes:
                    max_n_correlated_genes = len(correlated_genes[gene])                    

            with open(filename, 'w') as csv_file:
                for gene in correlated_genes.keys():
                    line = gene
                    genes = list(correlated_genes[gene])
                    for i in range(max_n_correlated_genes):
                        if i < len(genes):
                            line += ','+genes[i]
                        else:
                            line += ','                                       
                    line+='\n'
                    csv_file.write(line)
                csv_file.close()



        # --------------------- saving box plots -----------------
        freq_list = []
        for gene in freq_prot.keys():
            if freq_prot[gene] > len(ranks.keys())/3:   # > num_rankings * 1/3  * k_outer
                freq_list.append((freq_prot[gene], gene))
        freq_list = sorted(freq_list, reverse=True)
        selected_genes = [item[1] for item in freq_list]

        scores_by_gene = {}
        labels = row_lables.tolist()
        for gene in selected_genes:        
            values = []                 
            if gene in fold_proteins_scores.index.tolist():            
                for value in fold_proteins_scores.loc[gene].values:
                    values.append(value)
            scores_by_gene[gene] = values
        box_plot_values = [scores_by_gene[gene] for gene in selected_genes]
        filename = results_path + 'box_plot_top_10_appear_in_more_than_33p_of_ranks.png'
        saveBoxplots(box_plot_values, filename=filename, x_labels=selected_genes)
        # Boxplot that shows the scores from genes that appear in more than 1/3 of the ranks among the top-10


        scores_by_gene = {}
        max_n = len(ranks.keys())/2
        labels = row_lables.tolist()
        for gene in selected_genes:        
            values = []                 
            if gene in fold_proteins_scores.index.tolist():
                values_score = sorted(fold_proteins_scores.loc[gene].values, reverse=True)[0:max_n]
                for value in values_score:
                    values.append(value)
            scores_by_gene[gene] = values
        box_plot_values = [scores_by_gene[gene] for gene in selected_genes]
        filename = results_path + 'box_plot_top_10_from_50p_higher_scores_in_more_than_33p_ranks.png'
        saveBoxplots(box_plot_values, filename=filename, x_labels=selected_genes)



    time_message = '\n\nIt took %s to complete the script.\n' % str(datetime.now()-starting_time )
    print(time_message)
    report.write(time_message)

    report.close()







all_genes = set()
for score_m in score_matrices:
    for gene in score_m.index.tolist():
        all_genes.add(gene)
all_genes_list = list(all_genes)

first_dataframe = score_matrices[0]
methods = first_dataframe.columns.tolist()

first_m = first_dataframe.values
n_ranks = first_m.shape[1]
m_size = (len(all_genes),n_ranks)

print("m_size: ")
print(m_size)

values_matrix = []
for i in range(len(all_genes)):
    row = []
    for j in range(n_ranks):
        row.append([])
    values_matrix.append(row)  

mean_df = DataFrame(np.full(m_size,-0.5), index=all_genes_list, columns=first_dataframe.columns)
std_df = DataFrame(np.full(m_size,-0.5), index=all_genes_list, columns=first_dataframe.columns)

for gene in all_genes_list:
    for method in first_dataframe.columns:
        for score_df in score_matrices:
            if gene in score_df.index.tolist():
                values_matrix[all_genes_list.index(gene)][methods.index(method)].append(score_df[method][gene])                

for gene in all_genes_list:
    for method in first_dataframe.columns:
        values = values_matrix[all_genes_list.index(gene)][methods.index(method)]            
        mean_df[method][gene] = np.mean(values)
        std_df[method][gene] = np.std(values)

row_lables = first_dataframe.index
cols_labels = first_dataframe.columns

filename = results_path_parent + 'scores_mean_heatmap.png'
matrix = np.matrix(mean_df.values).astype(float)
saveHeatMapScores(matrix, rows_labels=mean_df.index , cols_labels=mean_df.columns, filename=filename, metric='euclidean')

df = DataFrame(matrix, index=mean_df.index, columns=mean_df.columns)
filename = results_path_parent + 'scores_mean.csv'
df.to_csv(filename, header=True)


filename = results_path_parent + 'scores_std_heatmap.png'
matrix = np.matrix(std_df.values).astype(float)
saveHeatMapScores(matrix, rows_labels=std_df.index , cols_labels=std_df.columns, filename=filename, metric='euclidean')

filename = results_path_parent + 'scores_lower_conf_level_heatmap.png'
matrix_mean = np.matrix(mean_df.values).astype(float)
matrix_std = np.matrix(std_df.values).astype(float)
matrix_conf = matrix_mean - (2*matrix_std)
matrix_conf[matrix_conf<0]=0.0
saveHeatMapScores(matrix_conf, rows_labels=std_df.index , cols_labels=std_df.columns, filename=filename, metric='euclidean')

df = DataFrame(matrix_conf, index=std_df.index, columns=std_df.columns)
filename = results_path_parent + 'scores_lower_conf_level.csv'
df.to_csv(filename, header=True)

high_score_df = DataFrame(np.full((len(all_genes),len(score_matrices)),-0.5), index=all_genes_list, columns=range(len(score_matrices)))
for gene in all_genes_list:    
    for j in range(len(score_matrices)):
        score_df = score_matrices[j]
        if gene in score_df.index.tolist():
            highest_scores = sorted(score_df.loc[gene].values, reverse=True)
            highest_scores = highest_scores[0:len(score_df.index.tolist())/2]
            high_score_df[j][gene] = np.mean(highest_scores)
        else:
            high_score_df[j][gene] = 0.0

filename = results_path_parent + 'scores_highest_50_mean_heatmap.png'
matrix = np.matrix(high_score_df.values).astype(float)
saveHeatMapScores(matrix, rows_labels=high_score_df.index , cols_labels=high_score_df.columns, filename=filename, metric='euclidean')



freq_genes_set = set()
for freq_prot in freq_prot_list:
    for gene in freq_prot.keys():
       freq_genes_set.add(gene)

matrix = []
for gene in freq_genes_set:
    row = [gene]
    for freq_prot in freq_prot_list:
        if gene in freq_prot:
            row.append(freq_prot[gene])
        else:
            row.append(0.0)
    matrix.append(row)
header = ['Protein'] + range(len(freq_prot_list))
df = DataFrame(matrix, columns=header)         
filename = results_path_parent + 'top10_prot_freq_from_each_fold.csv'
df.to_csv(filename, header=True)




global_freq = {}
for freq_prot in freq_prot_list:
    for gene in freq_prot.keys():
        if gene in global_freq:
            global_freq[gene] += freq_prot[gene]
        else:
            global_freq[gene] = freq_prot[gene]

freq_list = []
for gene in global_freq.keys():
    if global_freq[gene] > (first_m.shape[1]/3)*k_outer:   # > num_rankings * 1/3  * k_outer
        freq_list.append((global_freq[gene], gene))

freq_list = sorted(freq_list, reverse=True)
selected_genes = [item[1] for item in freq_list]

scores_by_gene = {}
labels = row_lables.tolist()
for gene in selected_genes:        
    values = []
    pen = -0.05
    for score_df in score_matrices:        
        if gene in score_df.index.tolist():            
            for value in score_df.loc[gene].values:
                values.append(value)            
        else:
            #for j in range(first_m.shape[1]):        
            values.append(pen)
            pen -= 0.001
    scores_by_gene[gene] = values

box_plot_values = [scores_by_gene[gene] for gene in selected_genes]
filename = results_path_parent + 'box_plot_top_10_from_each_loop_appear_in_more_than_33p_of_ranks.png'
saveBoxplots(box_plot_values, filename=filename, x_labels=selected_genes)
# Boxplot that shows the scores from genes that appear in more than 1/3 of the ranks among the top-10


scores_by_gene = {}
labels = row_lables.tolist()
max_n = n_ranks/2
df = DataFrame(np.full((len(all_genes_list),len(score_matrices)),-0.1), index=all_genes_list, columns=range(k_outer))

for gene in all_genes:        
    values = []
    pen = -0.05   
    for j in range(len(score_matrices)):    
        score_df = score_matrices[j]    
        if gene in score_df.index.tolist():            
            values_score = sorted(score_df.loc[gene].values, reverse=True)[0:max_n]
            print(values_score)
            df[j][gene] = np.mean(values_score)
            for value in values_score:
                values.append(value)            
        else:           
            values.append(pen)
            pen -= 0.001
    scores_by_gene[gene] = values

box_plot_values = [scores_by_gene[gene] for gene in selected_genes]
filename = results_path_parent + 'box_plot_top_10_from_each_loop_50p_higher_scores_per_loop.png'
saveBoxplots(box_plot_values, filename=filename, x_labels=selected_genes)

matrix = np.matrix(df.values.astype(float))
filename = results_path_parent + 'scores_mean_50p_higher_pergene_perloop_heatmap.png'
saveHeatMap(matrix, rows_labels=df.index , cols_labels=df.columns, classes=None, filename=filename, metric='euclidean')


#  ================ END MAIN ===============
