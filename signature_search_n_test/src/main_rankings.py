
# =================================
# =========== ARGS ================
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# import the necessary packages
import argparse

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
ap.add_argument('--testSize', help='If --test is not defined, --test_size is used to define the independent test set size. If --test_size is set to 0.0, then the independent test is not performed; that is, only the CV is performed to evaluate the selected signatures.', action='store', type=float, default=-1)

ap.add_argument('--nSSearch', help='Set the maximum number of proteins to search for small signatures formed by all prot. combinations (signature size).', action='store', type=int, default=3)

ap.add_argument('--nSmall', help='Set the number of proteins considered small. If the total number of proteins in a dataset is smaller or equal than NSMALL, it will compute all combinations of proteins to form signatures. Otherwise, it will consider NSSEARCH to compute only combinations of size up to the value set for this parameter.', action='store', type=int, default=10)

ap.add_argument('--topN', help='Create all combinations of top-N signatures from the average of ranks.', action='store', type=int, default=10)

ap.add_argument('--deltaRankCutoff', help='The percentage of difference from the maximum score value that is used as cutoff univariate ranks. The scores are normalized between 0 and 1. So, if the maximum value is 0.9, and deltaRankCutoff is set to 0.05, the cutoff value is 0.85. Proteins with score >= 0.85 are selected to form signatures by top-N proteins.', action='store', type=float, default=0.10)

ap.add_argument('--k', help='The value of K for all k-fold cross-validations.', action='store', type=int, default=10)

ap.add_argument('--limitSignatureSize', help='Limit the size of signatures created by rank to the number of samples. For instance, when selecting top-N proteins using 30 samples, the maximum value of N is 30.', action='store_true')

ap.add_argument('--nJobs', help='Number of parallel jobs when fitting models.', action='store', type=int, default=1)

ap.add_argument('--ignoreWarnings', help='Stop printing Warnings from models.', action='store_true')

ap.add_argument('--correlation', help='Filter variables highly correlated.', action='store_true')

ap.add_argument('--corrThreshold', help='Cuttoff for correlation.', action='store', type=float, default=0.95)

ap.add_argument('--positiveClass', help='The positive class name.', action='store')

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
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import numpy as np
from utils import saveRank, normalizeScores, getMaxNumberOfProteins, saveHeatMapScores
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize

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

scoreEstimator = None


# =========== Score estimator ===================
from sklearn.metrics import make_scorer, matthews_corrcoef
matthews_scorer = make_scorer(matthews_corrcoef)

uni, counts = np.unique(dataset.Y(), return_counts=True)

if len(dataset.levels()) == 2 and counts[0] == counts[1]:
    scoreEstimator = 'roc_auc'
    scoreEstimatorInfo = """
    """
elif len(dataset.levels()) == 3 or len(dataset.levels()) == 2:
    scoreEstimator = matthews_scorer#'matthews_corrcoef'
    scoreEstimatorInfo = """
    
    Chosen score estimator for imbalanced classes.

    The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary (two-class) classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.

    A correlation of:
            C =  1 indicates perfect agreement,
            C =  0 is expected for a prediction no better than random, and
            C = -1 indicates total disagreement between prediction and observation.
    
    """
else:
    print('\nDataset with more than 3 classes is not supported.\n\n')
    exit()

#pos_label=positive_class_index


#===================================





geneNames = [gene for gene in dataset.genes]

test_dataset = None
train_dataset = None
test_dataset_path = args['test']

report =  open(results_path+'report.txt','w')
report.write('============= REPORT =============\n\n')
report.write('Arguments used in this project: %s\n\n' % str(args))

saveGraphics = args['saveGraphics']
nJobs = args['nJobs']

report.write('\n\n Score used for rankings with Classifiers is: %s\n' % scoreEstimator)
report.write("Info about score: %s: \n\n" % scoreEstimatorInfo)

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

    if test_size_scale == 0:
        #copy dataset, train and test are going to be the same.
        train_dataset = dataset.get_sub_dataset_by_samples(range(len(dataset.samples)))
        test_dataset = dataset.get_sub_dataset_by_samples(range(len(dataset.samples))) 
    else:
        total_size = len(dataset.samples)    

        x_indexes = []
        for i in range(total_size):
            x_indexes.append([i])    
        X_train, X_test, y_train, y_test = train_test_split(x_indexes, dataset.Y(), test_size=test_size_scale, shuffle=True, stratify=dataset.Y())

        train_indexes = [v[0] for v in X_train]
        test_indexes = [v[0] for v in X_test]

        train_dataset = dataset.get_sub_dataset_by_samples(train_indexes)
        test_dataset = dataset.get_sub_dataset_by_samples(test_indexes)

    if saveGraphics:
        saveHeatMap(train_dataset.matrix, train_dataset.samples, train_dataset.genes, results_path+'heatmap_train_dataset_correlation', metric='correlation')

        saveHeatMap(train_dataset.get_scaled_data(), train_dataset.samples, train_dataset.genes, results_path+'heatmap_train_dataset_correlation_zscore', metric='correlation')   

        saveHeatMap(train_dataset.matrix, train_dataset.samples, train_dataset.genes, results_path+'heatmap_train_dataset_euclidean', metric='euclidean')

        saveHeatMap(train_dataset.get_scaled_data(), train_dataset.samples, train_dataset.genes, results_path+'heatmap_train_dataset_euclidean_zscore', metric='euclidean')    

positive_class_name = None
positive_class_index = None
if len(dataset.levels()) == 2:
    if args['positiveClass'] == None:
        print("\n\nYour dataset has two classes, thus --positiveClass must be informed.\n\n")
        exit()
    else:
        positive_class_name = args['positiveClass']
        positive_class_index = train_dataset.levels().tolist().index(positive_class_name)
        report.write("\nPositive class: %s\n: " % positive_class_name)
        report.write("Positive class index: %d\n: " % positive_class_index)
        report.write("Classes from train: %s:\n\n\n" % str(train_dataset.Y()))


report.write('Number of samples in the train dataset: %d\n' % len(train_dataset.samples))
report.write('Number of samples in the independent test dataset: %d \n\n' % len(test_dataset.samples))

report.write('Train samples: %s\n\n' % str(train_dataset.samples))
report.write('Test samples: %s\n\n' % str(test_dataset.samples))

report.write('Number of classes: %d\n' % len(train_dataset.levels()))
report.write('Classes: %s' % str(train_dataset.levels()))

if args['noFilter']:
    filter_options = ['noFilter']
elif args['onlyFilter']:
    filter_options = ['filter']
else:
    filter_options = ['filter', 'noFilter']


complete_train = train_dataset
complete_test = test_dataset

n_estimators = 30

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
    
        saveHeatMap(train_dataset_temp.matrix, train_dataset_temp.samples, train_dataset_temp.genes, results_path+'heatmap_train_dataset_correlation_filtered', metric='correlation', xticklabels=True)

        saveHeatMap(train_dataset_temp.get_scaled_data(), train_dataset_temp.samples, train_dataset_temp.genes, results_path+'heatmap_train_dataset_correlation_filtered_zscore', metric='correlation', xticklabels=True) 

        saveHeatMap(train_dataset_temp.matrix, train_dataset_temp.samples, train_dataset_temp.genes, results_path+'heatmap_train_dataset_euclidean_filtered', metric='euclidean', xticklabels=True)

        saveHeatMap(train_dataset_temp.get_scaled_data(), train_dataset_temp.samples, train_dataset_temp.genes, results_path+'heatmap_train_dataset_euclidean_filtered_zscore', metric='euclidean', xticklabels=True) 

        #saveScatterPlots(train_dataset_temp.getMatrixZscoreWithColClassAsDataFrame(), results_path+'scatterplots_filtered_zscore')   
    # ========= end graphics ===========                



    # =============== P-VALUE CORRECTION =============
    p_values_corrected = None
    filtered = None
    fdr_cutoff = args['fdrPvalue']
    print('Computing corrected p-values by FDR.\n')
    #correct p-values
    print('\nP-values before correction: %s \n\n' % str(p_values))
    p_values_corrected = fdrcorrection(p_values, alpha=0.25, method='indep', is_sorted=False)[1]
    print('P-values after correction: %s\n\n' % str(p_values_corrected))
    filtered = [train_dataset.genes[i] for i in range(len(train_dataset.genes)) if p_values_corrected[i]<fdr_cutoff]

    method = stat_test_name+'_corrected_fdr.csv'
    scores = []
    for i in range(len(train_dataset.genes)):
        score = p_values_corrected[i]
        scores.append((abs(score), train_dataset.geneIndex(train_dataset.genes[i]), train_dataset.genes[i]))            
    scores = sorted(scores, reverse = False, key=lambda tup: tup[0])
    saveRank(scores, results_path+method)  

    print('\nSelected proteins IF FDR correction: %s\n\n' % str(filtered))

    report.write('P-values were corrected by FDR and these are the remaining proteins with p-value < %f IF "--fdr" is set: %s\n' % (fdr_cutoff, str(filtered)))

    # ========= save graphics after FDR correction ===========
    if saveGraphics:
        saveHistogram(filename=results_path+'histogram_p_values_corrected_fdr.png', values=p_values_corrected, title=stat_test_name+' (corrected)', xlabel='p-values', ylabel='counts', bins=40, rwidth=0.9, color='#607c8e', grid=False, ygrid=True, alpha=0.75) 

        train_dataset_temp = train_dataset.get_sub_dataset([train_dataset.genes[i] for i in range(len(train_dataset.genes)) if p_values_corrected[i]<cutoff])

        saveHeatMap(train_dataset.matrix, train_dataset.samples, train_dataset.genes, results_path+'heatmap_train_dataset_correlation_filtered_fdr', metric='correlation', xticklabels=True)

        saveHeatMap(train_dataset.get_scaled_data(), train_dataset.samples, train_dataset.genes, results_path+'heatmap_train_dataset_correlation_filtered_fdr_zscore', metric='correlation', xticklabels=True)

        saveHeatMap(train_dataset.matrix, train_dataset.samples, train_dataset.genes, results_path+'heatmap_train_dataset_euclidean_filtered_fdr', metric='euclidean', xticklabels=True)

        saveHeatMap(train_dataset.get_scaled_data(), train_dataset.samples, train_dataset.genes, results_path+'heatmap_train_dataset_euclidean_filtered_fdr_zscore', metric='euclidean', xticklabels=True)
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


    # =================== filter by CORRELATION ========================
    # remove/store the correlated features, letting only one of each correlated group in the dataset
    correlation = args['correlation'] #TODO read from args
    corr_threshold = args['corrThreshold'] #TODO read from args
    correlated_genes = []
    if correlation:       
        print('Filtering by Correlation.\n') 
        result = train_dataset.correlatedAttributes(threshold=corr_threshold)
        corr_matrix = result['corr_matrix']
        correlated_genes = result['correlated_genes']
        genes_to_drop = result['genes_to_drop']
        
        saveHeatMap(np.matrix(corr_matrix).astype(float), train_dataset.genes, train_dataset.genes, xticklabels=train_dataset.genes, yticklabels=train_dataset.genes, filename=results_path+'heatmap_abs_euclidean.png', metric='euclidean')
        
        saveHeatMap(np.matrix(corr_matrix).astype(float), train_dataset.genes, train_dataset.genes, xticklabels=train_dataset.genes, yticklabels=train_dataset.genes, filename=results_path+'heatmap_abs_correlation.png', metric='correlation')

        new_matrix = []
        for row in corr_matrix:           
            new_row = []
            for value in corr_matrix[row]:                                
                if value < corr_threshold:
                    new_row.append(0.0)
                else:
                    new_row.append(value)
            new_matrix.append(new_row)

        saveHeatMap(np.matrix(new_matrix).astype(float), train_dataset.genes, train_dataset.genes, results_path+'heatmap_abs_correlation_cuttoff_threshold.png', metric='euclidean', xticklabels=True)

        report.write('\nThe following Genes are correlated to another and will not be considered in the Machine Learning steps: %s\n\nCorrelated Genes:\n' % str(genes_to_drop))
        for gene in correlated_genes:
            report.write('%s:     %s\n' % (gene, str(list(correlated_genes[gene]))))
        report.write('\n\n')

        # drop the correlated proteins
        genes = [gene for gene in train_dataset.genes]
        for gene in genes_to_drop:
            genes.remove(gene)

        train_dataset = train_dataset.get_sub_dataset(genes)
        test_dataset = test_dataset.get_sub_dataset(genes)

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
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression, Lars
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors.nearest_centroid import NearestCentroid
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.base import clone

    from sklearn.feature_selection import RFE    

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2, mutual_info_classif
        
    from signature import Signature

    signatures_data = dict()
    proteins_ranks = []
    ranks = {}

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

    type1, type2, type3, type4, type5, type6, type7 = True, True, True, True, True, True, True
    # True, True, True, True, True, True, True
    #type1, type2, type3, type4, type5, type6, type7 = False, False, True, False, False, False, False
    #type1, type2, type3, type4, type5, type6, type7 = False, True, True, False, False, False, False


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
        {'name': 'Linear SVM', 'model': LinearSVC(), 'lambda_name':'model__C', 'lambda_grid': np.arange(3, 10)}, # has decision_function

        {'name': 'Decision Tree', 'model': DecisionTreeClassifier(), 'lambda_name':'model__max_depth', 'lambda_grid': np.arange(1, 20)}, #'max_depth': np.arange(3, 10) #has predict_proba

        {'name': 'Random Forest', 'model': RandomForestClassifier(n_estimators=n_estimators,n_jobs=nJobs), 'lambda_name':'model__max_features', 'lambda_grid': np.array([0.5, 0.75, 1.0])}, # predict_proba(X)

        {'name': 'Ada Boost Decision Trees', 'model': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=n_estimators), 'lambda_name':'model__learning_rate', 'lambda_grid': np.array([0.01, 0.1, 0.3, 0.6, 1.0])}, # predict_proba(X)

        {'name': 'Gradient Boosting', 'model': GradientBoostingClassifier(n_estimators=n_estimators, loss="deviance" ), 'lambda_name':'model__learning_rate', 'lambda_grid': np.array([0.01, 0.1, 0.3, 0.6, 1.0])}, #predict_proba(X)

        {'name': 'Lasso', 'model': LogisticRegression(penalty='l1', C=LassoBestC), 'lambda_name':'model__C', 'lambda_grid': np.logspace(-5, 0, 10)}, #predict_proba(X)

        {'name': 'Ridge', 'model': LogisticRegression(penalty='l2', C=RidgeBestC), 'lambda_name':'model__C', 'lambda_grid': np.logspace(-5, 0, 10)}, #predict_proba(X)

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


    report.write('\n\nNumber of features: %d\n\n' % len(train_dataset.genes))


    # Type 1: Model Based Ranking
    # Type 2: Attributes' Weights
    # Type 3: Univariate Feature Selection (Statistics)
    # Type 4: Recursive Feature Elimination
    # Type 5: Stability Selection
    # Type 6: Decrease of Accuracy


    def storeSignaturesMaxSize(scores, method, signatures_data, maxNumberOfProteins):
        for i in range(1,getMaxNumberOfProteins(scores, maxNumberOfProteins)+2):
            genes_indexes = [item[1] for item in scores[0:i]]
            sig = Signature(genes_indexes)
            if sig in signatures_data:
                signatures_data[sig]['methods'].add(method)
            else:
                sig_data = {'methods':set()}
                sig_data['methods'].add(method)
                signatures_data[sig] = sig_data
    
    def getBestSignatureByCV(scores, estimator, maxNumberOfProteins, k, rep, n_jobs):            
        signatures = []
        max_score = 0.0       
        cv_scores = []
        genes_indexes = [item[1] for item in scores] #ordered by rank
        for i in range(1, getMaxNumberOfProteins(scores, maxNumberOfProteins)+2):            
            sig_indexes = genes_indexes[0:i]
            # evaluate this signature for each estimator
            # cross-validation
            cv_score = None
            #? store cv score (F1?) and estimator name in Signature
            if max_score < cv_score:
                max_score = cv_score
            cv_scores.append(cv_scores)
        
        n = cv_scores.index(max_score)
        # n: 0 -> genes_indexes[0:1]        
        return {'genes_indexes': genes_indexes[0:n+1], 'cv_score': max_score}



    # ---------------------- Type 1 - Model Based Ranks -------------------------
    if type1:
        print('\nExecuting Type 1\n')
        # Random Forests
        clf = RandomForestClassifier(n_jobs=nJobs, n_estimators=n_estimators)
        scores = []
        for i in range(len(train_dataset.genes)):
            x_train = train_dataset.X()[:, i] # data matrix
            y_train = train_dataset.Y()  # classes/labels of each sample from             

            scores_cv = cross_val_score(clone(clf), x_train, y_train, cv=k, scoring=scoreEstimator)
            score = np.mean(scores_cv)
            scores.append((score, train_dataset.geneIndex(train_dataset.genes[i]), train_dataset.genes[i]))
        scores = sorted(scores, reverse = True, key=lambda tup: tup[0])
        saveRank(scores, results_path_rank+'rank_t1_uni_random_forest_mean_accuracy.csv')
        scores = normalizeScores(scores)
        ranks['t1_uni_random_forest'] = scores
        saveRank(scores, results_path_rank+'rank_t1_uni_random_forest_mean_accuracy_normalizedMinMax.csv')
        maxScore = max(scores,key=lambda item:item[0])[0]
        cutoffScore = maxScore-deltaScore
        genes_to_show = [item for item in scores if item[0] > cutoffScore]
        
        method = 'type1_random_forest'
        storeSignaturesMaxSize(scores, method, signatures_data, maxNumberOfProteins)
        
        report.write('-- Type 1 - Model Based Rank - Random Forests --\n')
        report.write('Number of proteins with score > %f (max-deltaRankCutoff): %d:\n%s\n' %(cutoffScore, len(genes_to_show), str(genes_to_show))) 
        report.write('\n\n')


    # ---------------------- Type 2 - Rank based on attribute Weights -------------------------
    if type2:
        print('\nExecuting Type 2\n')
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
            scores = sorted(scores, reverse = True, key=lambda tup: tup[0])
            saveRank(scores, results_path_rank+'rank_t2_weights_'+method+'.csv')
            scores = normalizeScores(scores)
            ranks['t2_weights_rank_'+method]= scores
            saveRank(scores, results_path_rank+'rank_t2_weights_'+method+'_normalizedMinMax.csv')

            maxScore = max(scores,key=lambda item:item[0])[0]
            cutoffScore = maxScore-deltaScore
            genes_to_show = [item for item in scores if item[0] > cutoffScore]

            method='type2_'+method
            storeSignaturesMaxSize(scores, method, signatures_data, maxNumberOfProteins)

            report.write('-- Type 2 - Attributes\' Weights - '+name+' --\n') 
            report.write('Number of proteins with score > %f (max-deltaRankCutoff): %d:\n%s\n' %(cutoffScore, len(genes_to_show), str(genes_to_show)))
            report.write('\n\n')        
        

        for estimator in estimators:
            print('Attribute Weights: %s' % estimator['name'])
            regularizationRank(estimator, train_dataset)

    # 
    # ---------------------- Type 3 - Univariate Ranks -------------------------
    if type3:
        print('\nExecuting Type 3\n')

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

        scores = sorted(scores, reverse = False, key=lambda tup: tup[0])
        saveRank(scores, results_path_rank+'rank_t3_uni_'+name+'_pvalue.csv')
        # ! Weight of attribute is 1-p_value
        scores_weight  = [(1-score[0],score[1],score[2]) for score in scores]
        scores = sorted(scores, reverse = True, key=lambda tup: tup[0])
        scores_weight = normalizeScores(scores_weight)
        ranks['t3_uni_'+name]=scores_weight
        saveRank(scores_weight, results_path_rank+'rank_t3_uni_'+name+'_1MinusPvalue_normalizedMinMax.csv')

        minScore = min(scores,key=lambda item:item[0])[0]
        cutoffScore = minScore+deltaScore
        genes_to_show = [item for item in scores if item[0] < cutoffScore]
        
        method = 'univariate_statistic'+'_'+name
        storeSignaturesMaxSize(scores_weight, method, signatures_data, maxNumberOfProteins)

        report.write('-- Type 3 - Univariate by Statistical-test P-value - '+name+' --\n')
        report.write('Minimum p-value: %f\n' %minScore)
        report.write('%d proteins have p-value < %f (min+deltaRankCutoff): %s\n' %(len(genes_to_show), cutoffScore, str(genes_to_show)))
        report.write('\n\n')
        #-----------------------------------------------------------------------------


        #---------- Chi-Squared ----------
        test = SelectKBest(score_func=chi2, k=2)
        x_train = train_dataset.get_normalized_data()

        y_train = factorize(train_dataset.labels)[0] 
        test.fit(x_train, y_train)
        
        scores = []
        for i in range(len(train_dataset.genes)):
            score = test.scores_[i]
            scores.append((score, train_dataset.geneIndex(train_dataset.genes[i]), train_dataset.genes[i]))
        scores = sorted(scores, reverse = True, key=lambda tup: tup[0])
        saveRank(scores, results_path_rank+'rank_t3_uni_chi_squared_score.csv')
        scores = normalizeScores(scores)
        ranks['t3_uni_chi_squared'] = scores
        saveRank(scores, results_path_rank+'rank_t3_uni_chi_squared_score_normalized01.csv')

        maxScore = max(scores,key=lambda item:item[0])[0]
        cutoffScore = maxScore-deltaScore
        genes_to_show = [item for item in scores if item[0] > cutoffScore]

        
        method = 'univariate_chi_squared'
        storeSignaturesMaxSize(scores, method, signatures_data, maxNumberOfProteins)

        report.write('-- Type 3 - Univariate by Statistical-test - Chi-Squared --\n')
        report.write('Number of proteins with score > %f (max-deltaRankCutoff): %d:\n%s\n' %(cutoffScore, len(genes_to_show), str(genes_to_show)))
        report.write('\n\n')




        #---------- Mutual Information ----------
        test = SelectKBest(score_func = mutual_info_classif, k=2)
        x_train = train_dataset.matrix
        y_train = factorize(train_dataset.labels)[0] 
        test.fit(x_train, y_train)
        
        scores = []
        for i in range(len(train_dataset.genes)):
            score = test.scores_[i]
            scores.append((score, train_dataset.geneIndex(train_dataset.genes[i]), train_dataset.genes[i]))
        
        scores = sorted(scores, reverse = True, key=lambda tup: tup[0])
        saveRank(scores, results_path_rank+'rank_t3_uni_mutual_inf_score.csv')
        scores = normalizeScores(scores)
        ranks['t3_uni_mutual_inf'] = scores
        saveRank(scores, results_path_rank+'rank_t3_uni_mutual_inf_score_normalized01.csv')

        maxScore = max(scores,key=lambda item:item[0])[0]
        cutoffScore = maxScore-deltaScore
        genes_to_show = [item for item in scores if item[0] > cutoffScore]

        
        method = 'univariate_mutual_information'
        storeSignaturesMaxSize(scores, method, signatures_data, maxNumberOfProteins)

        report.write('-- Type 3 - Univariate by Statistical-test - Mutual Information --\n')        
        report.write('Number of proteins with score > %f (max-deltaRankCutoff): %d:\n%s\n' %(cutoffScore, len(genes_to_show), str(genes_to_show)))
        report.write('\n\n')



        #---------- ANOVA F-value ----------
        test = SelectKBest(score_func = mutual_info_classif, k=2)
        x_train = train_dataset.matrix
        y_train = factorize(train_dataset.labels)[0] 
        test.fit(x_train, y_train)
        
        scores = []
        for i in range(len(train_dataset.genes)):
            score = test.scores_[i]
            scores.append((score, train_dataset.geneIndex(train_dataset.genes[i]), train_dataset.genes[i]))
        
        scores = sorted(scores, reverse = True, key=lambda tup: tup[0])
        saveRank(scores, results_path_rank+'rank_t3_uni_anova_fvalue_score.csv')
        scores = normalizeScores(scores)
        ranks['t3_uni_anova_fvalue'] = scores
        saveRank(scores, results_path_rank+'rank_t3_uni_anova_fvalue_score_normalized01.csv')

        maxScore = max(scores,key=lambda item:item[0])[0]
        cutoffScore = maxScore-deltaScore
        genes_to_show = [item for item in scores if item[0] > cutoffScore]

        
        method = 'univariate_anova_fvalue'
        storeSignaturesMaxSize(scores, method, signatures_data, maxNumberOfProteins)
        
        report.write('-- Type 3 - Univariate by Statistical-test - ANOVA F-value --\n')
        report.write('Number of proteins with score > %f (max-deltaRankCutoff): %d:\n%s\n' %(cutoffScore, len(genes_to_show), str(genes_to_show)))
        report.write('\n\n')


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
            scores = sorted(scores, reverse = False, key=lambda tup: tup[0])
            saveRank(scores, results_path_rank+'rank_t4_rfe_'+method+'.csv')

            scores_weight = [(len(scores)-score[0],score[1],score[2]) for score in scores]
            scores_weight = normalizeScores(scores_weight)
            
            ranks['t4_'+method] = scores_weight        
            saveRank(scores_weight, results_path_rank+'rank_t4_rfe_'+method+'_normalizedMinMax.csv') 
            maxScore = max(scores_weight,key=lambda item:item[0])[0]
            cutoffScore = maxScore-deltaScore
            genes_to_show = [item for item in scores if item[0] > cutoffScore]

            
            storeSignaturesMaxSize(scores_weight, method, signatures_data, maxNumberOfProteins)

            report.write('-- Type 3 - Recursive Feature Elimination - '+name+' --\n') 
            report.write('Number of proteins with normalized score > %f (max-deltaRankCutoff): %d:\n%s\n' %(cutoffScore, len(genes_to_show), str(genes_to_show)))
            report.write('\n\n')


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
            scores = sorted(scores, reverse = True, key=lambda tup: tup[0])
            saveRank(scores, results_path_rank+'rank_t5_stability_'+method+'.csv')            
            scores = normalizeScores(scores)
            ranks['t5_stability_'+method] = scores        
            saveRank(scores, results_path_rank+'rank_t5_stability_'+method+'_normalizedMinMax.csv') 
            maxScore = max(scores,key=lambda item:item[0])[0]
            cutoffScore = maxScore-deltaScore
            genes_to_show = [item for item in scores if item[0] > cutoffScore]
            
            storeSignaturesMaxSize(scores, method, signatures_data, maxNumberOfProteins)

            report.write('-- Type 5 -- Stability Selection - '+name+' --\n')
            report.write('Number of proteins with score > %f (max-deltaRankCutoff): %d:\n%s\n' %(cutoffScore, len(genes_to_show), str(genes_to_show)))
            report.write('\n\n')

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
            score_normal = np.mean(cross_val_score(clone(estimator['model']), X, y, cv = k, scoring=scoreEstimator))
            
            for i in range(len(traindata.genes)):
                X_shuffled = X.copy()
                scores_shuffle = []
                for j in range(3):
                    np.random.seed(j*3)
                    np.random.shuffle(X_shuffled[:,i])
                    score = np.mean(cross_val_score(clone(estimator['model']), X_shuffled, y, cv = k, scoring=scoreEstimator))
                    scores_shuffle.append(score)               
                gene_name = traindata.genes[i]
                scores.append((score_normal - np.mean(scores_shuffle), train_dataset.geneIndex(gene_name), gene_name))
                        
            scores = sorted(scores, reverse = True, key=lambda tup: tup[0])
            saveRank(scores, results_path_rank+'rank_t6_decrease_acc_'+method+'.csv')            
            scores = normalizeScores(scores)
            ranks['t6_decrease_acc_'+method] = scores        
            saveRank(scores, results_path_rank+'rank_t6_decrease_acc_'+method+'_normalizedMinMax.csv') 
            maxScore = max(scores,key=lambda item:item[0])[0]
            cutoffScore = maxScore-deltaScore
            genes_to_show = [item for item in scores if item[0] > cutoffScore]
            
            storeSignaturesMaxSize(scores, method, signatures_data, maxNumberOfProteins)

            report.write('-- Type 6 --- Mean Decrease Accuracy - '+name+' --\n')
            report.write('Number of proteins with score > %f (max-deltaRankCutoff): %d:\n%s\n' %(cutoffScore, len(genes_to_show), str(genes_to_show)))
            report.write('\n\n')        
        
        for estimator in estimators:            
            print('Decrease Accuracy: %s' % estimator['name'])
            meanDecreaseAccuracyScore(estimator, train_dataset)






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
    saveHeatMapScores(num_matrix, rows_labels=row_lables , cols_labels=cols_labels, filename=filename, metric='euclidean')


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


    freq_prot = {}
    for gene in train_dataset.genes:
        freq_prot[gene] = 0

    for name in sorted(ranks.keys()):
        rank = ranks[name][0:10]
        for score in rank:
            gene = score[2]
            if gene in freq_prot:
                freq_prot[gene]+=1

    matrix = []
    for gene in freq_prot.keys():
        matrix.append([gene,freq_prot[gene]])

    matrix = sorted(matrix, key=lambda tup: tup[1], reverse=True)
    filename = results_path+'top_10_prot_frequency.csv'
    np.savetxt(filename, matrix, delimiter=",", fmt='%s')

    # todo plot the freq considereing x-axis: topN, 1 < N < 100
    # sum the freq, from 1 to 100 and rank the proteins
    # plot the freq of top 10 proteins


    # todo histogram



    #!! consider adding this estimator 
    estimators.append({'name': 'Bagging Classifier Nearest Centroid', 'model': BaggingClassifier(base_estimator=NearestCentroid(), n_estimators=n_estimators), 'lambda_name':'model__max_features', 'lambda_grid': np.array([0.5, 0.75, 1.0])}) #predict_proba(X)

    # todo Compare Rank Scores using Histograms -> we can see who is more Spiked, more specific/general, Use rank Scores varying the number of samples used.

    # todo Compare the mean rank of features now Changing the number os samples used

    # todo pick the best estimator considering all proteins
    
    # todo write results on Report


report.close()


#  ================ END MAIN ===============
