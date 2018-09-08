
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
from pandas import DataFrame
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

    results_path_rank = results_path+'rank/'
    try:  
        os.mkdir(results_path_rank)
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
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression, Lars
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

    from sklearn.feature_selection import RFE    

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2, mutual_info_classif
    
    from utils import saveRank, normalizeScores, getMaxNumberOfProteins
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

    estimators = [
        {'name': 'Decision Tree', 'model': DecisionTreeClassifier()},
        {'name': 'Random Forest', 'model': RandomForestClassifier()},
        {'name': 'Ada Boost', 'model': AdaBoostClassifier()},
        {'name': 'Gradient Boosting', 'model': GradientBoostingClassifier()},
        {'name': 'Lasso', 'model': LogisticRegression(penalty='l1', C=LassoBestC)},
        {'name': 'Ridge', 'model': LogisticRegression(penalty='l2', C=RidgeBestC)},
        {'name': 'Linear SVM', 'model': LinearSVC()},
        {'name': 'Linear Discriminant Analysis', 'model': LinearDiscriminantAnalysis()}
    ]

    def getScore(traindata, pipe, estimator_name, i):
        if estimator_name in ['Lasso', 'Ridge', 'Linear SVM', 'Linear Discriminant Analysis']:
            if len(traindata.levels()) == 2:
                return pipe.coef_[i]
            else:
                return np.mean(pipe.coef_[:,i])
        else:
            return pipe.feature_importances_[i]



    # Type 1: Model Based Ranking
    # Type 2: Attributes' Weights
    # Type 3: Univariate Feature Selection (Statistics)
    # Type 4: Recursive Feature Elimination
    # Type 5: Stability Selection

    # ---------------------- Type 1 - Model Based Ranks -------------------------
    if type1:
        # Random Forests
        clf = RandomForestClassifier()
        scores = []
        for i in range(len(train_dataset.genes)):
            x_train = train_dataset.X()[:, i] # data matrix
            y_train = train_dataset.Y()  # classes/labels of each sample from 
            scores_cv = cross_val_score(clf, x_train, y_train, cv=k)        
            score = np.mean(scores_cv)
            scores.append((score, geneIndex[train_dataset.genes[i]], train_dataset.genes[i]))
        scores = sorted(scores, reverse = True)
        saveRank(scores, results_path_rank+'rank_t1_uni_random_forest_mean_accuracy.csv')
        scores = normalizeScores(scores)
        ranks['t1_uni_random_forest'] = scores
        saveRank(scores, results_path_rank+'rank_t1_uni_random_forest_mean_accuracy_normalizedMinMax.csv')
        maxScore = max(scores,key=lambda item:item[0])[0]
        cutoffScore = maxScore-deltaScore
        genes_to_show = [item for item in scores if item[0] > cutoffScore]
        genes_for_signature = []
        for i in range(1,getMaxNumberOfProteins(scores, maxNumberOfProteins)+2):
            genes_indexes = [item[1] for item in scores[0:i]]
            sig = Signature(genes_indexes)
            if sig in signatures_data:
                signatures_data[sig]['methods'].add('type1_random_forest')
            else:
                sig_data = {'methods':set()}
                sig_data['methods'].add('type1_random_forest')
                signatures_data[sig] = sig_data
        report.write('-- Type1 - Model Based Rank - Random Forests --\n')
        report.write('Number of proteins with score > %f (max-deltaRankCutoff): %d:\n%s\n' %(cutoffScore, len(genes_to_show), str(genes_to_show))) 
        report.write('\n\n')



    # ---------------------- Type 2 -------------------------
    if type2:
        
        def regularizationRank(estimator, traindata):
            model = estimator['model']
            name = estimator['name']
            method = name.lower().replace(" ","_")

            pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])           
            pipe.fit(traindata.X(), traindata.Y())
            scores = []
            count_coef_nonzero = 0
            for i in range(len(train_dataset.genes)):
                score = getScore(traindata, pipe.named_steps['clf'], name, i)
                scores.append((abs(score), geneIndex[train_dataset.genes[i]], train_dataset.genes[i]))            
            scores = sorted(scores, reverse = True)
            saveRank(scores, results_path_rank+'rank_t2_weights_'+method+'.csv')
            scores = normalizeScores(scores)
            ranks['t2_weights_rank_'+method]= scores
            saveRank(scores, results_path_rank+'rank_t2_weights_'+method+'_normalizedMinMax.csv')

            maxScore = max(scores,key=lambda item:item[0])[0]
            cutoffScore = maxScore-deltaScore
            genes_to_show = [item for item in scores if item[0] > cutoffScore]

            genes_for_signature = []
            for i in range(1,getMaxNumberOfProteins(scores, maxNumberOfProteins)+2):
                genes_indexes = [item[1] for item in scores[0:i]]
                sig = Signature(genes_indexes)
                if sig in signatures_data:
                    signatures_data[sig]['methods'].add('type2_'+method)
                else:
                    sig_data = {'methods':set()}
                    sig_data['methods'].add('type2_'+method)
                    signatures_data[sig] = sig_data
            report.write('-- Type2 - Attributes\' Weights - '+name+' --\n') 
            report.write('Number of proteins with score > %f (max-deltaRankCutoff): %d:\n%s\n' %(cutoffScore, len(genes_to_show), str(genes_to_show)))
            report.write('\n\n')        
        

        for estimator in estimators:
            regularizationRank(estimator, train_dataset)



    # 
    # ---------------------- Type 3 - Univariate Ranks -------------------------
    if type3:
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
        saveRank(scores, results_path_rank+'rank_t3_uni_'+name+'_pvalue.csv')
        # ! Weight of attribute is 1-p_value
        scores_weight  = [(1-score[0],score[1],score[2]) for score in scores]
        scores = sorted(scores, reverse = True)
        scores_weight = normalizeScores(scores_weight)
        ranks['t3_uni_'+name]=scores
        saveRank(scores_weight, results_path_rank+'rank_t3_uni_'+name+'_1MinusPvalue_normalizedMinMax.csv')

        minScore = min(scores,key=lambda item:item[0])[0]
        cutoffScore = minScore+deltaScore
        genes_to_show = [item for item in scores if item[0] < cutoffScore]

        genes_for_signature = []
        method = 'univariate_statistic'
        for i in range(1,getMaxNumberOfProteins(scores, maxNumberOfProteins)+2):
            genes_indexes = [item[1] for item in scores[0:i]]
            sig = Signature(genes_indexes)        
            if sig in signatures_data:
                signatures_data[sig]['methods'].add(method+'_'+name)
            else:
                sig_data = {'methods':set()}
                sig_data['methods'].add(method+'_'+name)
                signatures_data[sig] = sig_data

        report.write('-- Type 3 - Univariate by Statistical-test P-value - '+name+' --\n')
        report.write('Minimum p-value: %f\n' %minScore)
        report.write('%d proteins have p-value < %f (min+deltaRankCutoff): %s\n' %(len(genes_to_show), cutoffScore, str(genes_to_show)))
        report.write('\n\n')
        #-----------------------------------------------------------------------------


        #---------- Chi-Squared ----------
        test = SelectKBest(score_func=chi2, k=2)
        x_train = train_dataset.matrix
        y_train = factorize(train_dataset.labels)[0] 
        test.fit(x_train, y_train)
        
        scores = []
        for i in range(len(train_dataset.genes)):
            score = test.scores_[i]
            scores.append((score, geneIndex[train_dataset.genes[i]], train_dataset.genes[i]))
        scores = sorted(scores, reverse = True)
        saveRank(scores, results_path_rank+'rank_t3_uni_chi_squared_score.csv')
        scores = normalizeScores(scores)
        ranks['t3_uni_chi_squared'] = scores
        saveRank(scores, results_path_rank+'rank_t3_uni_chi_squared_score_normalized01.csv')

        maxScore = max(scores,key=lambda item:item[0])[0]
        cutoffScore = maxScore-deltaScore
        genes_to_show = [item for item in scores if item[0] > cutoffScore]

        genes_for_signature = []
        method = 'univariate_chi_squared'
        for i in range(1,getMaxNumberOfProteins(scores, maxNumberOfProteins)+2):
            genes_indexes = [item[1] for item in scores[0:i]]
            sig = Signature(genes_indexes)
            if sig in signatures_data:
                signatures_data[sig]['methods'].add(method)
            else:
                sig_data = {'methods':set()}
                sig_data['methods'].add(method)
                signatures_data[sig] = sig_data

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
            scores.append((score, geneIndex[train_dataset.genes[i]], train_dataset.genes[i]))
        
        scores = sorted(scores, reverse = True)
        saveRank(scores, results_path_rank+'rank_t3_uni_mutual_inf_score.csv')
        scores = normalizeScores(scores)
        ranks['t3_uni_mutual_inf'] = scores
        saveRank(scores, results_path_rank+'rank_t3_uni_mutual_inf_score_normalized01.csv')

        maxScore = max(scores,key=lambda item:item[0])[0]
        cutoffScore = maxScore-deltaScore
        genes_to_show = [item for item in scores if item[0] > cutoffScore]

        genes_for_signature = []
        method = 'univariate_mutual_information'
        for i in range(1,getMaxNumberOfProteins(scores, maxNumberOfProteins)+2):
            genes_indexes = [item[1] for item in scores[0:i]]
            sig = Signature(genes_indexes)
            if sig in signatures_data:
                signatures_data[sig]['methods'].add(method)
            else:
                sig_data = {'methods':set()}
                sig_data['methods'].add(method)
                signatures_data[sig] = sig_data

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
            scores.append((score, geneIndex[train_dataset.genes[i]], train_dataset.genes[i]))
        
        scores = sorted(scores, reverse = True)
        saveRank(scores, results_path_rank+'rank_t3_uni_anova_fvalue_score.csv')
        scores = normalizeScores(scores)
        ranks['t3_uni_anova_fvalue'] = scores
        saveRank(scores, results_path_rank+'rank_t3_uni_anova_fvalue_score_normalized01.csv')

        maxScore = max(scores,key=lambda item:item[0])[0]
        cutoffScore = maxScore-deltaScore
        genes_to_show = [item for item in scores if item[0] > cutoffScore]

        genes_for_signature = []
        method = 'univariate_anova_fvalue'
        for i in range(1,getMaxNumberOfProteins(scores, maxNumberOfProteins)+2):
            genes_indexes = [item[1] for item in scores[0:i]]
            sig = Signature(genes_indexes)
            if sig in signatures_data:
                signatures_data[sig]['methods'].add(method)
            else:
                sig_data = {'methods':set()}
                sig_data['methods'].add(method)
                signatures_data[sig] = sig_data
        
        report.write('-- Type 3 - Univariate by Statistical-test - ANOVA F-value --\n')
        report.write('Number of proteins with score > %f (max-deltaRankCutoff): %d:\n%s\n' %(cutoffScore, len(genes_to_show), str(genes_to_show)))
        report.write('\n\n')




    #---------- Recursive Feature Elimination (RFE) ----------
    if type4:

        def rfeRank(estimator, traindata):        
            model = estimator['model']
            name = estimator['name']
            method = name.lower().replace(" ","_")
                    
            rfe = RFE(model, n_features_to_select=1)
            rfe.fit(StandardScaler().fit_transform(traindata.X()),traindata.Y())
            scores = []        
            for i in range(len(traindata.genes)):
                score = float(rfe.ranking_[i])
                scores.append((abs(score), geneIndex[traindata.genes[i]], traindata.genes[i]))          
            scores = sorted(scores, reverse = False)
            saveRank(scores, results_path_rank+'rank_t4_rfe_'+method+'.csv')

            scores_weight = [(len(scores)-score[0],score[1],score[2]) for score in scores]
            scores_weight = normalizeScores(scores_weight)
            
            ranks['t4_'+method] = scores_weight        
            saveRank(scores_weight, results_path_rank+'rank_t4_rfe_'+method+'_normalizedMinMax.csv') 
            maxScore = max(scores_weight,key=lambda item:item[0])[0]
            cutoffScore = maxScore-deltaScore
            genes_to_show = [item for item in scores if item[0] > cutoffScore]

            genes_for_signature = []
            for i in range(1,getMaxNumberOfProteins(scores, maxNumberOfProteins)+2):
                genes_indexes = [item[1] for item in scores[0:i]]
                sig = Signature(genes_indexes)
                if sig in signatures_data:
                    signatures_data[sig]['methods'].add('type4_'+method)
                else:
                    sig_data = {'methods':set()}
                    sig_data['methods'].add('type4_'+method)
                    signatures_data[sig] = sig_data
            report.write('-- Type 3 - Recursive Feature Elimination - '+name+' --\n') 
            report.write('Number of proteins with normalized score > %f (max-deltaRankCutoff): %d:\n%s\n' %(cutoffScore, len(genes_to_show), str(genes_to_show)))
            report.write('\n\n')


        for estimator in estimators:
            print('Rfe for '+estimator['name']+'\n')
            rfeRank(estimator, train_dataset)


        


    type5 = False
    #---------- type 6 - Stability Selection ----------
    if type5:
        from stability_selection import StabilitySelection

        def stabilitySelectionScores(estimator, traindata):          
            base_estimator = Pipeline([('scaler', StandardScaler()), ('model', estimator['model'])])
            selector = StabilitySelection(base_estimator=base_estimator).fit(x, y)
            scores = [] 
            name = estimator['name']  
            method = name.lower().replace(" ","_")     
            for i in range(len(traindata.genes)):
                score = float(selector.stability_scores_[i])
                scores.append((abs(score), geneIndex[traindata.genes[i]], traindata.genes[i]))
            scores = sorted(scores, reverse = True)
            saveRank(scores, results_path_rank+'rank_t5_'+method+'.csv')
            scores_weight = [(len(scores)-score[0],score[1],score[2]) for score in scores]
            scores_weight = normalizeScores(scores_weight)
            ranks['t5_'+method] = scores_weight        
            saveRank(scores_weight, results_path_rank+'rank_t5_'+method+'_normalized01.csv') 
            maxScore = max(scores_weight,key=lambda item:item[0])[0]
            cutoffScore = maxScore-deltaScore
            genes_to_show = [item for item in scores if item[0] > cutoffScore]
            genes_for_signature = []
            for i in range(1,getMaxNumberOfProteins(scores, maxNumberOfProteins)+2):
                genes_indexes = [item[1] for item in scores[0:i]]
                sig = Signature(genes_indexes)
                if sig in signatures_data:
                    signatures_data[sig]['methods'].add(method)
                else:
                    sig_data = {'methods':set()}
                    sig_data['methods'].add(method)
                    signatures_data[sig] = sig_data
            report.write('-- Stability Selection - '+name+' --\n')
            report.write('Number of proteins with score > %f (max-deltaRankCutoff): %d:\n%s\n' %(cutoffScore, len(genes_to_show), str(genes_to_show)))
            report.write('\n\n')

        for estimator in estimators:
            stabilitySelectionScores(estimator, train_dataset)


    #! scores lists must be already sorted

    # Protein names are ordered by rank
    matrix = []  
    for name in sorted(ranks.keys()):
        rank = ranks[name]              
        row = [name] + [score[2] for score in rank]
        matrix.append(row)
    matrix = np.matrix(matrix).transpose()    
    filename = results_path+'all_ranks_prot_names.csv'
    np.savetxt(filename, matrix, delimiter=",", fmt='%s')


    # Protein names are listed, values of the matrix are their Scores
    matrix = [['method']+train_dataset.genes]
    for name in sorted(ranks.keys()):
        rank = ranks[name]    
        # TODO VERIFY THE ORDER OF VALUES AND GENES... SEEMS TO BE UNPAIRED            
        order_by_index = sorted(rank, key=lambda tup: tup[1])
        values = [score[0] for score in order_by_index]
        row = [name] + values
        matrix.append(row)        
    matrix = np.matrix(matrix).transpose()   
    mean_column = ['mean'] + np.array(matrix[1:,1:]).astype(float).mean(axis=1).tolist()
    std_column = ['std'] +   np.array(matrix[1:,1:]).astype(float).std(axis=1).tolist()
    new_matrix = np.matrix([mean_column,std_column]).transpose()
    matrix = np.concatenate((matrix, new_matrix), axis=1)    
    sorted_matrix = matrix[1:,:]
    sorted_matrix.sort(axis=-2)
    matrix = np.concatenate((matrix[0,:],sorted_matrix[::-1]), axis=0) #! Score: the greater the value, the more important is the protein

    filename = results_path+'all_ranks_scores.csv'
    np.savetxt(filename, matrix, delimiter=",", fmt='%s')


    # Protein names are listed, values of the matrix are their Positions/Ranks
    matrix = [['method']+train_dataset.genes]
    for name in sorted(ranks.keys()):
        rank = ranks[name]    
        new_scores = [(rank[i][0], rank[i][1], rank[i][2], i) for i in range(len(rank))]
        values = [score[3] for score in sorted(new_scores, key=lambda tup: tup[1])]
        row = [name] + values
        matrix.append(row)
    matrix = np.matrix(matrix).transpose()   
    mean_column = ['mean'] + np.array(matrix[1:,1:]).astype(int).mean(axis=1).tolist()
    std_column = ['std'] +   np.array(matrix[1:,1:]).astype(int).std(axis=1).tolist()
    new_matrix = np.matrix([mean_column,std_column]).transpose()
    matrix = np.concatenate((matrix, new_matrix), axis=1)    
    sorted_matrix = matrix[1:,:]
    sorted_matrix.sort(axis=-2)
    matrix = np.concatenate((matrix[0,:],sorted_matrix), axis=0)  #! Position: the lower the value, the more important is the protein

    filename = results_path+'all_ranks_positions.csv'
    np.savetxt(filename, matrix, delimiter=",", fmt='%s')    




    # rank by mean-rank position (from above methods)

    # create all combinations considering args['top-n'] proteins
    max_n = args['topN']

    # if total number of proteins is <= args['n_small']
        #compute all possible combinations

    # if total number of proteins is > args['n_small']
        # small signatures from all proteins
    max_sig_size = args['nSSearch']



    print('Number of signatures to test: %d' % len(signatures_data.keys()))
    
    # for signature in signatures_data:        
    #     print('%d - %d - %s: %s' % (len(signatures_data[signature]['methods']), len(signature.genes), signature.toGeneNamesList(geneNames), str(signatures_data[signature]['methods']) ))



    # todo Print a table where columns are methods and values are the normalized weights.
    # todo Print a table where columns are the methods and values are the position in the rank.
    # todo Find a way to visualize and compare all this changes
    # todo Create a table where the mean/std Score and Position are shown.
    # ? heat map could show what are the ranks that desagree the most and what are the most similar etc.


    # todo Compare Rank Scores using Histograms -> we can see who is more Spiked, more specific/general, Use rank Scores varying the number of samples used.

    # todo Compare the mean rank of features now Changing the number os samples used



#  Signatures evaluation

    # todo Print given the best features, find others features highly correlated to these ones. Organize the features and print a heatmap, putting a * on those correlated that were not selected.


    # todo - parameter to enter a list of TruePositives, so that the report will contain information on their positions, will print all methods, the number of proteins each method found, etc.
    
    # ! The cutoff of the rank must be based ON scores > 0 and on Cross-validation


# ! variable importance Heatmap
# from sklearn.feature_selection import mutual_info_classif
# kepler_mutual_information = mutual_info_classif(kepler_X, kepler_y)

# plt.subplots(1, figsize=(26, 1))
# sns.heatmap(kepler_mutual_information[:, np.newaxis].T, cmap='Blues', cbar=False, linewidths=1, annot=True)
# plt.yticks([], [])
# plt.gca().set_xticklabels(kepler.columns[1:], rotation=45, ha='right', fontsize=12)
# plt.suptitle("Kepler Variable Importance (mutual_info_classif)", fontsize=18, y=1.2)
# plt.gcf().subplots_adjust(wspace=0.2)



report.close()


#  ================ END MAIN ===============

