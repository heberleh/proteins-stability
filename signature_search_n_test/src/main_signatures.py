
# import the necessary packages
import argparse
import csv
import os

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
from pandas import factorize

from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import (cohen_kappa_score, f1_score, make_scorer,
                             matthews_corrcoef)                              
from sklearn.feature_selection import (RFE, SelectKBest, chi2,
                                       mutual_info_classif)
from sklearn.linear_model import Lars, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from scipy.special import comb

from signature import Signature, Signatures
from dataset import Dataset
from utils import saveBoxplots
from random import sample

import itertools

from multiprocessing import Pool, Lock, cpu_count
import gc

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=Warning)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument('--inputFolder', help='The path for the folder where the scores files and others are.', action='store', required=True)

#ap.add_argument('--nSSearch', help='Set the maximum number of proteins to search for small signatures formed by all prot. combinations (signature size).', action='store', type=int, default=3)

#ap.add_argument('--nSmall', help='Set the number of proteins considered small. If the total number of proteins in a dataset is smaller or equal than NSMALL, it will compute all combinations of proteins to form signatures. Otherwise, it will consider NSSEARCH to compute only combinations of size up to the value set for this parameter.', action='store', type=int, default=10)

#ap.add_argument('--topN', help='Create all combinations of top-N signatures from the average of ranks.', action='store', type=int, default=10)


#ap.add_argument('--deltaRankCutoff', help='The percentage of difference from the maximum score value that is used as cutoff univariate ranks. The scores are normalized between 0 and 1. So, if the maximum value is 0.9, and deltaRankCutoff is set to 0.05, the cutoff value is 0.85. Proteins with score >= 0.85 are selected to form signatures by top-N proteins.', action='store', type=float, default=0.10)

#ap.add_argument('--limitSignatureSize', help='Limit the size of signatures created by rank to the number of samples. For instance, when selecting top-N proteins using 30 samples, the maximum value of N is 30.', action='store_true')

args = vars(ap.parse_args())


# ==============================================================================================


def leaveOneOutScore(X, y, estimator, scorer):
    loo = LeaveOneOut()
    y_pred = []
    y_true = []
    for train_index, test_index in loo.split(X):
        classifier = clone(estimator)
        classifier.fit(X[train_index], y[train_index])
        y_pred.append(classifier.predict(X[test_index])[0])
        y_true.append(y[test_index][0])
    return scorer(y_pred, y_true)




def cv_signature(parallel_arguments):
    sss, dataset_train, genes, scoring, estimators, method, signature = parallel_arguments[0], parallel_arguments[1], parallel_arguments[2], parallel_arguments[3], parallel_arguments[4], parallel_arguments[5], parallel_arguments[6]

    train_data = dataset_train.get_sub_dataset(genes)                      

    max_score = -np.inf
    for estimator in estimators:
        estimator_name = estimator['name']
        if not signature.hasScore(estimator_name=estimator_name):
           
            base_estimator = Pipeline([('scaler', StandardScaler()), ('model', clone(estimator['model']))])

            cv_scores = cross_val_score(base_estimator, train_data.X(), train_data.Y(), scoring=scoring, cv=sss, n_jobs=1)

            mean_cv_score = np.round(np.mean(cv_scores), decimals=2)
            
            signature.addScore(method=method, estimator_name=estimator_name, score=mean_cv_score)
            
            if mean_cv_score > max_score:
                max_score = mean_cv_score

    return {'max': max_score}

def evaluateRankParallel(dataset_train, dataset_test, scores, rank_id, estimators, max_n, fold_signatures, scorer):
    scores = sorted(scores, reverse=True)
    max_score = -np.inf

    sss = StratifiedShuffleSplit(n_splits=8, test_size=3, random_state=0)

    n_cpu = cpu_count()
    n_splits = n_cpu*20
    pool = Pool(processes=n_cpu)
    
    parallel_input = []
    for i in range(1, max_n+1):        
        genes = [item[1] for item in scores[0:i]]        
        # sss, dataset, genes, scoring, estimators, method, fold_signatures
        parallel_input.append((sss, train_data, genes, scorer[1], estimators, rank_id, fold_signatures.get(genes)))
    
    result_part = pool.map(cv_signature, parallel_input)
    max_score = -np.inf
    for i in range(len(result_part)):            
        score = result_part[i]['max']
        if score > max_score:
            max_score = score

    gc.collect()

    max_group_size = 4
    for sub_list_size in range(1, max_group_size+1):
        allgenes = [item[1] for item in scores[0:max_n]]

        all_possible_signatures = itertools.combinations(allgenes, sub_list_size)
        n_possible_groups = comb(len(allgenes), sub_list_size, exact=True)  
        all_possible_signatures = [genes for genes in all_possible_signatures]

        max_n_groups =  200*sub_list_size
        if n_possible_groups > max_n_groups:
            all_possible_signatures = sample(all_possible_signatures, max_n_groups)
            n_possible_groups = max_n_groups
    
        print("Signature size " + str(sub_list_size) + ". There are "+ str(n_possible_groups) + " combinations to be tested.")
        count = 0  
        iter_all_sig = iter(all_possible_signatures)

        hasnext = True
        while(hasnext):
            parallel_input = []        
            for rep in range(n_splits):
                try:
                    genes = next(iter_all_sig)                              
                    #dataset, outer_folds, classifier_name        
                    parallel_input.append((sss, train_data, genes, scorer[1], estimators, 'random_small_combinations', fold_signatures.get(genes)))
                except:
                    hasnext = False
                    break
            result_part = pool.map(cv_signature, parallel_input)
            for i in range(len(result_part)):            
                score = result_part[i]['max']
                if score > max_score:
                    max_score = score
            gc.collect()
    gc.collect()
    return max_score

def evaluateRank(dataset_train, dataset_test, scores, rank_id, estimators, max_n, fold_signatures, scorer):
    scores = sorted(scores, reverse=True)
    max_score = -np.inf

    sss = StratifiedShuffleSplit(n_splits=18, test_size=3, random_state=0)
    
    for i in range(1, max_n+1):        
        genes = [item[1] for item in scores[0:i]]  
        train_data = dataset_train.get_sub_dataset(genes)
        test_data = dataset_test.get_sub_dataset(genes)        
        

        for estimator in estimators:
            mean_cv_score = None
            estimator_name = estimator['name']
            if not signature.hasScore(estimator_name=estimator_name):                
                # CV score
                base_estimator = Pipeline([('scaler', StandardScaler()), ('model', clone(estimator['model']))])
                cv_scores = cross_val_score(base_estimator, train_data.X(), train_data.Y(), scoring=scorer[1], cv=sss, n_jobs=-1)                
                mean_cv_score = np.round(np.mean(cv_scores), decimals=2)

                signature.addScore(method=rank_id, estimator_name=estimator_name, score=mean_cv_score)

                # # Independent test score          
                # scaler = StandardScaler()
                # model = clone(estimator['model'])
                
                # base_estimator.fit(scaler.fit_transform(train_data.X()), train_data.Y())
                # y_pred = base_estimator.predict(scaler.transform(test_data.X()))

                # independent_score = scorer[0](test_data.Y(), y_pred)
                # independent_score = np.round(independent_score, decimals=2)

                # signature.addIndependentScore(estimator_name=estimator_name, score=independent_score, y_pred=y_pred, y_true=test_data.Y())

            else:
                mean_cv_score = signature.getScore(estimator_name=estimator_name)
            max_score = np.max([max_score, mean_cv_score])       


    max_group_size = 6
    for sub_list_size in range(1, max_group_size+1):
        allgenes = [item[1] for item in scores[0:max_n]]

        all_possible_signatures = itertools.combinations(allgenes, sub_list_size)
        n_possible_groups = comb(len(allgenes), sub_list_size, exact=True)  
        all_possible_signatures = [genes for genes in all_possible_signatures]

        max_n_groups =  200*sub_list_size
        if n_possible_groups > max_n_groups:
            all_possible_signatures = sample(all_possible_signatures, max_n_groups)
            n_possible_groups = max_n_groups
    
        print("Signature size " + str(sub_list_size) + ". There are "+ str(n_possible_groups) + " combinations to be tested.")
        count = 0  
        for genes in all_possible_signatures:

            train_data = dataset_train.get_sub_dataset(genes)
            test_data = dataset_test.get_sub_dataset(genes)        
            signature = fold_signatures.get(genes)

            for estimator in estimators:
                mean_cv_score = None
                estimator_name = estimator['name']
                if not signature.hasScore(estimator_name=estimator_name):                
                    # CV score
                    base_estimator = Pipeline([('scaler', StandardScaler()), ('model', clone(estimator['model']))])
                    cv_scores = cross_val_score(base_estimator, train_data.X(), train_data.Y(), scoring=scorer[1], cv=sss, n_jobs=-1)                
                    mean_cv_score = np.round(np.mean(cv_scores), decimals=2)

                    signature.addScore(method='small_combinations', estimator_name=estimator_name, score=mean_cv_score)

                    # # Independent test score          
                    # scaler = StandardScaler()
                    # model = clone(estimator['model'])
                    
                    # base_estimator.fit(scaler.fit_transform(train_data.X()), train_data.Y())
                    # y_pred = base_estimator.predict(scaler.transform(test_data.X()))

                    # independent_score = scorer[0](test_data.Y(), y_pred)
                    # independent_score = np.round(independent_score, decimals=2)

                    # signature.addIndependentScore(estimator_name=estimator_name, score=independent_score, y_pred=y_pred, y_true=test_data.Y())

                else:
                    mean_cv_score = signature.getScore(estimator_name=estimator_name)
                max_score = np.max([max_score, mean_cv_score])

                count += 1
                if count % 1000 == 0:
                    print("There are " + str(n_possible_groups-count) + " to be tested.")     

    return max_score


input_path = args['inputFolder']

n_estimators = 50
nJobs = 8
LassoBestC = 0.1 #! todo run CV on train dataset to get these
RidgeBestC = 0.1



folders = [ item for item in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, item))]
print(folders)



for folder_name in folders:
    folder_path = os.path.join(input_path, folder_name)

    subfolders = [ item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))]

    if 'filter' in subfolders:
        folder_path = os.path.join(folder_path, 'filter')
    elif 'nonFilter' in subfolders:
        folder_path = os.path.join(folder_path, 'noFilter')
    else:
        print("The filter/noFilter folder could not be found.")
        exit()


    train_path = os.path.join(folder_path, 'dataset_train_from_scrip.csv')
    test_path = os.path.join(folder_path, 'dataset_test_from_scrip.csv')

    train_data = Dataset(train_path, scale=False, normalize=False, sep=',')
    test_data = Dataset(test_path, scale=False, normalize=False, sep=',')
    
    score_matrix_path = os.path.join(folder_path, 'all_ranks_scores.csv')
    # read score matrix
    scores_df = pd.read_csv(score_matrix_path, index_col=0)
    ranks = {}

    fold_signatures = Signatures()
    k_bench = min(train_data.getMinNumberOfSamplesPerClass(), 10)

    # Benchmark of best parameter C for L1 and L2
    param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100]}
    pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'l1'))     
    grid = GridSearchCV(pipe, param_grid, cv = k_bench)
    grid.fit(train_data.X(), train_data.Y())
    LassoBestC = grid.best_params_['logisticregression__C']


    pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'l2'))     
    grid = GridSearchCV(pipe, param_grid, cv = k_bench)
    grid.fit(train_data.X(), train_data.Y())
    RidgeBestC = grid.best_params_['logisticregression__C']  

    estimators = [  #lambda_name='model__C',lambda_grid=np.logspace(-5, -1, 50)

    {'name': 'Radial SVM', 'model': SVC(kernel='rbf'), 'lambda_name':'model__C', 'lambda_grid': np.arange(3, 10)},

    {'name': 'Linear SVM', 'model': LinearSVC(), 'lambda_name':'model__C', 'lambda_grid': np.arange(3, 10)}, # has decision_function

    {'name': 'Decision Tree', 'model': DecisionTreeClassifier(), 'lambda_name':'model__max_depth', 'lambda_grid': np.arange(1, 20)}, #'max_depth': np.arange(3, 10) #has predict_proba

    #{'name': 'Random Forest', 'model': RandomForestClassifier(n_estimators=n_estimators,n_jobs=nJobs), 'lambda_name':'model__max_features', 'lambda_grid': np.array([0.5, 0.75, 1.0])}, # predict_proba(X)

    #{'name': 'Ada Boost Decision Trees', 'model': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=n_estimators), 'lambda_name':'model__learning_rate', 'lambda_grid': np.array([0.01, 0.1, 0.3, 0.6, 1.0])}, # predict_proba(X)

    #{'name': 'Gradient Boosting', 'model': GradientBoostingClassifier(n_estimators=n_estimators, loss="deviance" ), 'lambda_name':'model__learning_rate', 'lambda_grid': np.array([0.01, 0.1, 0.3, 0.6, 1.0])}, #predict_proba(X)

    {'name': 'Lasso', 'model': LogisticRegression(penalty='l1', C=LassoBestC), 'lambda_name':'model__C', 'lambda_grid': np.logspace(-5, 0, 10)}, #predict_proba(X)

    {'name': 'Ridge', 'model': LogisticRegression(penalty='l2', C=RidgeBestC), 'lambda_name':'model__C', 'lambda_grid': np.logspace(-5, 0, 10)}, #predict_proba(X)

    {'name': 'Linear Discriminant Analysis', 'model': LinearDiscriminantAnalysis(), 'lambda_name':'model__n_components', 'lambda_grid': None} #! need to be set after filtering
    # predict_proba(X)
    ]



    #A Bagging classifier.

    #A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

    #This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting [R154]. If samples are drawn with replacement, then the method is known as Bagging [R155]. When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces [R156]. Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches [R157].
    n_estimators_bagging = 65
    estimators_bagging = [
        {'name': 'Linear Discriminant Analysis',
        'model': BaggingClassifier(base_estimator=LinearDiscriminantAnalysis(), n_estimators=n_estimators_bagging, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=0, verbose=0)
        },
        {'name': 'Ridge',
        'model': BaggingClassifier(base_estimator=LogisticRegression(penalty='l2', C=RidgeBestC), n_estimators=n_estimators_bagging, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=0, verbose=0)
        },
        {'name': 'Lasso',
        'model': BaggingClassifier(base_estimator=LogisticRegression(penalty='l1', C=LassoBestC), n_estimators=n_estimators_bagging, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=0, verbose=0)
        },
        {'name': 'Decision Tree',
        'model': BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=n_estimators_bagging, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=0, verbose=0)
        },
        {'name': 'Linear SVM',
        'model': BaggingClassifier(base_estimator=LinearSVC(), n_estimators=n_estimators_bagging, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=0, verbose=0)
        },
        {'name': 'Radial SVM',
        'model': BaggingClassifier(base_estimator=SVC(kernel='rbf'), n_estimators=n_estimators_bagging, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=0, verbose=0)
        },
        {'name': 'Nearest Centroid',
        'model': BaggingClassifier(base_estimator=NearestCentroid(), n_estimators=n_estimators_bagging, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=0, verbose=0)
        },
        {'name': 'Gaussian Naive Bayes',
        'model': BaggingClassifier(base_estimator=GaussianNB(), n_estimators=n_estimators_bagging, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=0, verbose=0)
        },
    ]

    estimators_basic = {
        'Linear Discriminant Analysis': {'name': 'Linear Discriminant Analysis',
        'model': LinearDiscriminantAnalysis()
        },
         'Ridge': {'name': 'Ridge',
        'model': LogisticRegression(penalty='l2', C=RidgeBestC)
        },
        'Lasso': {'name': 'Lasso',
        'model': LogisticRegression(penalty='l1', C=LassoBestC)
        },
        'Decision Tree': {'name': 'Decision Tree',
        'model': DecisionTreeClassifier()
        },
        'Linear SVM': {'name': 'Linear SVM',
        'model': LinearSVC()
        },
        'Radial SVM': {'name': 'Radial SVM',
        'model': SVC(kernel='rbf')
        },
        'Nearest Centroid': {'name': 'Nearest Centroid',
        'model': NearestCentroid()
        },
        'Gaussian Naive Bayes': {'name': 'Gaussian Naive Bayes',
        'model': GaussianNB()
        }
    }    

    scoreEstimator = None
    matthews_scorer = make_scorer(matthews_corrcoef)
    kappa_scorer = make_scorer(cohen_kappa_score)

    uni, counts = np.unique(train_data.Y(), return_counts=True)

    if len(train_data.levels()) == 2 and counts[0] == counts[1]:
        scoreEstimator = (roc_auc_score, 'roc_auc')
        scoreEstimatorInfo = """
        """
    elif len(train_data.levels()) == 3 or len(dataset.levels()) == 2:
        scoreEstimator = (matthews_corrcoef, matthews_scorer)
        scoreEstimatorInfo = """Kappa"""
    else:
        print('\nDataset with more than 3 classes is not supported.\n\n')
        exit()


    estimators_bagging_scores = {}
    max_mean = -np.inf
    for estimator in estimators_bagging:
        sss = StratifiedShuffleSplit(n_splits=20, test_size=len(train_data.levels()), random_state=0)        
        base_estimator = Pipeline([('scaler', StandardScaler()), ('model', clone(estimator['model']))])
        cv_scores = cross_val_score(base_estimator, train_data.X(), train_data.Y(), scoring=scoreEstimator[1], cv=sss, n_jobs=-1)
        estimators_bagging_scores[estimator['name']] = cv_scores
        mean_scores = np.mean(cv_scores)
        if mean_scores > max_mean:
            selected_bagging_classifier = estimator
            max_mean = mean_scores

        print(estimator['name'] + ': ' + str(mean_scores))
    
    selected_classifier = estimators_basic[selected_bagging_classifier['name']]

    bagging_names = [estimator['name'] for estimator in estimators_bagging]
    box_plot_values = []
    for estimator_name in bagging_names:
        box_plot_values.append(estimators_bagging_scores[estimator_name])

    filename = os.path.join(folder_path, 'boxplot_bagging_classifiers_scores.png')
    saveBoxplots(box_plot_values, filename=filename, x_labels=bagging_names)
    print("Saved: "+filename)

    for method in scores_df.columns:        
        scores = []
        for gene in scores_df.index:
            scores.append((scores_df[method][gene],gene))
        ranks[method] = sorted(scores, reverse=True)

    for method in ranks:
        # Extracting signatures from Ranks
        # signatures are stored in the fold_signatures set
        max_score = evaluateRankParallel(train_data, test_data, scores=ranks[method], rank_id=method, estimators=[selected_classifier], max_n=np.min([len(ranks[method]),len(train_data.Y())]), fold_signatures=fold_signatures, scorer=scoreEstimator)

        
        print('Max score for '+method+': '+str(max_score))
        print('Number of signatures: '+str(len(fold_signatures.signatures)))
        filename = os.path.join(folder_path, 'all_signatures_tested_basic.png')
        fold_signatures.save(filename)


    

    best_signatures = fold_signatures.getSignaturesMaxScore()
    
    #Evaluate the best_signatures again, with bagging estimators where feature-boostraping is not used, but sample-boostraping is.
    signature_bagging_scores = {}
    for sig in best_signatures:
        print('Score: %f, Indep. Score: %f, Size: %d, Method: %s %s, Signature:\n%s' % (sig[0],sig[1], sig[2], sig[4], str(sig[3]), str(sig[5])))
        signature = sig[5]

        classifier = clone(selected_bagging_classifier['model'])
        params_bagging = {"n_estimators": 256, 'max_samples':1.0, 'max_features':1.0, 'bootstrap':True, 'bootstrap_features':False}
        classifier.set_params(**params_bagging)


        sss = StratifiedShuffleSplit(n_splits=1000, test_size=len(train_data.levels()), random_state=0)

        new_train = train_data.get_sub_dataset(signature.genes)
        base_estimator = Pipeline([('scaler', StandardScaler()), ('model', classifier)])
        cv_scores = cross_val_score(base_estimator, new_train.X(), new_train.Y(), scoring=scoreEstimator[1], cv=sss, n_jobs=-1)

        mean_score = np.mean(cv_scores)
        print("Signature bagging score: %f\n\n" % mean_score)

            # for estimator_name in signature.independent_scores:
            #     print(signature.independent_scores[estimator_name]['y_pred'])
            #     print(signature.independent_scores[estimator_name]['y_true'])

    # chose the best estimator ?

    # for each rank_method
        # find the best signature pairs    #! round for 2 decimals
        
        # create signature object and score

    correlated_proteins_path = os.path.join(folder_path, 'correlated_genes.csv')
    # read correlation matrix

    








    # def storeSignaturesMaxSize(scores, method, signatures_data, maxNumberOfProteins):
    #     for i in range(1,getMaxNumberOfProteins(scores, maxNumberOfProteins)+2):
    #         genes_indexes = [item[1] for item in scores[0:i]]
    #         sig = Signature(genes_indexes)
    #         if sig in signatures_data:
    #             signatures_data[sig]['methods'].add(method)
    #         else:
    #             sig_data = {'methods':set()}
    #             sig_data['methods'].add(method)
    #             signatures_data[sig] = sig_data
    
    # def getBestSignatureByCV(scores, estimator, maxNumberOfProteins, k, rep, n_jobs):            
    #     signatures = []
    #     max_score = 0.0       
    #     cv_scores = []
    #     genes_indexes = [item[1] for item in scores] #ordered by rank
    #     for i in range(1, getMaxNumberOfProteins(scores, maxNumberOfProteins)+2):            
    #         sig_indexes = genes_indexes[0:i]
    #         # evaluate this signature for each estimator
    #         # cross-validation
    #         cv_score = None
    #         #? store cv score (F1?) and estimator name in Signature
    #         if max_score < cv_score:
    #             max_score = cv_score
    #         cv_scores.append(cv_scores)
        
    #     n = cv_scores.index(max_score)
    #     # n: 0 -> genes_indexes[0:1]        
    #     return {'genes_indexes': genes_indexes[0:n+1], 'cv_score': max_score}


    # deltaScore = float(args['deltaRankCutoff'])

    # limitSigSize = args['limitSignatureSize']
    # maxNumberOfProteins = len(train_dataset.genes)
    # if limitSigSize:
    #     maxNumberOfProteins = len(train_dataset.samples)


    # # FOR METHOD... IN RANKS...
    #     storeSignaturesMaxSize(scores, method, signatures_data, maxNumberOfProteins)





    # todo plot the freq considereing x-axis: topN, 1 < N < 100
    # sum the freq, from 1 to 100 and rank the proteins
    # plot the freq of top 10 proteins



    #!! consider adding this estimator 
    # estimators.append({'name': 'Bagging Classifier Nearest Centroid', 'model': BaggingClassifier(base_estimator=NearestCentroid(), n_estimators=n_estimators), 'lambda_name':'model__max_features', 'lambda_grid': np.array([0.5, 0.75, 1.0])}) #predict_proba(X)

    # todo Compare Rank Scores using Histograms -> we can see who is more Spiked, more specific/general, Use rank Scores varying the number of samples used.

    # todo Compare the mean rank of features now Changing the number os samples used

    # todo pick the best estimator considering all proteins
    
    # todo write results on Report





    # todo for each rank, seek for signatures computing CV and the selected estimator

    # todo compute the union of proteins from this signatures... and print a table with more information... freq, scores, p-value, etc

    # todo create signatures using the union of proteins -> consider Max-size of signature...

    # todo create signatures using all proteins considering the "small size of signatures"...
    
    # ! todo track the origin of each signature: signature from rank, signature from union of potential proteins, small signature from other proteins (in combination with the good ones/union)

    # todo test all these signatures using the estimator and repeated cv

    # todo order the signatures

    # todo considering the top 100 or 1000 signatures, compute CV for ALL estimators that have AUC    

    # todo 


    # create all combinations considering args['top-n'] proteins
    # max_n = args['topN']

    # if total number of proteins is <= args['n_small']
        #compute all possible combinations

    # if total number of proteins is > args['n_small']
        # small signatures from all proteins
    # max_sig_size = args['nSSearch']



    # print('Number of signatures to test: %d' % len(signatures_data.keys()))
    
    # for signature in signatures_data:        
    #     print('%d - %d - %s: %s' % (len(signatures_data[signature]['methods']), len(signature.genes), signature.toGeneNamesList(geneNames), str(signatures_data[signature]['methods']) ))







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
