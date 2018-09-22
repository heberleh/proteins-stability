
# import the necessary packages
import argparse
import csv
import os

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
from pandas import DataFrame
from pandas import factorize

from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import (cohen_kappa_score, f1_score, make_scorer,
                             matthews_corrcoef, log_loss)                              
from sklearn.feature_selection import (RFE, SelectKBest, chi2,
                                       mutual_info_classif)
from sklearn.linear_model import Lars, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
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

from datetime import datetime

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

ap.add_argument('--input-folder', help='The path for the folder where the scores files and others are.', action='store', required=True)

ap.add_argument('--n-splits-best-classifier', help='The number of ShuffleSplits for the Cross-validation to select the best classifier.', action='store', type=int, default=64)

ap.add_argument('--n-estimators-best-classifier', help='The number of estimators to run the BaggingClassifer and select the best regular classifier.', action='store', type=int, default=32)

ap.add_argument('--n-splits-searching-signatures', help='The number of ShuffleSplits for the Cross-validation evaluate signatures.', action='store', type=int, default=64)

ap.add_argument('--n-splits-testing-candidates-signatures', help='The number of ShuffleSplits for the Cross-validation to select the final good and best signatures.', action='store', type=int, default=256)

ap.add_argument('--n-estimators-testing-candidates-signatures', help='The number of estimators to run the BaggingClassifer and select the final good and best signatures.', action='store', type=int, default=128)

ap.add_argument('--max-size-all-combinations', help='The maximum size to perform all the combinations of features to for signatures. For size greater than max-size-all-combinations, the algorithm can select random signatures if the parameter max-size-random-combinations is set.', action='store', type=int, default=4)

ap.add_argument('--max-size-random-combinations', help='The maximum size to perform random combinations of the topest features from ranks to form signatures: max-size-all-combinations < |random signatures| <= max-size-random-combinations. If max-size-all-combinations is equal to max-size-random-combinations,random signatures are not created.', action='store', type=int, default=5)

ap.add_argument('--n-splits-testing-final-signatures', help='The number of ShuffleSplits for the Cross-validation used to test the final best signatures with many classifiers.', action='store', type=int, default=256)

ap.add_argument('--n-estimators-final-signatures', help='The number of estimators to run the selected BaggingClassifer, Ada Boost, Gradient Boosting and Random Forest classifiers to evaluate the final signatures.', action='store', type=int, default=128)

ap.add_argument('--debug-fast', help='Set parameters to minimum for debugging.', action='store_true')


args = vars(ap.parse_args())



# ==============================================================================================


def fitClassifiers(estimators, train_data, test_data, scoring, signature, n_splits, test_size):

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
    
    train = train_data.get_sub_dataset(signature.genes)
    test = test_data.get_sub_dataset(signature.genes) 

    estimator_cv_scores = {}
    estimator_independent_scores = {}
    for estimator_name in estimators:
        estimator = estimators[estimator_name]
        base = Pipeline([('scaler', StandardScaler()), ('model', clone(estimator['model']))])
        X = train.X()
        y = train.Y()
        cv_scores = cross_validate(base, X, y, scoring=scoring, cv=sss, n_jobs=-1)
                
        estimator_cv_scores[estimator['name']] = cv_scores

        X_test = test.X()
        y_test = test.Y()
        base = Pipeline([('scaler', StandardScaler()), ('model', clone(estimator['model']))])
        base.fit(X, y)
        # y_pred = base.predict(X_test)
        
        independent_scores = {}
        for scoring_name in scoring:
            independent_scores[scoring_name] = scoring[scoring_name](base, X_test, y_test)
        estimator_independent_scores[estimator['name']] = independent_scores
    
    return {'cv': estimator_cv_scores, 'independent': estimator_independent_scores}



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
    sss, dataset_train, genes, scoring, estimator = parallel_arguments[0], parallel_arguments[1], parallel_arguments[2], parallel_arguments[3], parallel_arguments[4]

    train_data = dataset_train.get_sub_dataset(genes)                      
        
    base_estimator = Pipeline([('scaler', StandardScaler()), ('model', clone(estimator['model']))])

    cv_scores = cross_val_score(base_estimator, train_data.X(), train_data.Y(), scoring=scoring, cv=sss, n_jobs=1)

    mean_cv_score = np.round(np.mean(cv_scores), decimals=2)      

    del parallel_arguments, sss, dataset_train, scoring, train_data, base_estimator, cv_scores
    gc.collect()
    return {'genes': genes, 'score': mean_cv_score, 'estimator_name': estimator['name']}


def close_pool(pool):
    pool.close()
    pool.join()
    pool.terminate()
    gc.collect()

def registerResult(result, method, fold_signatures):
    signature = fold_signatures.get(result['genes'])    
    signature.setScore(estimator_name=result['estimator_name'], score=result['score'])
    signature.addMethod(method)


def evaluateRankParallel(dataset_train, dataset_test, scores, rank_id, estimators, max_n, fold_signatures, scorer, max_all_comb_size, max_random_comb_size, test_size, n_splits):
    scores = sorted(scores, reverse=True)
    max_score = -np.inf

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)

    n_cpu = cpu_count()
    n_splits = n_cpu*20
    
    parallel_input = []
    for i in range(1, max_n+1):        
        genes = [item[1] for item in scores[0:i]]
        for estimator in estimators:
            if not fold_signatures.get(genes).hasScore(estimator['name']):
                parallel_input.append((sss, train_data, genes, scorer, estimator))
    
    pool = Pool(processes=n_cpu)
    result_part = pool.map(cv_signature, parallel_input)
    #? {'genes': genes, 'score': score, 'estimator_name': estimator_name}

    max_score = -np.inf
    for i in range(len(result_part)):
        score = result_part[i]['score']
        if score > max_score:
            max_score = score
        registerResult(result_part[i], rank_id, fold_signatures)

    close_pool(pool)
    
    del result_part, parallel_input
    gc.collect()

    for sub_list_size in range(1, max_random_comb_size+1):
        allgenes = [item[1] for item in scores[0:max_n]]

        all_possible_signatures = itertools.combinations(allgenes, sub_list_size)
        n_possible_groups = comb(len(allgenes), sub_list_size, exact=True)  
        all_possible_signatures = [genes for genes in all_possible_signatures]
        
        if sub_list_size > max_all_comb_size:
            n_possible_groups = 200*sub_list_size
            all_possible_signatures = sample(all_possible_signatures, n_possible_groups)           
    
        print("Signature size " + str(sub_list_size) + ". There are "+ str(n_possible_groups) + " combinations to be tested.")
        count = 0  
        iter_all_sig = iter(all_possible_signatures)

        pool = Pool(processes=n_cpu)
        hasnext = True
        while(hasnext):
            parallel_input = []        
            for rep in range(n_splits):
                try:
                    genes = next(iter_all_sig)                              
                    #dataset, outer_folds, classifier_name 
                    
                    for estimator in estimators:
                        if not fold_signatures.get(genes).hasScore(estimator['name']):
                            parallel_input.append((sss, train_data, genes, scorer, estimator))   
                except:
                    hasnext = False
                    break
            result_part = pool.map(cv_signature, parallel_input)
            for i in range(len(result_part)):
                if result_part[i]['score'] > max_score:
                    max_score = result_part[i]['score']
                registerResult(result_part[i], 'smallcomb_'+rank_id, fold_signatures)    

            del result_part, parallel_input
            gc.collect()    

        close_pool(pool)
    return max_score


def correlatedSignatures(main_signature, init, correlated_genes):
    correlated_signatures = set()
    for i in range(init, len(main_signature.genes)):              
        if main_signature.genes[i] in correlated_genes:            
            for gene in correlated_genes[main_signature.genes[i]]:               
                copy_genes = [item for item in main_signature.genes]                
                copy_genes[i] = gene                              
                for signature in correlatedSignatures(Signature(copy_genes), i, correlated_genes):
                    correlated_signatures.add(signature)

    correlated_signatures.add(main_signature) 
    return correlated_signatures

print(args)
input_path = args['input_folder']

nJobs = 8


n_splits_select_classifier = args['n_splits_best_classifier']
n_estimators_bagging_select_classifier = args['n_estimators_best_classifier']

n_splits_searching_signature = args['n_splits_searching_signatures']

n_splits_bagging_final = args['n_splits_testing_candidates_signatures']
n_estimators_bagging_final = args['n_estimators_testing_candidates_signatures']

max_all_comb_size  = args['max_size_all_combinations']
max_random_comb_size  = args['max_size_random_combinations']

n_split_final_signature = args['n_splits_testing_final_signatures']
n_estimators_final_signatures = args['n_estimators_final_signatures']


limit_n_ranks = -1
if args['debug_fast']:
    n_splits_select_classifier = 3
    n_estimators_bagging_select_classifier = 10

    n_splits_searching_signature = 3

    n_splits_bagging_final = 3
    n_estimators_bagging_final = 3

    max_all_comb_size  = 2
    max_random_comb_size  = 3
    
    n_split_final_signature = 5
    n_estimators_final_signatures = 5

    limit_n_ranks = 3


good_genes_freq_global = {}
best_genes_freq_global = {}

good_signatures_global = {}
best_signatures_global = {}

sig_freq_in_all_ranks_global = {}
best_sig_freq_global = {}

best_sig_scores_global = {}


from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score, fbeta_score, matthews_corrcoef

def getScoring(train_data):
    scoring = {}
    scoring['accuracy'] = make_scorer(accuracy_score)
    scoring['kappa'] = make_scorer(cohen_kappa_score)
    scoring['f1_weighted'] = make_scorer(f1_score, average='weighted')
    scoring['precision_weighted'] = make_scorer(precision_score, average='weighted')    
    scoring['fbeta_weighted'] = make_scorer(fbeta_score, average='weighted', beta=0.5)
    if len(train_data.levels()) == 2:
        scoring['average_precision_weighted'] = make_scorer(average_precision_score, average='weighted')
        scoring['matthews_corrcoef'] = make_scorer(matthews_corrcoef)
        classes_labels = train_data.levels()
        for i in range(len(train_data.levels())):
            label = classes_labels[i]
            scoring['f1_'+label] = make_scorer(f1_score, pos_label=i)
            scoring['precision_'+label] = make_scorer(precision_score, pos_label=i)
            scoring['recall_'+label] = make_scorer(recall_score, pos_label=i)
            scoring['fbeta_'+label] = make_scorer(fbeta_score, pos_label=i, beta=0.5)
    return scoring



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


    train_path = os.path.join(folder_path, 'train_including_correlated_genes.csv')
    test_path = os.path.join(folder_path, 'test_including_correlated_genes.csv')

    train_data = Dataset(train_path, scale=False, normalize=False, sep=',')
    test_data = Dataset(test_path, scale=False, normalize=False, sep=',')
    

    test_size = None
    n_levels = len(train_data.levels())
    uni, c = np.unique(train_data.Y(), return_counts=True)  

    if np.min(c)/float(np.max(c)) < 0.05:
        # if balanced and small
        if len(train_data.Y()) < n_levels * 14:
            test_size = len(train_data.levels())
        else:
            test_size = 0.2
            print(np.min(c)/np.max(c))
    else:
        # if unbalanced and small
        if train_data.getMinNumberOfSamplesPerClass() < 13:
            if (0.1*len(train_data.Y())) < n_levels:
                test_size = len(train_data.levels())
            else:
                test_size = 0.1
        else:
            test_size = 0.2     

    print('Size of test set for cv: %f' % test_size)

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
    estimators_bagging = [
        {'name': 'Linear Discriminant Analysis',
        'model': BaggingClassifier(base_estimator=LinearDiscriminantAnalysis(), n_estimators=n_estimators_bagging_select_classifier, max_samples=1.0, max_features=1.0, bootstrap=False, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=0, verbose=0)
        },
        {'name': 'Ridge',
        'model': BaggingClassifier(base_estimator=LogisticRegression(penalty='l2', C=RidgeBestC), n_estimators=n_estimators_bagging_select_classifier, max_samples=1.0, max_features=1.0, bootstrap=False, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=0, verbose=0)
        },
        {'name': 'Lasso',
        'model': BaggingClassifier(base_estimator=LogisticRegression(penalty='l1', C=LassoBestC), n_estimators=n_estimators_bagging_select_classifier, max_samples=1.0, max_features=1.0, bootstrap=False, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=0, verbose=0)
        },
        {'name': 'Decision Tree',
        'model': BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=n_estimators_bagging_select_classifier, max_samples=1.0, max_features=1.0, bootstrap=False, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=0, verbose=0)
        },
        {'name': 'Linear SVM',
        'model': BaggingClassifier(base_estimator=LinearSVC(), n_estimators=n_estimators_bagging_select_classifier, max_samples=1.0, max_features=1.0, bootstrap=False, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=0, verbose=0)
        },
        {'name': 'Radial SVM',
        'model': BaggingClassifier(base_estimator=SVC(kernel='rbf'), n_estimators=n_estimators_bagging_select_classifier, max_samples=1.0, max_features=1.0, bootstrap=False, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=0, verbose=0)
        },
        {'name': 'Nearest Centroid',
        'model': BaggingClassifier(base_estimator=NearestCentroid(), n_estimators=n_estimators_bagging_select_classifier, max_samples=1.0, max_features=1.0, bootstrap=False, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=0, verbose=0)
        },
        {'name': 'Gaussian Naive Bayes',
        'model': BaggingClassifier(base_estimator=GaussianNB(), n_estimators=n_estimators_bagging_select_classifier, max_samples=1.0, max_features=1.0, bootstrap=False, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=0, verbose=0)
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


    final_estimators = {
        'Random Forest': {'name': 'Random Forest', 'model': RandomForestClassifier(n_estimators=n_estimators_final_signatures,n_jobs=1)},
        'Ada Boost Decision Trees': {'name': 'Ada Boost Decision Trees', 'model': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=n_estimators_final_signatures)},
        'Gradient Boosting': {'name': 'Gradient Boosting', 'model': GradientBoostingClassifier(n_estimators=n_estimators_final_signatures, loss="deviance" )
        }
    }
    for estimator_name in estimators_basic:
        final_estimators[estimator_name] = {
            'name': estimator_name, 
            'model': clone(estimators_basic[estimator_name]['model'])
        }
    



    def meanTrueProbability(clf, X, y_true):
        class_labels = clf.classes_.tolist()
        y_pred_proba = clf.predict_proba(X)        
        max_probabilities = [y_pred_proba[i][class_labels.index(y_true[i])] for i in range(len(y_true))]
        return np.mean(max_probabilities)




    scoreEstimator = None
    matthews_scorer = make_scorer(matthews_corrcoef)
    kappa_scorer = make_scorer(cohen_kappa_score)
    log_loss_scorer = make_scorer(log_loss, labels=factorize(train_data.levels())[0])
    roc_auc_scorer = make_scorer(roc_auc_score, average='weighted')
    
    uni, counts = np.unique(train_data.Y(), return_counts=True)   

    scoreEstimator = kappa_scorer
    scoreEstimatorBagging = kappa_scorer #! used only after selecting the best features by Kappa

    estimators_bagging_scores = {}
    max_mean = -np.inf
    selected_bagging_classifier = None
    for estimator in estimators_bagging:
        
        sss = StratifiedShuffleSplit(n_splits=n_splits_select_classifier, test_size=test_size, random_state=0)  

        base_estimator = Pipeline([('scaler', StandardScaler()), ('model', clone(estimator['model']))])
        
        cv_scores = cross_val_score(base_estimator, train_data.X(), train_data.Y(), scoring=scoreEstimator, cv=sss, n_jobs=-1)

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

    starting_time = datetime.now() 
    count = 1
        
    keys = ranks.keys()
    if limit_n_ranks != -1:
        keys = keys[0:limit_n_ranks]   

    max_n_features=np.min([len(ranks[method]),len(train_data.Y())]) 

    for method in keys:
        # Extracting signatures from Ranks
        # signatures are stored in the fold_signatures set
        max_score = evaluateRankParallel(train_data, test_data, scores=ranks[method], rank_id=method, estimators=[selected_classifier], max_n=max_n_features, fold_signatures=fold_signatures, scorer=scoreEstimator, max_all_comb_size=max_all_comb_size, max_random_comb_size=max_random_comb_size, test_size=test_size, n_splits=n_splits_searching_signature)
        
        print('Max score for '+method+': '+str(max_score))     
        filename = os.path.join(folder_path, 'all_signatures_tested_basic.csv')
        fold_signatures.save(filename)

        print("Rank (%d/%d)" % (count, len(keys)))
        count+=1

        time_message = 'Time: %s\n----------------------------\n' % str(datetime.now()-starting_time)
        print(time_message)
       
    best_signatures = fold_signatures.getSignaturesMaxScore(delta=0.1)


    good_sig_data = []
    for data in best_signatures:
        signature = data[5]
        rank_names = []
        for name in ranks:
            rank_genes = [item[1] for item in ranks[name][0:max_n_features]] #! already sorted
            if set(signature.genes).issubset(rank_genes):
                rank_names.append(name)        

        sig_data = {'signature': signature, 'selected_classifier': selected_classifier['name'], 'frequency': len(rank_names)/float(len(ranks)), 'methods': str(rank_names)}

        good_sig_data.append(sig_data)
    good_signatures_global[folder_name] = good_sig_data


    gene_freq = {}
    for data in best_signatures:       
        signature = data[5]
        for gene in signature.genes:
            if gene in gene_freq:
                gene_freq[gene] += 1
            else:
                gene_freq[gene] =  1
    for gene in gene_freq:
        gene_freq[gene] /= float(len(best_signatures))    
    good_genes_freq_global[folder_name] = gene_freq

    matrix = []
    for gene in gene_freq:
        matrix.append([gene, gene_freq[gene]])    
    matrix = sorted(matrix, key=lambda tup: tup[1], reverse=True)
    filename = os.path.join(folder_path, 'prot_freq_in_good_signatures.csv')
    df = DataFrame(matrix)
    df.to_csv(filename, header=True)
    #np.savetxt(filename, matrix, delimiter=",", fmt='%s')

    def meanFrequency(signature, gene_freq):
        sum = 0.0
        for gene in signature.genes:
            if gene not in gene_freq:
                return 10.0 #! signatures with correlated genes are not counted in gene_freq, thus, let the user specify the 'freq' of this signature by means of the original ones that can be converted to this one
            sum += gene_freq[gene]
        return sum/float(len(signature.genes))

    correlated_proteins_path = os.path.join(folder_path, 'correlated_genes.csv')
    # read correlation matrix
    correlation_df = pd.read_csv(correlated_proteins_path, index_col=0, header=None)
    print(correlation_df)
    correlated_genes = {}
    for gene in correlation_df.index:
        correlated_genes[gene] = set()
        for gene2 in correlation_df.loc[gene]:
            if pd.isna(gene2) and not gene2 == '':
                correlated_genes[gene].add(gene2)

    #Evaluate the best_signatures again, with bagging estimators where feature-boostraping is not used, but sample-boostraping is.
    signature_bagging_scores = []
    for data in best_signatures:
        #print('Score: %f, Indep. Score: %f, Size: %d, Method: %s %s, Signature:\n%s' % (sig[0],sig[1], sig[2], sig[4], str(sig[3]), str(sig[5])))
        main_signature = data[5]        

        signatures_set = correlatedSignatures(main_signature, 0, correlated_genes)
        
        for signature in signatures_set:
            classifier = clone(selected_bagging_classifier['model'])
            params_bagging = {"n_estimators": n_estimators_bagging_final, 'max_samples':1.0, 'max_features':1.0, 'bootstrap':True, 'bootstrap_features':False}
            classifier.set_params(**params_bagging)

            sss = StratifiedShuffleSplit(n_splits=n_splits_bagging_final, test_size=test_size, random_state=0)

            new_train = train_data.get_sub_dataset(signature.genes)
            base_estimator = Pipeline([('scaler', StandardScaler()), ('model', classifier)])
            cv_scores = cross_val_score(base_estimator, new_train.X(), new_train.Y(), scoring=  scoreEstimatorBagging, cv=sss, n_jobs=-1)        
            mean_score = np.mean(cv_scores)
            mean_freq = meanFrequency(signature, gene_freq)
            signature_bagging_scores.append((mean_score*mean_freq, mean_score, mean_freq, signature))

    

    # set(t2).issubset(t1)
    # set(t1).issuperset(t2)
    header = ['1st_decision (cv)', '2nd_decision (mean_freq*bag_cv)', 'bagging_cv', 'mean_freq (prot)', 'size', 'frequency', "signature", "ranks"]
    matrix = []
    for signature in signature_bagging_scores:
        rank_names = []
        for name in ranks:
            rank_genes = [item[1] for item in ranks[name][0:max_n_features]] #! already sorted
            if set(signature[3].genes).issubset(rank_genes):
                rank_names.append(name)
        row = [signature[3].getScore(selected_classifier['name']), signature[0], signature[1], signature[2], signature[3].size(), len(rank_names)/float(len(ranks)), str(signature[3].genes), str(rank_names)]
        matrix.append(row)
    filename = os.path.join(folder_path, 'good_signatures.csv')
    df = DataFrame(data=matrix, columns=header)
    df.sort_values([df.columns[1],df.columns[0],df.columns[2]], ascending=[0,0,1])
    df.to_csv(filename, header=True)

    # Reduce the number of signatures again
    signature_bagging_scores = sorted(signature_bagging_scores, reverse=True)
    max_score = signature_bagging_scores[0][0]    
    best_signatures = [signature for signature in signature_bagging_scores if signature[0] > max_score-0.02]

    gene_freq = {}
    for data in best_signatures:       
        signature = data[3]
        for gene in signature.genes:
            if gene in gene_freq:
                gene_freq[gene] += 1
            else:                
                gene_freq[gene] =  1
    for gene in gene_freq:
        gene_freq[gene] /= float(len(best_signatures))    
    best_genes_freq_global[folder_name] = gene_freq

    matrix = []
    for gene in gene_freq:
        matrix.append([gene, gene_freq[gene]])    
    matrix = sorted(matrix, key=lambda tup: tup[1], reverse=True)
    filename = os.path.join(folder_path, 'prot_freq_in_best_signatures.csv')
    df = DataFrame(matrix)
    df.to_csv(filename, header=True)

    signature_bagging_max_prob = []
    for data in best_signatures:
        score = data[0]
        mean_cv = data[1]
        mean_freq = data[2]
        signature = data[3]
        
        classifier = clone(selected_bagging_classifier['model'])
        params_bagging = {"n_estimators": n_estimators_bagging_final, 'max_samples':1.0, 'max_features':1.0, 'bootstrap':True, 'bootstrap_features':False}
        classifier.set_params(**params_bagging)

        sss = StratifiedShuffleSplit(n_splits=n_splits_bagging_final, test_size=test_size, random_state=0)

        new_train = train_data.get_sub_dataset(signature.genes)
        base_estimator = Pipeline([('scaler', StandardScaler()), ('model', classifier)])
        cv_probs = cross_val_score(base_estimator, new_train.X(), new_train.Y(), scoring=meanTrueProbability, cv=sss, n_jobs=-1)        
        mean_mean_max_prob = np.round(np.mean(cv_probs), decimals=2)
        signature_bagging_max_prob.append((mean_mean_max_prob, signature, mean_cv, mean_freq, score))

    signature_bagging_max_prob = sorted(signature_bagging_max_prob, reverse=True)

    header = ['1st_decision (cv)', '2nd_decision (mean_freq*bag_cv)', '3rd decision (prob)',  'mean_cv', 'mean_freq (prot)', 'size', 'frequency in ranks', "signature", "ranks"]
    matrix = []
    for signature in signature_bagging_max_prob:
        rank_names = []
        for name in ranks:
            rank_genes = [item[1] for item in ranks[name][0:max_n_features]]
            if set(signature[1].genes).issubset(rank_genes): #! already sorted
                rank_names.append(name)
        row = [signature[1].getScore(selected_classifier['name']), signature[4], signature[0], signature[2], signature[3], signature[1].size(), len(rank_names)/float(len(ranks)), str(signature[1].genes), str(rank_names)]
        matrix.append(row)
    filename = os.path.join(folder_path, 'even_better_signatures.csv')
    df = DataFrame(data=matrix, columns=header)
    df.sort_values([df.columns[2],df.columns[1],df.columns[3]], ascending=[0,0,1])
    df.to_csv(filename, header=True)

    max_prob = signature_bagging_max_prob[0][0]
    bestSignature = [item for item in signature_bagging_max_prob if item[0] == max_prob]


    print("Best signature of this fold is: ")
    header = ['1st_decision (cv)', '2nd_decision (mean_freq*bag_cv)', '3rd decision (prob)',  'mean_cv', 'mean_freq (prot)', 'size', 'frequency in ranks', "signature", "ranks"]
    matrix = []
    best_sig_data = []
    best_sig_scores = []
    for data in bestSignature:
        #mean_mean_max_prob, signature, mean_cv, mean_freq, score
        signature = data[1]
        prob = data[0]
        cv_score = data[2]
        mean_freq = data[3]
        score = data[4]
        internal_score = signature.getScore(selected_classifier['name'])
        rank_names = []
        for name in ranks:
            rank_genes = [item[1] for item in ranks[name][0:max_n_features]] #! already sorted
            if set(signature.genes).issubset(rank_genes):    
                rank_names.append(name)
        print("     "+str(signature.genes))
        row = [signature.getScore(selected_classifier['name']), score, prob, cv_score, mean_freq, signature.size(), len(rank_names)/float(len(ranks)), str(signature.genes), str(rank_names)]

        sig_data = {'signature': signature, 'proba': prob, 'score*mean_prot_freq': score, 'mean_freq': mean_freq, 'bagging_score': cv_score, 'selected_classifier': selected_classifier['name'], 'frequency': len(rank_names)/float(len(ranks)), 'methods': str(rank_names)}

        best_sig_data.append(sig_data)
        matrix.append(row)

        filename = os.path.join(folder_path, 'final_signatures.csv')
        df = DataFrame(matrix, columns=header)        
        df.to_csv(filename, header=True)

        name = selected_bagging_classifier['name']
       
        bootstrap_features_only = clone(selected_bagging_classifier['model'])
        params_bagging = {"n_estimators": n_estimators_final_signatures, 'max_samples':1.0, 'max_features':1.0, 'bootstrap':False, 'bootstrap_features':True}
        classifier.set_params(**params_bagging)
        final_estimators[name + "_bs_features"] = selected_bagging_classifier
        
        bootstrap_samples_only = clone(selected_bagging_classifier['model'])
        params_bagging = {"n_estimators": n_estimators_final_signatures, 'max_samples':1.0, 'max_features':1.0, 'bootstrap':True, 'bootstrap_features':False}
        bootstrap_samples_only.set_params(**params_bagging)
        final_estimators[name + "_bs_samples"] = selected_bagging_classifier

        bootstrap_features_samples = clone(selected_bagging_classifier['model'])
        params_bagging = {"n_estimators": n_estimators_final_signatures, 'max_samples':1.0, 'max_features':1.0, 'bootstrap':True, 'bootstrap_features':True}
        bootstrap_features_samples.set_params(**params_bagging)
        final_estimators[name + "_bs_features_samples"] = selected_bagging_classifier

        result = fitClassifiers(final_estimators, train_data, test_data, getScoring(train_data), signature, n_splits=n_split_final_signature, test_size=test_size)

        best_sig_scores.append(result)
        print(result)

    best_signatures_global[folder_name] = best_sig_data
    best_sig_scores_global[folder_name] = best_sig_scores




    
mean_genes_freq = {}
for fold in good_genes_freq_global:
    for gene in good_genes_freq_global[fold]:  
        value = None
        if gene in good_genes_freq_global[fold]:
            value = good_genes_freq_global[fold][gene]
        else:
            value = 0.0
        if gene in mean_genes_freq:
            mean_genes_freq[gene] += value
        else:
            mean_genes_freq[gene] = value
for gene in mean_genes_freq:
    mean_genes_freq[gene] /= float(len(good_genes_freq_global))

matrix = []
for gene in mean_genes_freq:
    matrix.append([gene, mean_genes_freq[gene]])    
matrix = sorted(matrix, key=lambda tup: tup[1], reverse=True)
filename = os.path.join(folder_path, 'prot_mean_freq_in_good_signatures.csv')
df = DataFrame(matrix)
df.to_csv(filename, header=True)


mean_genes_freq = {}
for fold in best_genes_freq_global:
    for gene in best_genes_freq_global[fold]:  
        value = None
        if gene in best_genes_freq_global[fold]:
            value = best_genes_freq_global[fold][gene]
        else:
            value = 0.0
        if gene in mean_genes_freq:
            mean_genes_freq[gene] += value
        else:
            mean_genes_freq[gene] = value
for gene in mean_genes_freq:
    mean_genes_freq[gene] /= float(len(best_genes_freq_global))

matrix = []
for gene in mean_genes_freq:
    matrix.append([gene, mean_genes_freq[gene]])    
matrix = sorted(matrix, key=lambda tup: tup[1], reverse=True)
filename = os.path.join(folder_path, 'prot_mean_freq_in_best_signatures.csv')
df = DataFrame(matrix)
df.to_csv(filename, header=True)






# ====================================== COMPARE =================================


good_genes_freq_global = {}
best_genes_freq_global = {}

good_signatures_global = {}
best_signatures_global = {}

sig_freq_in_all_ranks_global = {}
best_sig_freq_global = {}

best_sig_scores_global = {}






























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
