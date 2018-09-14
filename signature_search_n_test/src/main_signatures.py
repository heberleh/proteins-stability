
#ap.add_argument('--nSSearch', help='Set the maximum number of proteins to search for small signatures formed by all prot. combinations (signature size).', action='store', type=int, default=3)

#ap.add_argument('--nSmall', help='Set the number of proteins considered small. If the total number of proteins in a dataset is smaller or equal than NSMALL, it will compute all combinations of proteins to form signatures. Otherwise, it will consider NSSEARCH to compute only combinations of size up to the value set for this parameter.', action='store', type=int, default=10)

ap.add_argument('--topN', help='Create all combinations of top-N signatures from the average of ranks.', action='store', type=int, default=10)


ap.add_argument('--deltaRankCutoff', help='The percentage of difference from the maximum score value that is used as cutoff univariate ranks. The scores are normalized between 0 and 1. So, if the maximum value is 0.9, and deltaRankCutoff is set to 0.05, the cutoff value is 0.85. Proteins with score >= 0.85 are selected to form signatures by top-N proteins.', action='store', type=float, default=0.10)

ap.add_argument('--limitSignatureSize', help='Limit the size of signatures created by rank to the number of samples. For instance, when selecting top-N proteins using 30 samples, the maximum value of N is 30.', action='store_true')


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


    deltaScore = float(args['deltaRankCutoff'])

    limitSigSize = args['limitSignatureSize']
    maxNumberOfProteins = len(train_dataset.genes)
    if limitSigSize:
        maxNumberOfProteins = len(train_dataset.samples)


    # FOR METHOD... IN RANKS...
        storeSignaturesMaxSize(scores, method, signatures_data, maxNumberOfProteins)





    # todo plot the freq considereing x-axis: topN, 1 < N < 100
    # sum the freq, from 1 to 100 and rank the proteins
    # plot the freq of top 10 proteins



    #!! consider adding this estimator 
    estimators.append({'name': 'Bagging Classifier Nearest Centroid', 'model': BaggingClassifier(base_estimator=NearestCentroid(), n_estimators=n_estimators), 'lambda_name':'model__max_features', 'lambda_grid': np.array([0.5, 0.75, 1.0])}) #predict_proba(X)

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
    max_n = args['topN']

    # if total number of proteins is <= args['n_small']
        #compute all possible combinations

    # if total number of proteins is > args['n_small']
        # small signatures from all proteins
    max_sig_size = args['nSSearch']



    print('Number of signatures to test: %d' % len(signatures_data.keys()))
    
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
