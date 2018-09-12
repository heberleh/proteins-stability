


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
