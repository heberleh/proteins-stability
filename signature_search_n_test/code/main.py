

# =========== ARGS ================

# import the necessary packages
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()


ap.add_argument('--no_filter', help='The algorithm will not filter proteins by Wilcoxon/Kruskal tests.', action='store_true')

ap.add_argument('--only_filter', help='The algorithm will only test the dataset composed by filtered proteins.', action='store_true')

ap.add_argument('--fdr', help='Correct the Wilcoxon/Kruskal p-values used for filtering proteins by False Discovery Rate.', action='store_true')

ap.add_argument('--p_value', help='Proteins are discarted if their p-values from Kruskal/Wilcoxon are greater or equal to the value of this argument (--p_value). Common choices: 0.01, 0.05.', action='store', default=0.05, type=float)

ap.add_argument('--train', help='Path for the train dataset.', action='store', required=True)

ap.add_argument('--test', help='Path for the independent test dataset.', action='store', required=True)

ap.add_argument('--n_s_search', help='Set the maximum number of proteins to search for small signatures formed by all prot. combinations (signature size).', action='store', type=int, default=3)

ap.add_argument('--n_small', help='Set the number of proteins considered small. If the total number of proteins in a dataset is smaller or equal than n_small, it will compute all combinations of proteins to form signatures. Otherwise, it will consider n_s_search to compute only combinations of size up to the value set for this parameter.', action='store', type=int, default=10)

ap.add_argument('--top_n', help='Create all combinations of top-N signatures from the average of ranks.', action='store', type=int, default=10)

ap.add_argument('--t_test', help='Uses T-Test instead of Wilcoxon when the dataset has 2 classes.', action='store_true')


args = vars(ap.parse_args())
#args["p_value"]
print(args)
exit()


if args['no_filter']:
    filter_options = ['no_filter']
elif args['only_filter']:
    filter_options = ['filter']
else:
    filter_options = ['filter','no_filter']


# read datasets
#train_dataset
#test_dataset
#create folder results+date

for option in filter_options:
    if option == 'filter':
        test_name = ''
        p_values = []
        # if multiclass
            # Kruskal
            test_name = 'Kruskal Wallis p-values histogram'

        # if 2-class
            if args['t_test']:
                test_name = 'T-test p-values histogram'
            
            else:
                #wilcoxon
                test_name = 'Wilcoxon p-values histogram'

        # print p-value histogram
        saveHistogram(filename=results_path+'p_values_histogram_from_filter_no_correction', values=p_values, title=test_name)

        if args['fdr']:
            #correct p-values
        
        cutoff = args['p_value']
        # filter proteins if p-value < cutoff

    signatures = []
    proteins_ranks = []

    #================== signatures by ranking =========================
    # for each ranking method, create signatures by selecting top-N proteins
    # if a signature exists, add the method to its methods list
    # sum the ranks position for each protein
    # SAVE ALL THE RANKS

    # svm-rfe

    # beta-binomial

    # kruskal, t-test or wilcoxon

    # ensemble variable importance
    # ... 

    # ...

    # rank by mean-rank position (from above methods)

    # create all combinations considering args['top-n'] proteins
    max_n = args['top-n']

    # if total number of proteins is <= args['n_small']
        #compute all possible combinations

    # if total number of proteins is > args['n_small']
        # small signatures from all proteins
        max_sig_size = args['n_s_search']




def saveHistogram(filename, values, title):
    return None


