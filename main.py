# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 13:22:02 2015

@author: Henry

This script can be adapted to your problem.

Usual parameters to change:
    - when loading data, you can set True or False for std (z-score) - DataSet class
    - when creating the object CrossValidation() you can set:
            - the K of K-fold crossvalidation (look for variable globalK)
            - the N of N repetitions of K-fold (look for variable globalN)
"""


from wilcoxon import WilcoxonRankSumTest
import time
import sys
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances
from sklearn import manifold
import sklearn
import pylab
import numpy as np
from classifier import *
from crossvalidation import CrossValidation
from dataset import Dataset


nsc ={'id': "NSC", 'genes': ["RPS27A_P62988","ACPP_P15309","PTPRS_Q13332","HSP90AA1_P07900","HSPA8_P11142","CP_P00450","IGHA1_P01876","CNDP2_Q96KP4","PRDX6_P30041","ALB_P02768","TF_P02787","CKB_P12277","ANXA1_P04083","CFB_P00751","HEXB_P07686","IDH1_O75874","GGT1_P19440","PARK7_Q99497","C3_P01024","EIF5A_P63241","SFN_P31947","PKM2_P14618","HSP90AB1_P08238","MME_P08473","ADAMTS1_Q9UHI8","BASP1_P80723","MYH9_P35579","LGALS3BP_Q08380","TTR_P02766","DYNC1H1_Q14204","ACTN4_O43707","CAT_P04040","RNASE4_P34096","IGHA2_P01877","YWHAQ_P27348","TPI1_P60174","AZGP1_P25311","VAT1_Q99536","BTD_P43251","GAA_P10253","DPP4_P27487","FOLH1_Q04609","CAPZB_P47756","CLTC_Q00610","CRISPLD2_Q9H0B8","ORM1_P02763","LAMA5_O15230","FBP1_P09467","APLP2_Q06481","CFL1_P23528","YWHAG_P61981","IGKC_P01834","CPE_P16870","TGM4_P49221","RHOC_P08134","FGB_P02675","FASN_P49327","APOA1_P02647","CST3_P01034","CD109_Q6YHK3","RHOA_P61586","ANXA2_P07355","FSTL1_Q12841","PSAP_P07602","CFH_P08603","C4B_P0C0L5","PGK1_P00558","TIMP1_P01033","TUBB2C_P68371","CHIT1_Q13231","EEF2_P13639","ANXA3_P12429","EEF1AL3_Q5VTE0","ARHGDIB_P52566","GDI2_P50395","PRDX1_Q06830","VCP_P55072","EZR_P15311","_P01623","_P01742","A1BG_P04217","SIAE_Q9HAT2","TIMP2_P16035","LCN2_P80188","EEF1A2_Q05639","GSN_P06396","CCT6A_P40227","SYTL1_Q8IYJ3","RNASET2_O00584","KLK2_P20151","EPHX2_P34913","CTSD_P07339","SOD1_P00441","LGMN_Q99538","SORD_Q00796","CMPK1_P30085","GAPDH_P04406","IGLC1_P01842","SEMG1_P04279","PEBP1_P30086","GMPR_P36959","RAB3D_O95716","DDAH1_O94760","CTBS_Q01459","VAMP8_Q9BV40","AGRN_O00468","EEF1A1_P68104","PTGR1_Q14914","ARHGDIA_P52565","PSMA7_O14818","CAP1_Q01518","HRG_P04196","FAM129A_Q9BZQ8","HSPB1_P04792","APOA2_P02652","PI15_O43692","SERPINF1_P36955","HPR_P00739","LSAMP_Q13449","ACE_P12821","MDH1_P40925","PFN1_P07737","HBB_P68871","IGJ_P01591","MLPH_Q9BV36","ABP1_P19801","SYT7_O43581","LTF_P02788","PTPRF_P10586","FGA_P02671","TUBB3_Q13509","ALDH9A1_P49189","ENO3_P13929","SHISA5_Q8N114","HINT1_P49773","PYGB_P11216","IGHM_P01871","IGHG1_P01857","NDRG1_Q92597","HSPA1A_P08107","ENO1_P06733","SELENBP1_Q13228","DNASE1_P24855","IGF2R_P11717","PPIA_P62937","PGLS_O95336","SEMG2_Q02383","LDHA_P00338","PIGR_P01833","FGG_P02679","GDI1_P31150","GSS_P48637","PLG_P00747","RAB3B_P20337","CDH1_P12830","FCGBP_Q9Y6R7","C4A_P0C0L4","SMPDL3B_Q92485","THBS1_P07996","YWHAE_P62258","ACLY_P53396","ORM2_P19652","RAB1B_Q9H0U4","CNTNAP2_Q9UHC6","HIST1H4A_P62805","SPINT1_O43278","HPX_P02790","GSTP1_P09211","S100A11_P31949","A2M_P01023","PDCD6IP_Q8WUM4","HP_P00738","RPL7A_P62424","PRSS8_Q16651","TMSL6_A9Z1Y9","CALM1_P62158","HBA1_P69905","PIP_P12273","_P01761","RNH1_P13489","MSMB_P08118","DOPEY2_Q9Y3R5","ACTN1_P12814","SLC44A4_Q53GD3","ARF3_P61204","PEBP4_Q96S96","DSTN_P60981","NPNT_Q6UXI9","NAGLU_P54802","ACTC1_P68032","GOLM1_Q8NBJ4","S100A8_P05109","AGR2_O95994","IGHG4_P01861","SMS_P52788","CALML5_Q9NZT1","XRCC6_P12956","PSMA5_P28066","COL12A1_Q99715","PPP2R1A_P30153","GP2_P55259","YWHAB_P31946","KIF5B_P33176","AKR1A1_P14550","MUC6_Q6W4X9","PIP4K2C_Q8TBX8","ALDOA_P04075","HSPD1_P10809","CLIC1_O00299","MARCKS_P29966","GSR_P00390","RAD23B_P54727","ST6GAL1_P15907","RAB14_P61106","AFM_P43652","F2_P00734","TKT_P29401","LAMC1_P11047","SERPINH1_P50454","ANXA5_P08758","FABP5_Q01469","CAPG_P40121","FTH1_P02794","IGHG3_P01860","NOV_P48745","DYNC1I2_Q13409","KIF5C_O60282","AKAP12_Q02952","TUBB6_Q9BUF5","_P01764","OLA1_Q9NTK5","GLG1_Q92896","CAD_P27708","KNG1_P01042","CANT1_Q8WVQ1","HIST2H3A_Q71DI3","CPB1_P15086","DNAJA1_P31689","YWHAZ_P63104","SET_Q01105","HUWE1_Q7Z6Z7","CD9_P21926","SOD3_P08294","PSMB4_P28070","LAP3_P28838","PSMD2_Q13200","PHGDH_O43175","HBD_P02042","GMPS_P49915","TC2N_Q8N9U0","TUBA1A_Q71U36","GTF2I_P78347","GLO1_Q04760","RAB27A_P51159","ANPEP_P15144","HIST1H3A_P68431","ASAH1_Q13510","CANX_P27824","ACTG1_P63261","DCXR_Q7Z4W1","FKBP4_Q02790","LIFR_P42702","PSMA1_P25786","ACTR2_P61160","LSP1_P33241","TUBB2B_Q9BVA1","KRT10_P13645","FBLN2_P98095","_P04206","AGA_P20933","ALDH1A3_P47895","LRG1_P02750","OTUB1_Q96FW1","KRT6B_P04259","HIST1H2AJ_Q99878","HIST2H2AA3_Q6FI13","CDC42_P60953","FLNB_O75369","_P06326","KRT9_P35527","MYH10_P35580","CAPZA1_P52907","KRT2_P35908","BPIL1_Q8N4F0","ARG1_P05089","DDX18_Q9NVP1","PSMA2_P25787","LDHB_P07195","PRDX4_Q13162","ARF1_P84077","SULT2B1_O00204","ITIH4_Q14624","TGOLN2_O43493","PHB_P35232","GNMT_Q14749","P4HB_P07237","PIK3IP1_Q96FE7","STIP1_P31948","UBE2N_P61088","RNASE3_P12724","NCLN_Q969V3","PXDN_Q92626","GCN1L1_Q92616","ADH5_P11766","CCT2_P78371","APOB_P04114","APOA1BP_Q8NCW5","GFPT1_Q06210","H3F3A_P84243","ADAM10_O14672","NAMPT_P43490","KRT1_P04264","DPP3_Q9NY33","RPL15_P61313","CORO1A_P31146","GC_P02774","SND1_Q7KZF4","CALR_P27797","TGM2_P21980","THSD4_Q6ZMP0","RPS4X_P62701","SEC23A_Q15436","TUBA3E_Q6PEY2"]}
svmrfe = {'id': "SVM-RFE", 'genes': ["HSP90AA1_P07900","MME_P08473","PROS1_P07225","VAMP8_Q9BV40","TIMP2_P16035","CNDP2_Q96KP4","CACNA2D1_P54289","EIF5A_P63241","RPS27A_P62988","TIMP1_P01033","MSMB_P08118","AKAP12_Q02952","HEXB_P07686","VAT1_Q99536","CFL1_P23528","SERPINF1_P36955","IGHA1_P01876","SFN_P31947","GLG1_Q92896","_P01772","SHISA5_Q8N114","_P01622","GGT1_P19440","RAB3D_O95716","ADAMTS1_Q9UHI8","THBS1_P07996","_P01742","SEZ6L2_Q6UXD5","ANXA1_P04083","PI15_O43692","IGHA2_P01877","SERPINA1_P01009","GP2_P55259","LGALS3BP_Q08380","IGHG2_P01859","HRG_P04196","RNASE3_P12724","CAPZB_P47756","ITIH4_Q14624","ITIH2_P19823","NOV_P48745","IGJ_P01591","ALB_P02768","CAPG_P40121","TGM4_P49221","ACTN4_O43707","HIST1H2AB_P04908","TUBA3E_Q6PEY2","PTPRS_Q13332","NME1_P15531","C1R_P00736","IGF2R_P11717","ANXA2_P07355","ORM1_P02763","CFB_P00751","CPB1_P15086","PSMA7_O14818","CD109_Q6YHK3","RAB3B_P20337","ACPP_P15309","HIST1H2AH_Q96KK5","RPS3_P23396","SPON2_Q9BUD6","FTH1_P02794","TMSL6_A9Z1Y9","_P01761","_P04208","SOD1_P00441","GFPT1_Q06210","CHIT1_Q13231","HIST2H2AA3_Q6FI13"]}
ttest = {'id': "T-Test",'genes':["RPS27A_P62988","PTPRS_Q13332","ANXA1_P04083","CNDP2_Q96KP4","TIMP2_P16035","ACPP_P15309","HSP90AA1_P07900","PRDX6_P30041","GGT1_P19440","CP_P00450","EIF5A_P63241","HSPA8_P11142","YWHAG_P61981","IGHA1_P01876","CFB_P00751","PARK7_Q99497","VAT1_Q99536","VAMP8_Q9BV40","HEXB_P07686","SFN_P31947","TF_P02787","SHISA5_Q8N114","CKB_P12277","ALB_P02768","FBP1_P09467","GMPR_P36959","RNASE4_P34096","YWHAQ_P27348","PKM2_P14618","IDH1_O75874","ADAMTS1_Q9UHI8","DYNC1H1_Q14204","HSP90AB1_P08238","CAT_P04040","BTD_P43251","ACTN4_O43707","SYTL1_Q8IYJ3","CFL1_P23528","SERPINF1_P36955","C3_P01024","_P01742","CD109_Q6YHK3","CAPZB_P47756","LAMA5_O15230","BASP1_P80723","GAA_P10253","CST3_P01034","TTR_P02766","MYH9_P35579","DDAH1_O94760","MLPH_Q9BV36","SYT7_O43581","MME_P08473","LGALS3BP_Q08380","GDI1_P31150"]}
little1 = {'id':"10-wilcoxon", 'genes': ['RNASE4_P34096', 'FGB_P02675', '_P01742', 'IGKC_P01834', 'RPS27A_P62988', 'TTR_P02766', 'ADAMTS1_Q9UHI8', 'IGHM_P01871', 'CKB_P12277', 'YWHAG_P61981']}
little2 = {'id':"20-wilcoxon", 'genes': ['RNASE4_P34096', 'FGB_P02675', '_P01742', 'IGKC_P01834', 'RPS27A_P62988', 'TTR_P02766', 'ADAMTS1_Q9UHI8', 'IGHM_P01871', 'CKB_P12277', 'YWHAG_P61981', 'SFN_P31947', 'YWHAQ_P27348', 'A1BG_P04217', 'A2M_P01023', 'ACTN1_P12814', 'ACTN4_O43707', 'ALB_P02768', 'ANXA1_P04083', 'ANXA3_P12429', 'APOA1_P02647']}
wilcoxon_genes = {'genes': ['RNASE4_P34096', 'FGB_P02675', '_P01742', 'IGKC_P01834', 'RPS27A_P62988', 'TTR_P02766', 'ADAMTS1_Q9UHI8', 'IGHM_P01871', 'CKB_P12277', 'YWHAG_P61981', 'SFN_P31947', 'YWHAQ_P27348', 'A1BG_P04217', 'A2M_P01023', 'ACTN1_P12814', 'ACTN4_O43707', 'ALB_P02768', 'ANXA1_P04083', 'ANXA3_P12429', 'APOA1_P02647', 'BTD_P43251', 'CAP1_Q01518', 'CAT_P04040', 'CTSD_P07339', 'CPE_P16870', 'CD109_Q6YHK3', 'CP_P00450', 'CFB_P00751', 'CFH_P08603', 'CHIT1_Q13231', 'CLTC_Q00610', 'CNDP2_Q96KP4', 'C3_P01024', 'CRISPLD2_Q9H0B8', 'CST3_P01034', 'DDAH1_O94760', 'DYNC1H1_Q14204', 'EEF2_P13639', 'FOLH1_Q04609', 'GDI1_P31150', 'ARHGDIA_P52565', 'GGT1_P19440', 'GMPR_P36959', 'HEXB_P07686', 'HSP90AA1_P07900', 'HSP90AB1_P08238', 'HSPA8_P11142', 'EPHX2_P34913', 'IDH1_O75874', 'IGHA1_P01876', 'PKM2_P14618', 'LDHA_P00338', 'GAA_P10253', 'MARCKS_P29966', 'LCN2_P80188', 'OLFM4_Q6UX06', 'SERPINF1_P36955', 'PI15_O43692', 'ACPP_P15309', 'PRDX6_P30041', 'PSMA7_O14818', 'SHISA5_Q8N114', 'SPINT1_O43278', 'SYT7_O43581', 'SYTL1_Q8IYJ3', 'TGM4_P49221', 'TIMP2_P16035', 'TPI1_P60174', 'TF_P02787', 'FBP1_P09467', 'VAMP8_Q9BV40', 'VAT1_Q99536', 'PTPRF_P10586', 'HP_P00738', 'LAMA5_O15230', 'MYH9_P35579', 'PARK7_Q99497', 'SELENBP1_Q13228', 'CFL1_P23528', 'FGG_P02679', 'EIF5A_P63241', 'IGHA2_P01877', 'PTPRS_Q13332', 'IGJ_P01591', 'BASP1_P80723', 'CAPZB_P47756'], 'id': 'Wilcoxon'}

def pearson(matrix):
    """
    Computes the pearson correlation(similarity) between all lines of the matrix
    :param matrix: numeric matrix, each line correspond to a sample
    :return: a distance matrix of n samples per n samples
    """

    rows, cols = matrix.shape[0], matrix.shape[1]
    r = np.ones(shape=(rows, rows))
    p = np.ones(shape=(rows, rows))
    matrix = matrix.tolist()
    for i in range(rows):
        for j in range(i+1, rows):
            r_, p_ = pearsonr(matrix[i], matrix[j])
            r[i, j] = r[j, i] = r_
            p[i, j] = p[j, i] = p_
    return r, p


def pearson_distance(matrix):
    """
    Computes the pearson distance between all lines of the matrix
    :param matrix: numeric matrix, each line correspond to a sample
    :return: a distance matrix of n samples per n samples
    """
    r, p = pearson(matrix)
    ones = np.ones(r.shape)
    distances = ones - np.array(r)
    return distances


def pearson_squared_distance(matrix):
    """
    Computes the pearson squared distance between all lines of the matrix
    :param matrix: numeric matrix, each line correspond to a sample
    :return: a distance matrix of n samples per n samples
    """
    r, p = pearson(matrix)
    ones = np.ones(r.shape)
    distances = ones - (np.array(r)**2)
    return distances


def calc_stuff(args):
    """ Computes the k-fold crossvalidation for the multiprocess map function.
    :param args: must be in this order: classifier_type, dataset, list_of_selected_genes, k, n, scale, normalize
    :return: accuracy dictionary for N repetitions, for each classifier method, for each k-fold
    """
    global list_of_selected_genes
    global scale
    global normalize
    global globalK
    global globalN
    classifier_type, dataset = args[0], args[1]
    methods_acc = {}


    #print scale, normalize, globalN, globalK, classifier_type
    #print dataset.labels

    for method in list_of_selected_genes:
        #print method
        start_time = time.time()
        cross_validation = CrossValidation(
                                  classifier_type=classifier_type,
                                  x=dataset.get_sub_dataset(method['genes']).matrix,
                                  y=dataset.labels,
                                  k=globalK, n=globalN,
                                  scale=scale,
                                  normalize=normalize)
        cross_validation.run()
        print "Execution time of", dataset.name,",", classifier_type.name, ",",method['id'], ":",\
            time.time() - start_time, "seconds"
        l = cross_validation.get_list(metric=cross_validation.ACURACY_MEAN)
        methods_acc[method['id']] = l
    return methods_acc





# =================== Global variables, change as you want =====================
box_plot_file_name = "teste_milti_processos_zscore_3rep_10-foldcross.PNG"
globalN = 5
globalK = 10
scale = True # apply z-score to attributes in the cross-validation class?
normalize = False # apply normalization (0-1) to attributes in the cross-validation class?
classifiers_types = [SVM_linear,SVM_poly, SVM_rbf, NSC, LinearDA, GaussianNaiveBayes]  # DecisionTree,
# RandomForest, AdaBoost]  #MultinomialNaiveBayes, (non-negative...)




# =========== Loading list os biomarkers candidates to test ====================
# load selected genes from files ... TODO
list_of_complete_genes_rankings = []
list_of_selected_genes = [svmrfe, ttest, nsc, wilcoxon_genes, little1, little2]
# ==============================================================================


if __name__ == '__main__':  # freeze_support()

    # =================== Loading Datasets =========================================
    # para cada pasta dentro da pasta datasets
    # seleciona nome da pasta como ID (breast_cancer)
    # cria o tipo Dataset (leitura do conjunto de dados) com o arquivo dentro e que começa com a palavra dataset
    # mapeia o conjunto de dados com o ID (nome do arquivo) em um dicionário

    # organização das pastas
    # datasets/breast_cancer/dataset_wang.csv
    # datasets/breast_cancer/selected_genes/svm-rfe_wang.csv
    # datasets/breast_cancer/selected_genes/nsc_wang.csv

    datasets_names = ["spectral.csv"]
    datasets = []
    for name in datasets_names:
        data = Dataset(name, scale=False, normalize=False)
        datasets.append(data)
        wil = WilcoxonRankSumTest(data)
        z,p = wil.run()

        print "\n\n\nThe dataset", name, "has", len([i for i in range(len(p)) if p[i] < 0.05]), \
            "differential expression genes with p < 0.05 for Wilcoxon test.\n\n\n"
    # ==============================================================================




    # =========== Computing the tests ==============================================

    global_start_time = time.time()
    acc_list = {}
    for dataset in datasets:
        classifiers_acc = {}
        pool = multiprocessing.Pool()
        args = [(c, dataset) for c in classifiers_types]
        for r in args:
            print r
        out = pool.map(calc_stuff, args)
        for i, classifier_type in enumerate(classifiers_types):
            classifiers_acc[classifier_type.name] = out[i]
        acc_list[dataset.name] = classifiers_acc

        #to-do save each dataset results

    # ==============================================================================
    print "\n\nTime to finish the complete test:", time.time()-global_start_time, "seconds.\n\n"





    # ====================== Creating Box plots =====================================

    # creates matrix for panda box plot
    classifiers_names = []
    for classif in classifiers_types:
        classifiers_names.append(classif.name)

    methods_label = []
    for dataset in datasets:
        current_classifier_res = acc_list[dataset.name]
        label_ready = False
        values_matrix = []
        for classifier_type in classifiers_types:
            current_methods_res = current_classifier_res[classifier_type.name]
            values = []
            for method in list_of_selected_genes:
                l = current_methods_res[method['id']]
                size = len(l)
                values += l
                if not label_ready:
                    methods_label += size*[method['id']]
            label_ready = True
            values_matrix.append(values)

        box_plot_matrix = np.matrix(values_matrix).transpose()
        y_min = np.min(box_plot_matrix) - 0.02
        df = pd.DataFrame(box_plot_matrix, columns=classifiers_names)
        df['Genes List'] = pd.Series(methods_label)
        pd.options.display.mpl_style = 'default'

        df.boxplot(by='Genes List')

        fig = plt.gcf()
        plt.ylim(y_min, 1.02)
        fig.set_size_inches(15,15)
        plt.savefig(box_plot_file_name,dpi=400)
    # ==============================================================================

    # todo
    # embaralhar as classes
    #

    # todo
    # criar listas de genes aleatórios e de mesmo tamanho das listas encontradas por cada método
    # plotar de cada método a comparação


    # todo
    # gráfico em linha mostrando a acurácia média conforme aumenta o valor de N em cada ranking
    # atribuir valor a variável ja criada list_of_complete_genes_rankings





    # ============== Multidimensional Projection Overview ==========================
    # t-sne projection of samples
    metrics = ["pearson","euclidean", "pearson_squared"]
    for dataset in datasets:
        for method in list_of_selected_genes:
            for metric in metrics:
                cmatrix = dataset.get_sub_dataset(method['genes']).matrix
                print cmatrix
                print
                print cmatrix[0]
                print
                print cmatrix[0,1]
                distances = []
                #try:
                if metric == "pearson":
                    distances = pearson_distance(cmatrix)
                elif metric == "pearson_squared":
                    distances = pearson_squared_distance(cmatrix)
                else:
                    distances = pairwise_distances(cmatrix.tolist(), metric=metric)

                #print distances

                t_sne = sklearn.manifold.TSNE(n_components=2, perplexity=20, init='random',
                                      metric="precomputed",
                                      random_state=7, n_iter=200, early_exaggeration=6,
                                      learning_rate=1000)
                coordinates = t_sne.fit_transform(distances)

                c = pd.factorize(dataset.labels)[0]
                categories = np.unique(c)

                x = [e[0] for e in coordinates]
                y = [e[1] for e in coordinates]

                fig = pylab.figure(figsize=(20,20))
                ax = fig.add_subplot(111)
                ax.set_title("TSNE projection using "+method["id"]+" selected proteins and "+metric+" distance",fontsize=12)
                ax.grid(True,linestyle='-',color='0.75')
                scatter = ax.scatter(x, y, c=c, marker = 'o',
                                     cmap=plt.get_cmap('Set1', len(categories)),s=200)
                plt.savefig("samples_projection_t-sne_with_"+metric+"_dist_and_"+method["id"]+"_selected_proteins.pdf")
                #except:
                #    print "Unexpected error:", sys.exc_info()[0]
                # tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=[ids[i] for i in unknown_index])
                # mpld3.plugins.connect(fig, tooltip)

# global_start_time = time.time()
# acc_list = {}
# for dataset in datasets:
#     classifiers_acc = {}
#     for classifier_type in classifiers_types:
#         methods_acc = {}
#         for method in list_of_selected_genes:
#             start_time = time.time()
#             crossvalidation = CrossValidation(
#                                        classifier_type=classifier_type,
#                                       x=dataset.get_sub_dataset(method['genes']),
#                                       genes=method['genes'],
#                                       y=dataset.labels,
#                                       k=globalK, n=globalN,
#                                       scale=scale,
#                                       normalize=normalize)
#             crossvalidation.run()
#            # print "Executou com", dataset.name,",", classifier_type.name, ",",method['id'], "em",
# # time.time() - start_time, "segundos"
#             l = crossvalidation.get_list(metric=crossvalidation.ACURACY_MEAN)
#             methods_acc[method['id']] = l
#         classifiers_acc[classifier_type.name] = methods_acc
#     acc_list[dataset.name] = classifiers_acc
#
# print "\n\nTempo para rodar o teste completo", time.time()-global_start_time, "segundos.\n\n"
