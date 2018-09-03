from pylab import *
import matplotlib.pyplot as plt
import csv

if __name__ == '__main__':
    rankes_names = ["rank_nsc.csv","rank_svm_linear.csv","rank_svm_rbf.csv","rank_kruskal.csv"]
    classifiers_names = ["nsc","svm-linear","svm-rbf", "tree"]
    
    view_rankes_names = {"rank_nsc.csv":"NSC","rank_svm_linear.csv":"SVM-RFE (linear)","rank_svm_rbf.csv":"SVM-RFE (radial)","rank_kruskal.csv": "Kruskal"}
    view_classifiers_names = {"nsc":"NSC","svm-linear":"SVM (linear)","svm-rbf":"SVM (radial)", "tree":"Decision Tree"}
    
    for rank_name in rankes_names:
        for classifier_name in classifiers_names:
            matrix = []
            with open('./results/cv/'+'cv_'+rank_name.replace('.csv','')+'_'+classifier_name+'.csv', 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=',')
                i = 0
                for row in reader:
                    if i == 0:
                        labels = row
                    else:
                        matrix.append(row) 
                    i+=1
            
            matrix = np.matrix(matrix).astype(float)
            
            # multiple box plots on one figure
            figure(figsize=(15,8),dpi=90)
            # NYCdiseases['chickenPox'] is a matrix 
            # with 30 rows (1 per year) and 12 columns (1 per month)
            boxplot(matrix)
            plt.ylim(-0.02, 1.02)

            xticks(range(1,len(labels)+1),labels, rotation=75)
            yticks()
            xlabel('Top-N proteins')
            ylabel('Accuracies')
            title(view_classifiers_names[classifier_name] + ' accuracies from '+view_rankes_names[rank_name]+' rank')
            plt.tight_layout()
            savefig('./results/cv/'+'cv_'+rank_name.replace('.csv','')+'_'+classifier_name+'.png')
            plt.close()  


            for i in range(len(matrix[0])):
                means = mean(matrix[:,i])

            plt.plot(fpr, tpr, label="Avg F1 scores")                  

            plt.legend(loc='lower right')
            plt.plot([0,1],[0,1],'r--')
            plt.xlim([-0.1,1.2])
            plt.ylim([-0.1,1.2])
            plt.ylabel('F1')
            plt.xlabel('Top-N')
            plt.tight_layout()
            savefig('./results/roc/'+'independent_roc_'+str(signature)+'_________'+classifier_name+'_'+'.png')
            plt.close()



            
