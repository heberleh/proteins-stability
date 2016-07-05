# ALGORITHM DESCRIPTION
#
#   for rep repetitions:
#        k-fold
#           -> train: rank by svm-rfe
#           -> test: acc, selected proteins
#   return (mean_acc_kfold,  selected_proteins|frequencies )
#
# overall_acc = 0
# for rep in repetitions:
#     overall_acc += rep[acc]
#     sum the freq. of each selected protein
#
#
# overall_acc/=length(repetitions)
#
# plot CV using SVM using fisrt N proteins sorting by freq.
#
# possible signature: proteins with freq >= 0.5 * length(repetitions) * k             0.5 (50%) or other cutoff
# CV with SVM using this signature, length(repetitions) times
#
# END



O svm-rfe com o menor N e máxima acurácia dá valores de N bem pequenos...
como a proteína ADH1 já classifica bem sozinha, é difícil as outras proteínas serem escolhidas!
Uma possibilidade seria pegar o Maior N de máxima acurácia, ou seja, 
considerar as proteínas que estão entre a primeira posição do ranke e a máxima posição que deria máxima acurácia.
