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



O svm-rfe com o maior N e máxima acurácia dá valores maiores de N, portanto, mais proteínas são selecionadas e as frequencias delas aumentam.
Com isso ele tende a considerar mais proteínas além da ADH1... que é a primeira do rank (N=1).
E daí... com 50% de repetição já encontramos outras marcadoras adicionadas como GPB e BSA... e até a ENO1.

