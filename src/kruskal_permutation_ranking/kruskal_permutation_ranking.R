
#Installing required packages
list.of.packages <- c("combinat", "doSNOW","snow", "rpart", "parallel", "MASS", "e1071")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

if(length(new.packages)) install.packages(new.packages,repos="http://brieger.esalq.usp.br/CRAN/")

library("rpart")
library("MASS")
#library("class")
library("e1071")
#library("dismo")
#library("caret")
library("parallel")
require(foreach)
require(doSNOW)
require(combinat)


# LOOK FOR THE SPECIFIC "TEXT" TO EDIT INPUT OR PARAMETERS:
# #========================= SETUP YOUR INPUT AND PARAMETERS ==============================




#=========================== =============================== ==============================
#=========================== =============================== ==============================
# FUNCTIONS, DO NOT EDIT, UNLESS YOU WANT TO CHANGE THE SCRIPT
balanced.folds <- function (y, nfolds = min(min(table(y)), 10))
{
  totals <- table(y)
  fmax <- max(totals)
  nfolds <- min(nfolds, fmax)
  nfolds = max(nfolds, 2)
  folds <- as.list(seq(nfolds))
  yids <- split(seq(y), y)
  bigmat <- matrix(NA, ceiling(fmax/nfolds) * nfolds, length(totals))
  for (i in seq(totals)) {
    cat(i)
    if (length(yids[[i]]) > 1) {
      bigmat[seq(totals[i]), i] <- sample(yids[[i]])
    }
    if (length(yids[[i]]) == 1) {
      bigmat[seq(totals[i]), i] <- yids[[i]]
    }
  }
  smallmat <- matrix(bigmat, nrow = nfolds)
  smallmat <- permute.rows(t(smallmat))
  res <- vector("list", nfolds)
  for (j in 1:nfolds) {
    jj <- !is.na(smallmat[, j])
    res[[j]] <- smallmat[jj, j]
  }
  return(res)
}

permute.rows <- function (x)
{
  dd <- dim(x)
  n <- dd[1]
  p <- dd[2]
  mm <- runif(length(x)) + rep(seq(n) * 10, rep(p, n))
  matrix(t(x)[order(mm)], n, p, byrow = TRUE)
}

#=========================== =============================== ==============================
#=========================== =============================== ==============================





#=========================== =============================== ==============================
#=========================== =============================== ==============================
#=========================== =============================== ==============================
#=========================== SETUP YOUR INPUT AND PARAMETERS ==============================
#=========================== =============================== ==============================
#=========================== =============================== ==============================
#=========================== =============================== ==============================
# tip: try Sublime Text Editor for easy navigation in this code ;)

# PARAMETERS:    train.txt
# SEE THE train.txt AND FOLLOW THE PATTERN
input_file_name <- "./dataset/current/train.txt"
db <- read.table(input_file_name, header=TRUE, sep="\t")

nCluster <- detectCores()    # AUTOMATICALLY SELECT ALL POSSIBLE CORES FOR PARALLEL PROCESSING.
#nCluster <- 8               # MANUALLY SET THE NUMBER OF CORES/NUCLEUS FOR PARALLEL PROCESSING.

class_delta <- 1

TUNE <- FALSE
#=========================== =============================== ==============================
#=========================== =============================== ==============================
#=========================== =============================== ==============================



fileConn<-file("log.txt")






#=========================== =============================== ==============================
#=========================== ===== PRE-PROCESSING DATA ===== ==============================
db2 <-t(db)
dataset.x <- as.matrix(db2[3:nrow(db2),2:(ncol(db2))])
class(dataset.x) <- "numeric"
colnames(dataset.x)<-db2[2,2:ncol(db2)]

writeLines(paste(rownames(dataset.x)), fileConn)
writeLines("\n\n", fileConn)

# first line of original matrix is already col/rownames
#print(db2[3:nrow(db2),1]) -> classes
#print(db2[3:nrow(db2),2]) -> values
dataset.y <- as.factor(db2[3:nrow(db2),1])

writeLines(paste(table(dataset.y)), fileConn)

# z-score? Models like SVM have padronization by z-score by default.
# Do not use this z-score code, unless you really want and know what
# you are doing.
# part of the code padronizes by lines.
# the other part padronizes by columns.
#x = log(x)
#scale samples with mean zero and standard deviation one
#for(i in 1:nrow(x))x[i,] = (x[i,]-mean(x[i,]))/sd(x[i,])
#scale features with mean zero and standard deviation one
#for(i in 1:ncol(x))x[,i] = (x[,i]-mean(x[,i]))/sd(x[,i])
#x = 2*atan(x/2)

#featureRankedList = svmrfeFeatureRankingForMulticlass(dataset.x, dataset.y)

# selected_outer = list()
# final_outer_error = vector()
# final_outer_error_min = vector()
# final_outer_error_max = vector()
# final_N = vector()
# logdouble = list()
#=========================== =============================== ==============================
#=========================== =============================== ==============================




#=========================== =============================== ==============================
#=========================== =============================== ==============================
# Defining the K-Fold cross-validation in outer-loop
# Each test fold is independent from inner-loop, where rankings are being computed
# We do not compute mean of K-Folds accuracy. We are interested in its distribution only.
# Doing that we can analyse the stability of accuracy of the new ranking method
# and the stability of new rankings generated as well

folds <- balanced.folds(y=dataset.y)
nfold <- length(folds)
allrep <- list()

cat("Number of folds", nfold,"\n")

writeLines(paste("Number of folds: ",nfold), fileConn)


# Set up parallel computing of K-Fold cross-validation
cl <- makeCluster(nCluster,type="SOCK")
registerDoSNOW(cl)
getDoParWorkers()




#=========================== =============================== ==============================
#=========================== ========= OUTER LOOP ========== ==============================
# Run parallel computing of K-Fold
stime <- system.time({
allrep <- foreach (i = 1:nfold, .combine="cbind", .packages = c("e1071","dismo","class","rpart","MASS")) %dopar%{

# global genes frequency
global_genes_freq <- rep(0, ncol(dataset.x))



#for (i in 1:nfold){ #non-parallel test

  # selected_inner = list()
  # outer_error = vector()
  # mean_error_by_N =  list()
  # Nf = vector()
  # logouter = list()
  # outer_N = vector()

  # Define ith train and test set
  fold_snames_train <-  rownames(dataset.x[-folds[[i]], ,drop=FALSE])
  fold_snames_test <- rownames(dataset.x[folds[[i]], ,drop=FALSE])

  dataset2.x <- dataset.x[-folds[[i]], ,drop=FALSE]
  dataset2.y <- dataset.y[-folds[[i]]]
  dataset2.testX <- dataset.x[folds[[i]], ,drop=FALSE]
  dataset2.testY <- dataset.y[folds[[i]]]
  dataset2.class_count <- table(dataset2.y) #classes count
  dataset2.class_levels <- unique(dataset2.y)
  dataset2.n_classes <- length(dataset2.class_levels)
  dataset2.n_samples <- nrow(dataset2.x)
  dataset2.n_genes <- ncol(dataset2.x)

  genes_freq <- rep.int(0,dataset2.n_genes)


  #=========================== ===============================
  #=========================== ========= INNER LOOP ==========
  # For each train set we will compute all combinations of samples, removing 1 sample of each class/condition
  # For example: if we have 10 samples per class, the new train set will have 9 samples per class
  #              The new train set is used to calculate 1 rank
  #              There will be many possibilities. Each one will derive 1 rank.
  #              We increment, then, the frequency of genes in those ranks when they have p-value < 0.05

  # number of samples will be combined


  # Computes All possible combinations, they will form the loop
  init <- 1
  class_combinations <- list()
  for (c in 1:length(dataset2.class_count)){
      intra_class_combinations <- list()
      c_size <- dataset2.class_count[[c]]
      if(c_size - class_delta < 3){
        stop("Error: c_size - class_delta must be > 2. Number os samples per class in inner-loop must be > 2.")
      }
      end <- init + c_size - 1
      comb_matrix <- combn(init:end, c_size - class_delta)
      #print(comb_matrix[,1:5])
      #print(list(comb_matrix[,1]))

      for (comb_i in 1:ncol(comb_matrix)){
        intra_class_combinations <- c(intra_class_combinations,list(comb_matrix[,comb_i]))
      }
      #print(intra_class_combinations)
      init <- end + 1
      class_combinations <- c(class_combinations, list(intra_class_combinations))
  }
  cpsizes <- list()
  for (ci in 1:length(class_combinations)){
    cpsizes <- c(cpsizes,list(1:length(class_combinations[[ci]])))
  }

  indexes_combinations <- do.call("expand.grid", as.list(cpsizes))

  #print(combinations)


  #print(combinations[1:5])

  cat("\nNumber of possible combinations of ", dataset2.n_samples, "samples in inner-loop is ", nrow(indexes_combinations), "\n\n")

  #print(indexes_combinations[1:5,])

  # Compute p-values for each gene of each combination
  for (j in 1:nrow(indexes_combinations)){
    #cat("\n=== Combination-",j,"===\n")

    combination_indexes <- as.vector(as.matrix(indexes_combinations[j,]))
    current_combination <- c()
    ci <- 1
    for (index in combination_indexes){
      current_combination <- c(current_combination,class_combinations[[ci]][[index]])
      ci <- ci + 1
    }

    #cat("\nCurrent combination: \n")
    #print(current_combination)

    # Define jth train and test set
    dataset3.x <- dataset2.x[current_combination, ,drop=FALSE]
    dataset3.y <- dataset2.y[current_combination]
    dataset3.testX <- dataset2.x[-current_combination, ,drop=FALSE]
    dataset3.testY <- dataset2.y[-current_combination]

    dataset3.class_count <- table(dataset3.y) #classes count
    dataset3.class_levels <- unique(dataset3.y)
    dataset3.n_classes <- length(dataset3.class_levels)
    dataset3.n_samples <- nrow(dataset3.x)
    dataset3.n_genes <- ncol(dataset3.x)

    #cat("\nNumber of samples per class: \n", dataset3.class_count,"\n\n")
    #cat("\nNumber of samples in this fold: ", dataset3.n_samples, "\n")
    #cat("\nClasses of dataset3: ", unique(dataset3.y), "\n")


    for (c in unique(dataset3.y)){
      if(matrix(dataset3.class_count[c])[1,1] < 3){
        stop("Error: number os samples per class in inner-loop must be > 2.")
      }
    }

    #print(dataset3.y)
    #print(dataset3.x[,1])


    # Kruskal-test for each gene
    for (g in 1:dataset3.n_genes){
        ktest <- kruskal.test(x=dataset3.x[,g], g=dataset3.y)
        if (as.numeric(ktest[3][1]) < 0.05){
          genes_freq[g] <- genes_freq[g] + 1
        }
    }
    #print(genes_freq)

  }

  #print(genes_freq)

  # Store genes freq
  genes_freq_matrix <- cbind(cbind(colnames(dataset.x),genes_freq),t(dataset.x))
  colnames(genes_freq_matrix) <- append(c("protein","protein frequency"),rownames(dataset.x))
  write.matrix(genes_freq_matrix, file =paste("./results/kruskal_permuted_rank/genes_freq_fold_",i,".csv",collapse=""), sep=",")


  # Update global_genes_freq
  global_genes_freq <- global_genes_freq + genes_freq

  #===========================
  # Permutate class labels for gene's frequency distribution in randomized data
  permuted_dataset.x <- dataset2.x[sample(dataset2.n_samples),]
  permuted_dataset.y <- dataset2.y
  genes_freq_at_random <- rep.int(0,dataset2.n_genes)

  # Compute p-values for each gene of each combination
  for (j in 1:nrow(indexes_combinations)){
    combination_indexes <- as.vector(as.matrix(indexes_combinations[j,]))
    current_combination <- c()
    ci <- 1
    for (index in combination_indexes){
      current_combination <- c(current_combination,class_combinations[[ci]][[index]])
      ci <- ci + 1
    }

    # Define jth train and test set
    permuted_dataset2.x <- permuted_dataset.x[current_combination, ,drop=FALSE]
    permuted_dataset2.y <- permuted_dataset.y[current_combination]

    permuted_dataset2.class_count <- table(permuted_dataset2.y) #classes count
    permuted_dataset2.class_levels <- unique(permuted_dataset2.y)
    permuted_dataset2.n_classes <- length(permuted_dataset2.class_levels)
    permuted_dataset2.n_samples <- nrow(permuted_dataset2.x)
    permuted_dataset2.n_genes <- ncol(permuted_dataset2.x)

    # Kruskal-test for each gene
    for (g in 1:permuted_dataset2.n_genes){
        ktest <- kruskal.test(x=permuted_dataset2.x[,g], g=permuted_dataset2.y)
        if (as.numeric(ktest[3][1]) < 0.05){
          genes_freq_at_random[g] <- genes_freq_at_random[g] + 1
        }
    }
  }
  #print(genes_freq_at_random)
  #===========================
  # end permuted dataset


  # Store genes freq
  rand_freq_matrix <- cbind(colnames(dataset.x),genes_freq_at_random)
  colnames(rand_freq_matrix) <- c("protein","protein frequency")
  write.matrix(rand_freq_matrix, file =paste("./results/kruskal_permuted_rank/genes_freq_at_random_fold_",i,".csv",collapse=""), sep=",")


  # Rank the proteins by higher frequency
  ranking_index <- order(genes_freq, decreasing=T)
  rank_accuracy <- {}
  bestN <- 1
  bestAcc <- 0

  # Test accuracy of ranking using test set for N first proteins in ranking
  best <- list(parameters=list(2))
  tuned <- list(best=best)
  tuned$best.parameters[1] = 10^(-4)
  tuned$best.parameters[2] = 10
  for(nfeatures in 1:ncol(dataset.x)){
    # Choose best parameters for svm model
    if (TUNE){
      tuned <- tune.svm(x=dataset2.x[,ranking_index[1:nfeatures]], y=dataset2.y, gamma = 10^(-6:-1), cost = 10^(1:2))
    }

    # Create svm model
    svmModel <- svm(x=dataset2.x[,ranking_index[1:nfeatures]], y=dataset2.y, cost = tuned$best.parameters[2], gamma=tuned$best.parameters[1], cachesize=100,  scale=T, type="C-classification", kernel="linear")

    # Test the model
    pred <- predict(svmModel, dataset2.testX[,ranking_index[1:nfeatures]])

    exp_pred <- factor(dataset2.testY, levels=levels(pred))

    # Calculate accuracy
    accuracy <- (length(which(as.logical(pred == exp_pred)))/length(dataset2.testY))

    rank_accuracy<-rbind(rank_accuracy,accuracy)

    # If it is the best accuracy, select N
    if (rank_accuracy[bestN] < accuracy){
      bestN <- nfeatures
      bestAcc <- accuracy
    }

  }

  accuracy_by_genes_original_position <- rep(0.0,ncol(dataset.x))
  for (g in 1:ncol(dataset.x)){
    acc <- rank_accuracy[g]
    index <- ranking_index[g]
    accuracy_by_genes_original_position[index] <- acc
  }




  result <- list(fold=i, g_freqs = genes_freq, g_freq_at_random = genes_freq_at_random, N = bestN, bestAccuracy = bestAcc, accuracy_by_org_g_pos = accuracy_by_genes_original_position, train_names = fold_snames_train, test_names= fold_snames_test)

  result
  # Compute new rank for this fold
  # genes_freq -> rank

  # Plot the gene's frequency distribution from each fold, versus random frequencies

  #sink()
}#
})[3]
stopCluster(cl)
stime
# end K-Fold


global_genes_random_freqs <- c()
global_genes_freqs <- c()
global_rank_accuracy <- {}
global_n_values <- {}
train_names <- {}
test_names <- {}
i_order <- c()
bestAcc <- {}
for (k in 1:nfold){
  i <- allrep[[1,k]]
  i_order <- c(i_order,i)
  global_genes_freqs <- cbind(global_genes_freqs, allrep[[2,i]])
  global_genes_random_freqs <- cbind(global_genes_random_freqs, allrep[[3,i]])
  global_n_values <- cbind(global_n_values, allrep[[4,i]] )
  bestAcc <- cbind(bestAcc, allrep[[5,i]])
  global_rank_accuracy <- cbind(global_rank_accuracy, allrep[[6,i]])
  train_names <- cbind(train_names, allrep[[7,i]])
  test_names <- cbind(test_names, allrep[[8,i]])
}

print(i_order)

print(global_genes_random_freqs[1:5,])
print(global_genes_freqs[1:5,])

global_genes_freqs <- as.matrix(global_genes_freqs)
global_genes_freqs <- apply(global_genes_freqs, 1,as.numeric)

global_genes_random_freqs <- as.matrix(global_genes_random_freqs)
global_genes_random_freqs <- apply(global_genes_random_freqs, 1,as.numeric)

# Compute Wilcoxon rank sum test for freq of each gene compared to permuted data
gene_p_value <- rep(-1,ncol(dataset.x))
for (g in 1:ncol(dataset.x)){
  #print(global_genes_freqs[,g])
  #print(global_genes_random_freqs[,g])
  wtest <- wilcox.test(global_genes_freqs[,g], global_genes_random_freqs[,g])
  #print(wtest[3][1])
  #cat("=================================")
  gene_p_value[g] <- as.numeric(wtest[3][1])
}


complete_train_p_value <- c()
# Kruskal-test for each gene using complete train data
for (g in 1:ncol(dataset.x)){
    ktest <- kruskal.test(x=dataset.x[,g], g=dataset.y)
    complete_train_p_value <- c(complete_train_p_value,as.numeric(ktest[3][1]))
}

print(dim(global_genes_freqs))
print(dim(complete_train_p_value))
print(ncol(dataset.x))
total_freq_genes <- colSums(global_genes_freqs)
print(dim(total_freq_genes))
print(dim(gene_p_value))
# Store genes freq, freq at random, and p-values
p_matrix <- cbind(cbind(cbind(cbind(colnames(dataset.x),complete_train_p_value),total_freq_genes),gene_p_value),cbind(cbind(t(global_genes_freqs),t(global_genes_random_freqs)), global_rank_accuracy))

colnames(p_matrix) <- c("protein",c("complete_train_p_value",c("total_freq",c("p-value_orig_vs_random",append(append(i_order,rep('permuted',nfold)),i_order)))))

write.matrix(p_matrix, file =paste("./results/kruskal_permuted_rank/genes_freq_all_folds.csv",collapse=""), sep=",")


global_n_values <- as.matrix(global_n_values)
colnames(global_n_values) <- i_order
write.matrix(rbind(global_n_values, bestAcc), file =paste("./results/kruskal_permuted_rank/selected_N_acc_each_folder.csv",collapse=""), sep=",")

colnames(train_names) <- i_order
write.matrix(train_names, file =paste("./results/kruskal_permuted_rank/sample_names_train_k-fold.csv",collapse=""), sep=",")

colnames(test_names) <- i_order
write.matrix(test_names, file =paste("./results/kruskal_permuted_rank/sample_names_test_k-fold.csv",collapse=""), sep=",")


writeLines("\n", fileConn)
close(fileConn)

# Plot the global gene's frequency distribution versus random freq distr.


# Scaterplot de  ranks finais de cada fold 2 a 2, quanto mais na diagonal, melhor.. como no trbaalho de ideker.
# Definir forma de verificar estabilidade das listas... como quantificar? como dizer se é alta ou baixa?
# Gerar permutações de dados, várias, e calcular estabilidade... como referência de baixa estbailidade.... verificar quão diferente é? vai ser mto diferente, não? -.- hum... q mais?

# !!!!!!!!!!! Comparar estabilidade de Kruskal normal com a nova proposta.. via frqueências!!!!










# double_error = vector()
# all_N = vector()
# for (rep in 1:repetition){
#   double_error[rep] = allrep[[4,rep]]
#   all_N[rep] = allrep[1,rep]
# }

# double_error = as.vector(unlist(double_error))
# all_N = as.vector(unlist(all_N))
# mean_double_error = mean(double_error)
# min_double_error = min(double_error)
# max_double_error = max(double_error)
# dif_double_error = abs(double_error - mean_double_error)
# min_dif = min(dif_double_error)
# index_error = which(dif_double_error == min_dif)
# min_N = min(all_N[index_error])
# index_N = which(all_N == min_N)
# index = intersect(index_N,index_error)
# final_double_error = mean(double_error[index])

# if(length(index) != 1){
#   cat ("Mais de um modelo pode ser utilizado para calcular as métricas!")
#   index = index[1]
# }

# index2 = (7*(index))
# size = length(allrep[[index2]][[index]])
# pred = list()
# ref = list()
# for (i in 1:size){
#   obj = allrep[[index2]][[index]][[i]]
#   pred[[i]] = obj$pred
#   ref[[i]] = obj$ref
# }
# pred = as.vector(unlist(pred))
# ref = as.vector(unlist(ref))


# sink("./results/svm-rfe/caret_svm-rfe.txt")
# cat("Average accuracy of  double cross-validation repetitions: ")
# cat(mean_double_error)
# cat("\n\n")
# cat("Maximum accuracy between repetitions: ")
# cat(max_double_error)
# cat("\n\n")
# cat("Minimum accuracy between repetitions: ")
# cat(min_double_error)
# cat("\n\n")
# cat("Best model indexes: ")
# cat(index_error)
# cat("\n\n")
# cat("All possible N according to selected models:")
# cat(all_N[index_error])
# cat("\n\n")
# cat("Minimum N (selected final N): ")
# cat(min_N)
# cat("\n\n")
# cat("Double-Cross-Validation accuracy: ")
# cat(final_double_error)
# cat("\n\n")

# pred=factor(pred, levels=unique(dataset.y))
# ref=factor(ref, levels=unique(dataset.y))

# confusionMatrix(pred, ref, positive=NULL)
# sink()

# m = cbind(all_N, double_error)
# write.csv(m,"./results/svm-rfe/all_n_and_double_error_repetition_svm-rfe.csv")

# tuned = NULL
# #Tuna para encontrar parâmetros com o conjunto completo
# if (global_tune==TRUE){
#   tuned <- tune.svm(x = dataset.x, y=dataset.y, gamma = 10^(-6:-1), cost = 10^(1:2))
# }else{
#   tuned$best.parameters[1] = 10^(-4)
#   tuned$best.parameters[2] = 10
# }

# #Calcula SVM-RFE
# aux = svmrfeFeatureRankingForMulticlass(dataset.x, dataset.y, tuned)
# featureRankedList = aux$featureRankedList
# mean_weights = aux$mean_weights[featureRankedList]
# weights = aux$weights
# weights = do.call(rbind, weights)
# weights = weights[featureRankedList,]



# #6-fold
# nf <- {}
# Accuracy <- {}
# for(pow in 1550:1600){
#   nfeatures = pow
#   truePredictions = 0
#   svmModel = svm(x[, featureRankedList[1:nfeatures]], y, cost = 10, gamma=0.0001, cachesize=500,  scale=T, type="C-classification", kernel="linear",cross=5)
#   nf<-rbind(nf,nfeatures)
#   Accuracy<-rbind(Accuracy,mean(svmModel$accuracies))
# }
# plot(nf,Accuracy)

# x = dataset.x
# names = colnames(subset(x,select=featureRankedList)) # ou... colnames(x[,featureRankedList])
# x_ordered = cbind(names,t(x[,featureRankedList]))
# merged = cbind(names,featureRankedList,1:length(names),weights) #merge(names,featureRankedList, by = "row.names", all = TRUE)
#merged = as.matrix(merged[-1])

#colnames para 2 classes (dataset4):
#colnames(merged) <- c("gene name","index","rank","v1")

#colnames para 2 classes (secretoma):
#colnames(merged) <- c("gene name","index","rank","v1","v2")

#colnames para 3 classes (secretoma):
#colnames(merged) <- c("gene name","index","rank","v1","v2","v3")

# #colnames para 4 classes:
# colnames(merged) <- c("gene name","index","rank","v1","v2","v3","v4","v5","v6")

# #colnames para 5 classes:
# #colnames(merged) <- c("gene name","index","rank","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10")

# #colnames para 7 classes:
# #colnames(merged) <- c("gene name","index","rank","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15","v16","v17","v18","v19","v20","v21")

# merged = cbind(merged,x_ordered)

# #write.matrix(merged, file = "big_matrix_svm-rfe.csv", sep = ",")
# write.csv(merged, file = "./results/svm-rfe/big_matrix_svm-rfe.csv")

# #write.matrix(x_ordered, file = "matrix_by_rank_scale.csv", sep = ",")

#v = c(55,54,25,1578,108,24,1575,51,1528,26,23,1550,34,1521,23,27,13,18,1560,1554,1520,26,25,37,36,71,43,25,29,34,52,1547,29,42,1568,164)
#d = density(v)

#plot(d)
#axis(1,at=36,labels=c("36"))


# #for ranking validation with independent test
# write.matrix(featureRankedList, file ="./results/svm-rfe/rank_index_svmrfe.csv", sep=",")
# write(min_N, file="./results/svm-rfe/double_selected_N_svmrfe.txt")

