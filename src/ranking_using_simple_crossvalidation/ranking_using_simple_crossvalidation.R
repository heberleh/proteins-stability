
# Author: Henry Heberle
# contact: henry at icmc usp br

#Installing required packages

list.of.packages <- c("combinat", "doSNOW","snow", "rpart", "parallel", "MASS", "e1071")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

if(length(new.packages)) install.packages(new.packages,repos="http://brieger.esalq.usp.br/CRAN/")

library("rpart")
library("MASS")
#library("class")
library("e1071")
#library("dismo")
library("caret")
library("parallel")
require(foreach)
require(doSNOW)
require(combinat)



# ALGORITHM DESCRIPTION
#
#   for rep repetitions:
#        k-fold
#           -> train: rank
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
# plot CV using fisrt N proteins sorting by freq.
#
# possible signature: proteins with freq >= 0.5 * length(repetitions) * k
# CV using this signature, length(repetitions) times
#
# END

# search"SETUP YOUR INPUT AND PARAMETERS" to edit input or parameters



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


#requires library CARET
stratified_balanced_cv <- function(y, nfolds = min(min(table(y)), 10)){
  totals <- table(y)
  fmax <- max(totals)
  nfolds <- min(nfolds, fmax)
  nfolds = max(nfolds, 2)
  folds <- createFolds(y, k = nfolds, list = TRUE)

  res <- vector("list", nfolds)
  for (j in 1:nfolds) {
    res[[j]] <- folds[[j]]
  }
  return(res)
}


################################################
# Feature Ranking with SVM-RFE   REFERENCE: http://www.uccor.edu.ar/paginas/seminarios/Software/SVM_RFE_R_implementation.pdf
################################################
svmrfeFeatureRanking = function(x,y){
    n = ncol(x)

    survivingFeaturesIndexes = seq(1:n)
    featureRankedList = vector(length=n)
    rankedFeatureIndex = n

    while(length(survivingFeaturesIndexes)>0){
        #train the support vector machine
        svmModel = svm(x[, survivingFeaturesIndexes], y, cost = 10, cachesize=500,  scale=F, type="C-classification", kernel="linear" )

        #compute the weight vector
        w = t(svmModel$coefs)%*%svmModel$SV

        #compute ranking criteria
        rankingCriteria = w * w

        #rank the features
        ranking = sort(rankingCriteria, index.return = TRUE)$ix

        #update feature ranked list
        featureRankedList[rankedFeatureIndex] = survivingFeaturesIndexes[ranking[1]]
        rankedFeatureIndex = rankedFeatureIndex - 1

        #eliminate the feature with smallest ranking criterion
        (survivingFeaturesIndexes = survivingFeaturesIndexes[-ranking[1]])

    }

    return (featureRankedList)
}
################################################
# Feature Ranking with Average Multiclass SVM-RFE
################################################

auxm <- list()
svmrfeFeatureRankingForMulticlass = function(x,y,tuned){
    n = ncol(x)

    survivingFeaturesIndexes = seq(1:n)
    featureRankedList = vector(length=n)
    featureRankedListWeights = list()
    rankedFeatureIndex = n
    rankingCriteriaFull = vector()
    while(length(survivingFeaturesIndexes)>0){
        #train the support vector machine
        svmModel = svm(x[, survivingFeaturesIndexes], y, tuned$best.parameters[2], gamma = tuned$best.parameters[1], scale=T, type="C-classification", kernel="linear" )

        #compute the weight vector
        multiclassWeights = svm.weights(svmModel)

        #compute ranking criteria
        multiclassWeights = multiclassWeights * multiclassWeights
        rankingCriteria = 0

        for(i in 1:ncol(multiclassWeights))rankingCriteria[i] = mean(multiclassWeights[,i])

        #rank the features
        (ranking = sort(rankingCriteria, index.return = TRUE)$ix)

        #gene weight - which interaction
        featureRankedListWeights[[survivingFeaturesIndexes[ranking[1]]]] = multiclassWeights[,ranking[1]]
        rankingCriteriaFull[survivingFeaturesIndexes[ranking[1]]] = mean(multiclassWeights[,ranking[1]])

    #ranking[i] é o index do index do atributo. em surviving pode ter restado o atributo de index 200 na posição 1. ou seja, surv[ranking[1]] = 200 -> atributo de index 200 da tabela original.

        #update feature ranked list
        (featureRankedList[rankedFeatureIndex] = survivingFeaturesIndexes[ranking[1]])
        rankedFeatureIndex = rankedFeatureIndex - 1

        #eliminate the feature with smallest ranking criterion
        (survivingFeaturesIndexes = survivingFeaturesIndexes[-ranking[1]])
        #cat(length(survivingFeaturesIndexes),"\n")
    }

    result = list(featureRankedList=featureRankedList, mean_weights = rankingCriteriaFull, weights=featureRankedListWeights)
  return (result)
}

################################################
# This function gives the weights of the hiperplane
################################################
svm.weights<-function(model){
w=0
  if(model$nclasses==2){
       w=t(model$coefs)%*%model$SV
  }else{    #when we deal with OVO svm classification
      ## compute start-index
      start <- c(1, cumsum(model$nSV)+1)
      start <- start[-length(start)]

      calcw <- function (i,j) {
        ## ranges for class i and j:
        ri <- start[i] : (start[i] + model$nSV[i] - 1)
        rj <- start[j] : (start[j] + model$nSV[j] - 1)

      ## coefs for (i,j):
        coef1 <- model$coefs[ri, j-1]
        coef2 <- model$coefs[rj, i]
        ## return w values:
        w=t(coef1)%*%model$SV[ri,]+t(coef2)%*%model$SV[rj,]
        return(w)
      }

      W=NULL
      for (i in 1 : (model$nclasses - 1)){
        for (j in (i + 1) : model$nclasses){
          wi=calcw(i,j)
          W=rbind(W,wi)
        }
      }
      w=W
  }
  return(w)
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


class_delta <- 1

n_repetitions <- 3

TUNE <- FALSE                # PRE-TUNE PARAMETERS OF CLASSIFICATION MODELS?
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

writeLines("teste", fileConn)
writeLines(paste("Classes: ", dataset.y), fileConn)

writeLines(paste(table(dataset.y)), fileConn)

#=========================== =============================== ==============================
#=========================== =============================== ==============================





#=========================== =============================== ==============================
#=========================== ============ START ============ ==============================

folds <- stratified_balanced_cv(y=dataset.y,nfolds=5)
nfold <- length(folds)
cat("Number of folds", nfold,"\n")
writeLines(paste("Number of folds: ",nfold), fileConn)


global_genes_freq <- rep(0, ncol(dataset.x))
avg_acc_by_rep <- rep(0,n_repetitions)
stime <- system.time({
for (rep in 1:n_repetitions){
  folds <- stratified_balanced_cv(y=dataset.y,nfolds=5)
  nfold <- length(folds)


  # START K-Fold
  accs_kfold = rep(0,nfold)
  for (i in 1:nfold){
    # Define ith train and test set
    dataset2.x <- dataset.x[-folds[[i]], ,drop=FALSE]
    dataset2.y <- dataset.y[-folds[[i]]]
    dataset2.testX <- dataset.x[folds[[i]], ,drop=FALSE]
    dataset2.testY <- dataset.y[folds[[i]]]
    dataset2.class_count <- table(dataset2.y) #classes count
    dataset2.class_levels <- unique(dataset2.y)
    dataset2.n_classes <- length(dataset2.class_levels)
    dataset2.n_samples <- nrow(dataset2.x)
    dataset2.n_genes <- ncol(dataset2.x)

    tuned = NULL
    if (TUNE == TRUE){
      # Tune variable is used when CV and when classify the independent test of each loop
      tuned <- tune.svm(x=dataset2.x, y=dataset2.y, gamma = 2^(-4:2), cost = 2^(1:4))
    }else{
      tuned$best.parameters[1] = 10^(-4)
      tuned$best.parameters[2] = 10
    }

    # rank the genes using SVM-RFE
    rank_data <- svmrfeFeatureRankingForMulticlass(dataset2.x, dataset2.y, tuned)
    rank = rank_data$featureRankedList

    # testing N genes of rank
    acc_by_n = rep(0,length(rank))
    for(nfeatures in 1:length(rank)){
      svmModel = svm(dataset2.x[, rank[1:nfeatures]], dataset2.y, cost = tuned$best.parameters[2], gamma = tuned$best.parameters[1], scale = T, type="C-classification", kernel="linear")
      pred <- predict(svmModel, dataset2.testX[, rank[1:nfeatures]])
      acc_by_n[nfeatures] = length(which(as.logical(pred == dataset2.testY)))/length(dataset2.testY)
    }

    # Select the smallest list of genes with max Accuracy
    max_acc = max(acc_by_n)
    best_n = length(acc_by_n)
    for (j in length(acc_by_n):1){
      if(acc_by_n[j] == max_acc){
        best_n = j
        break
      }
    }
    accs_kfold[i] = max_acc

    # Filter the selected genes (get it from rank)
    selected_genes <- rank[1:best_n]

    # If a gene is selected, than increment its global frequency
    for (j in 1:length(selected_genes)){
      global_genes_freq[selected_genes[j]] = global_genes_freq[selected_genes[j]] + 1
    }
  }
  # END - K-fold

  avg_acc_by_rep[rep] = mean(accs_kfold)
}})

avg_acc = mean(avg_acc_by_rep)
sd_acc = sd(avg_acc_by_rep)

result <- cbind(data.frame(avg_acc),data.frame(sd_acc))
colnames(result) <- c("avg acc","std acc")
write.csv(result, file = "./results/ranking_using_simple_crossvalidation/cv_result_avg_repetitions.csv")


# sort genes by freq.
rank_by_freq <- rev(sort(global_genes_freq, index.return = TRUE)$ix)

acc_by_n_freq_rank = list()
n_values = list()
#acc_by_n_freq_rank <- append(acc_by_n_freq_rank, 0)
#n_values <- append(n_values,1)
for(nfeatures in 1:length(rank_by_freq)){
  folds <- stratified_balanced_cv(y=dataset.y,nfolds=5)
  nfold <- length(folds)
  acc_kfold = list()
  for (i in 1:nfold){
        # Define ith train and test set
    dataset2.x <- dataset.x[-folds[[i]], ,drop=FALSE]
    dataset2.y <- dataset.y[-folds[[i]]]
    dataset2.testX <- dataset.x[folds[[i]], ,drop=FALSE]
    dataset2.testY <- dataset.y[folds[[i]]]
    dataset2.class_count <- table(dataset2.y) #classes count
    dataset2.class_levels <- unique(dataset2.y)
    dataset2.n_classes <- length(dataset2.class_levels)
    dataset2.n_samples <- nrow(dataset2.x)
    dataset2.n_genes <- ncol(dataset2.x)

    tuned = NULL
    if (TUNE == TRUE){
      # Tune variable is used when CV and when classify the independent test of each loop
      tuned <- tune.svm(x=dataset2.x, y=dataset2.y, gamma = 2^(-4:2), cost = 2^(1:4))
    }else{
      tuned$best.parameters[1] = 10^(-4)
      tuned$best.parameters[2] = 10
    }
    svmModel = svm(dataset2.x[, rank_by_freq[1:nfeatures]], dataset2.y, cost = tuned$best.parameters[2], gamma = tuned$best.parameters[1], scale = T, type="C-classification", kernel="linear")
    pred <- predict(svmModel, dataset2.testX[, rank_by_freq[1:nfeatures]])
    acc_kfold <- append(acc_kfold, length(which(as.logical(pred == dataset2.testY)))/length(dataset2.testY))
  }
  acc_by_n_freq_rank <- append(acc_by_n_freq_rank, mean(unlist(acc_kfold)))
  n_values <- append(n_values,nfeatures)
}

# write global rank and associated mean Acc by N value in rank
print(length(rank_by_freq))
print(length(acc_by_n_freq_rank))
result <- cbind(data.frame(1:length(rank_by_freq)),data.frame(global_genes_freq[rank_by_freq]),data.frame(colnames(dataset.x)[rank_by_freq]),data.frame(unlist(acc_by_n_freq_rank)))
colnames(result) <- c("rank","freq","gene","avg_cv_acc_after_svm-rfe")
#write.matrix(merged, file = "big_matrix_svm-rfe.csv", sep = ",")
write.csv(result, file = "./results/ranking_using_simple_crossvalidation/genes_freq_matrix_cv_whithout_repetition.csv")




# plot the cv result of rank by freq
pdf("./results/ranking_using_simple_crossvalidation/freq_rank_cv_svm-rfe.pdf")
plot(n_values,unlist(acc_by_n_freq_rank))
dev.off()


# generate a candidates Signatures selecting genes that appears at least at X% of experements
# freq >= 0.5 * length(repetitions) * k

acc_by_signature <- list()
signatures <- list()
sig = 1
signature_heuristic <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
for (pc in signature_heuristic){
  signature = list()

  minimum_freq = pc * n_repetitions * nfold

  for (geneidx in 1:length(global_genes_freq)){
    if (global_genes_freq[geneidx] >= minimum_freq){
      signature <- append(signature, geneidx)
    }
  }
  signature <- unlist(signature)
  # make a CV using the signature
  folds <- stratified_balanced_cv(y=dataset.y,nfolds=5)
  nfold <- length(folds)
  acc_kfold = list()
  for (i in 1:nfold){
        # Define ith train and test set
    dataset2.x <- dataset.x[-folds[[i]], ,drop=FALSE]
    dataset2.y <- dataset.y[-folds[[i]]]
    dataset2.testX <- dataset.x[folds[[i]], ,drop=FALSE]
    dataset2.testY <- dataset.y[folds[[i]]]
    dataset2.class_count <- table(dataset2.y) #classes count
    dataset2.class_levels <- unique(dataset2.y)
    dataset2.n_classes <- length(dataset2.class_levels)
    dataset2.n_samples <- nrow(dataset2.x)
    dataset2.n_genes <- ncol(dataset2.x)

    tuned = NULL
    if (TUNE == TRUE){
      # Tune variable is used when CV and when classify the independent test of each loop
      tuned <- tune.svm(x=dataset2.x, y=dataset2.y, gamma = 2^(-4:2), cost = 2^(1:4))
    }else{
      tuned$best.parameters[1] = 10^(-4)
      tuned$best.parameters[2] = 10
    }
    svmModel = svm(dataset2.x[, signature], dataset2.y, cost = tuned$best.parameters[2], gamma = tuned$best.parameters[1], scale = T, type="C-classification", kernel="linear")
    pred <- predict(svmModel, dataset2.testX[, signature])
    acc_kfold<-append(acc_kfold, length(which(as.logical(pred == dataset2.testY)))/length(dataset2.testY))
  }

  acc_by_signature[sig] <- mean(unlist(acc_kfold))
  signatures[sig] <- toString(colnames(dataset.x)[signature])
  sig = sig + 1
}

# write signatures and CV results in File
# ALERT: this CV results is highly biased since the frequencies of genes and the K-fold that test the signatures are computed using the same dataset
result <- cbind(data.frame(signature_heuristic), data.frame(unlist(acc_by_signature)), data.frame(unlist(signatures)))
colnames(result) <- c("presence_in_folds_and_repetitions","acc","signature")
#write.matrix(merged, file = "big_matrix_svm-rfe.csv", sep = ",")
write.csv(result, file = "./results/ranking_using_simple_crossvalidation/potential_signatures_cv_svm-rfe.csv")


#the end