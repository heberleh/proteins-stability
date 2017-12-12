
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
input_file_name <- "./dataset/train_6_samples_independent.txt"
db <- read.table(input_file_name, header=FALSE, sep="\t")

input_file_name_test <- "./dataset/test_6_samples_independent.txt"
db_test <- read.table(input_file_name, header=FALSE, sep="\t")

class_delta <- 1
n_repetitions <- 1

TUNE <- FALSE                # PRE-TUNE PARAMETERS OF CLASSIFICATION MODELS?
#=========================== =============================== ==============================
#=========================== =============================== ==============================
#=========================== =============================== ==============================


#=========================== =============================== ==============================
#=========================== ===== PRE-PROCESSING DATA ===== ==============================
dataset.x <- as.matrix(db[3:(nrow(db)),2:ncol(db)])
class(dataset.x) <- "numeric"
rownames(dataset.x)<-db[3:nrow(db),1]
thresholds_number <- nrow(dataset.x)
dataset.y <- as.factor(as.matrix(db[2,2:ncol(db)]))
cat("labels",dataset.y,"\n")
cat("colnames", colnames(dataset.x),"\n")
dataset.genesnames <- rownames(dataset.x)
cat("vet pos 1", colnames(dataset.x)[1])

dataset_test.x <- as.matrix(db_test[3:(nrow(db_test)),2:ncol(db_test)])
class(dataset_test.x) <- "numeric"
rownames(dataset_test.x)<-db_test[3:nrow(db_test),1]
dataset_test.y <- as.factor(as.matrix(db_test[2,2:ncol(db_test)]))
dataset_test.genesnames <- rownames(dataset_test.x)

#=========================== =============================== ==============================
#=========================== =============================== ==============================





#=========================== =============================== ==============================
#=========================== ============ START ============ ==============================

outer_k = 4
inner_k = 3

cat("Number of genes",nrow(dataset.x))

global_genes_freq_inner <- rep(0, nrow(dataset.x))
global_genes_freq_outer <- rep(0, nrow(dataset.x))
avg_acc_by_rep <- rep(0,n_repetitions)
std_acc_by_rep <- rep(0,n_repetitions)

accs_kfold_outer = list()
best_n_outer = 0
max_acc_outer = -999999

for (rep in 1:n_repetitions){
  folds <- stratified_balanced_cv(y=dataset.y, nfolds=outer_k)
  nfold_outer <- length(folds)
  cat("Number of outer folds", nfold_outer,"\n")

   # START K-Fold   [outer-loop]
  for (i_outer in 1:nfold_outer){
    # Define ith train and test set
    dataset2.x <- dataset.x[, -folds[[i_outer]],drop=FALSE]
    dataset2.y <- dataset.y[-folds[[i_outer]]]
    dataset2.testX <- dataset.x[, folds[[i_outer]],drop=FALSE]
    dataset2.testY <- dataset.y[folds[[i_outer]]]
    dataset2.class_count <- table(dataset2.y) #classes count
    dataset2.class_levels <- unique(dataset2.y)
    dataset2.n_classes <- length(dataset2.class_levels)
    dataset2.n_samples <- nrow(dataset2.x)
    dataset2.n_genes <- ncol(dataset2.x)

    cat("colnames dataset2 size ", length(colnames(dataset2.x)),": ", colnames(dataset2.x),"\n")
    cat("labels dataset2 size ", length(dataset2.y),": ", dataset2.y,"\n")

    folds_inner <- stratified_balanced_cv(y=dataset2.y, nfolds=inner_k)
    nfold_inner <- length(folds_inner)
    cat("Number of inner folds", nfold_inner,"\n")

    # START K-Fold   [inner-loop]    
    selected_genes_inner = NULL
    max_acc_inner = 0.0
    best_n_inner = 0

    for (i_inner in 1:nfold_inner){
      # Define ith train and test set
      cat("Fold number", i_inner,"\n")
      cat("Fold: ", folds_inner[[i_inner]],"\n")
      dataset3.x <- dataset2.x[, -folds_inner[[i_inner]], drop=FALSE]
      dataset3.y <- dataset2.y[-folds_inner[[i_inner]]]
      dataset3.testX <- dataset2.x[, folds_inner[[i_inner]], drop=FALSE]
      dataset3.testY <- dataset2.y[folds_inner[[i_inner]]]
      dataset3.class_count <- table(dataset3.y) #classes count
      dataset3.class_levels <- unique(dataset3.y)
      dataset3.n_classes <- length(dataset3.class_levels)
      dataset3.n_samples <- nrow(dataset3.x)
      dataset3.n_genes <- ncol(dataset3.x)
      
      cat("colnames dataset3 size ", length(colnames(dataset3.x)),": ", colnames(dataset3.x),"\n")
      cat("labels dataset3 size ", length(dataset3.y),": ", dataset3.y,"\n")

      cat("samples per class **", dataset3.class_count, "\n")

      tuned = NULL
      if (TUNE == TRUE){
        # Tune variable is used when CV and when classify the independent test of each loop
        tuned <- tune.svm(x=t(dataset3.x), y=dataset3.y, gamma = 2^(-4:2), cost = 2^(1:4))
      }else{
        tuned$best.parameters[1] = 10^(-4)
        tuned$best.parameters[2] = 10
      }

      # rank the genes using SVM-RFE
      cat("row names", length(rownames(dataset3.x)), "\n")
      cat("col names", length(colnames(dataset3.x)), "\n")
      cat("classes", dataset3.y, "\n")
                  
      rank_data <- svmrfeFeatureRankingForMulticlass(t(dataset3.x), dataset3.y, tuned)

      rank = rank_data$featureRankedList
      
      cat ("computed rank\n")

      # testing N genes of rank
      acc_by_n = rep(0,length(rank))
      if (length(rank)>1){      
        for(nfeatures in 2:length(rank)){
          svmModel = svm(t(dataset3.x[rank[1:nfeatures], ]), dataset3.y, cost = tuned$best.parameters[2], gamma = tuned$best.parameters[1], scale = T, type="C-classification", kernel="linear")
          pred <- predict(svmModel, t(dataset3.testX[rank[1:nfeatures], ]))
          acc_by_n[nfeatures] = length(which(as.logical(pred == dataset3.testY)))/length(dataset3.testY)
        }
      }else{
        acc_by_n[0] = 0
        acc_by_n[1] = 0
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
      if (max_acc_inner < max_acc){
        max_acc_inner = max_acc
        best_n_inner = best_n
      }

      # Filter the selected genes (get it from rank)
      selected_genes <- rank[1:best_n]

      # If a gene is selected, than increment its global frequency
      for (j in 1:length(selected_genes)){
        global_genes_freq_inner[selected_genes[j]] = global_genes_freq_inner[selected_genes[j]] + 1
      }
    
    } # END K-fold inner
      cat("\nEnd of one inner K-fold\n")

      tuned = NULL
      if (TUNE == TRUE){
        # Tune variable is used when CV and when classify the independent test of each loop
        tuned <- tune.svm(x=t(dataset2.x), y=dataset2.y, gamma = 2^(-4:2), cost = 2^(1:4))
      }else{
        tuned$best.parameters[1] = 10^(-4)
        tuned$best.parameters[2] = 10
      }

      # rank the genes using SVM-RFE
      rank_data <- svmrfeFeatureRankingForMulticlass(t(dataset2.x), dataset2.y, tuned)
      rank = rank_data$featureRankedList
      
      # testing min N of max acc selected in inner Loop genes of rank
      svmModel = svm(t(dataset2.x[rank[1:best_n_inner], ]), dataset2.y, cost = tuned$best.parameters[2], gamma = tuned$best.parameters[1], scale = T, type="C-classification", kernel="linear")
      pred <- predict(svmModel, t(dataset2.testX[rank[1:best_n_inner], ]))
      acc_by_n = length(which(as.logical(pred == dataset2.testY)))/length(dataset2.testY)

      accs_kfold_outer <- append(accs_kfold_outer, acc_by_n)
    
      # Select the smallest list of genes with max Accuracy
      if (max_acc_outer < acc_by_n){
        max_acc_outer = acc_by_n
        best_n_outer = best_n_inner
      }

      # Filter the selected genes (get it from rank)
      selected_genes <- rank[1:best_n_inner]

      # If a gene is selected, than increment its global frequency
      for (j in 1:length(selected_genes)){
        global_genes_freq_outer[selected_genes[j]] = global_genes_freq_outer[selected_genes[j]] + 1
      }

  }
  # END - K-fold outer
  cat("\nEnd of outer K-fold\n")


  accs_kfold_outer =  as.numeric(accs_kfold_outer)
  
  for (j in 1:length(accs_kfold_outer)){
    cat("acc: ",accs_kfold_outer[j],"\n")
  }

  avg_acc_by_rep[rep] = mean(accs_kfold_outer)
  cat("\n# Accuracy for rep.",rep," is", avg_acc_by_rep[rep],"\n")

  std_acc_by_rep[rep] = sd(accs_kfold_outer)
  cat("\n# Std for rep.",rep," is", std_acc_by_rep[rep],"\n")

}# end - repetitions
cat("\nEnd of repetitions\n")




avg_acc = avg_acc_by_rep[1]
sd_acc = std_acc_by_rep[1]

cat("Avg Acc DCV", avg_acc,"\n")
cat("Std Acc DCV", sd_acc,"\n")

result <- cbind(data.frame(avg_acc),data.frame(sd_acc))
colnames(result) <- c("avg acc","std acc")
write.csv(result, file = "./results/double_cross_validation/svm/avg_std_acc.csv")

# sort genes by freq.
rank_by_freq <- rev(sort(global_genes_freq_inner, index.return = TRUE)$ix)

cat("rank  by frequency? ",rank_by_freq,"\n")

acc_by_n_freq_rank = list()
n_values = list()
#acc_by_n_freq_rank <- append(acc_by_n_freq_rank, 0)
#n_values <- append(n_values,1)
acc_by_n_freq_rank <- append(acc_by_n_freq_rank, 0.0)
for(nfeatures in 2:length(rank_by_freq)){
  folds <- stratified_balanced_cv(y=dataset.y,nfolds=4)
  nfold <- length(folds)
  acc_kfold = list()
  for (idx in 1:nfold){
    # Define ith train and test set
    dataset2.x <- dataset.x[, -folds[[idx]],drop=FALSE]
    dataset2.y <- dataset.y[-folds[[idx]]]
    dataset2.testX <- dataset.x[, folds[[idx]],drop=FALSE]
    dataset2.testY <- dataset.y[folds[[idx]]]
    dataset2.class_count <- table(dataset2.y) #classes count
    dataset2.class_levels <- unique(dataset2.y)
    dataset2.n_classes <- length(dataset2.class_levels)
    dataset2.n_samples <- nrow(dataset2.x)
    dataset2.n_genes <- ncol(dataset2.x)

    tuned = NULL
    if (TUNE == TRUE){
      # Tune variable is used when CV and when classify the independent test of each loop
      tuned <- tune.svm(x=t(dataset2.x), y=dataset2.y, gamma = 2^(-4:2), cost = 2^(1:4))
    }else{
      tuned$best.parameters[1] = 10^(-4)
      tuned$best.parameters[2] = 10
    }
    selected_genes_idx = rank_by_freq[1:nfeatures]
    svmModel = svm(t(dataset2.x[selected_genes_idx, ]), dataset2.y, cost = tuned$best.parameters[2], gamma = tuned$best.parameters[1], scale = T, type="C-classification", kernel="linear")

    pred <- predict(svmModel, t(dataset2.testX[selected_genes_idx, ]))

    acc_kfold <- append(acc_kfold, length(which(as.logical(pred == dataset2.testY)))/length(dataset2.testY))

  }
  acc_by_n_freq_rank <- append(acc_by_n_freq_rank, mean(unlist(acc_kfold)))
  n_values <- append(n_values,nfeatures)
}

# write global rank and associated mean Acc by N value in rank
print(length(rank_by_freq))
print(length(acc_by_n_freq_rank))
result <- cbind(data.frame(rank_by_freq),data.frame(1:length(rank_by_freq)),data.frame(global_genes_freq_inner[rank_by_freq]),data.frame(rownames(dataset.x)[rank_by_freq]),data.frame(unlist(acc_by_n_freq_rank)))
colnames(result) <- c("index","rank","freq","gene","avg_cv_acc_after_nsc")
#write.matrix(merged, file = "big_matrix_svm-rfe.csv", sep = ",")
write.csv(result, file = "./results/double_cross_validation/svm/genes_freq_matrix_inner_CV.csv")


result <- cbind(data.frame(1:length(global_genes_freq_outer)),data.frame(global_genes_freq_outer),data.frame(rownames(dataset.x)))
colnames(result) <- c("index","freq","gene")
write.csv(result, file = "./results/double_cross_validation/svm/genes_freq_matrix_outer.csv")



# plot the cv result of rank by freq
pdf("./results/double_cross_validation/svm/freq_rank_cv_nsc.pdf")
plot(1:length(rank_by_freq),unlist(acc_by_n_freq_rank))
dev.off()



# INDEPENDENT TEST - MIN N FROM DCV -> 22 SAMPLES
tuned = NULL
if (TUNE == TRUE){
  # Tune variable is used when CV and when classify the independent test of each loop
  tuned <- tune.svm(x=t(dataset.x), y=dataset.y, gamma = 2^(-4:2), cost = 2^(1:4))
}else{
  tuned$best.parameters[1] = 10^(-4)
  tuned$best.parameters[2] = 10
}

# rank the genes using SVM-RFE
rank_data <- svmrfeFeatureRankingForMulticlass(t(dataset.x), dataset.y, tuned)
rank = rank_data$featureRankedList

# testing min N of max acc selected in inner Loop genes of rank
svmModel = svm(t(dataset.x[rank[1:best_n_outer], ]), dataset.y, cost = tuned$best.parameters[2], gamma = tuned$best.parameters[1], scale = T, type="C-classification", kernel="linear")

pred <- predict(svmModel, t(dataset_test.x[rank[1:best_n_outer], ]))

acc = length(which(as.logical(pred == dataset_test.y)))/length(dataset_test.y)

result <- cbind(data.frame(c("independent acc"),data.frame(c(acc))))

cat("independent acc: ", acc)

write.csv(result, file = "./results/double_cross_validation/svm/independent_test_threshold_from_DCV.csv")



#final genes according to Double-Cross
dc_selected_genes <- as.numeric(as.vector(rank[1:best_n_outer]))

result <- cbind(data.frame(dc_selected_genes),data.frame(1:length(dc_selected_genes)),data.frame(rownames(dataset.x)[dc_selected_genes]))

colnames(result) <- c("index","rank?","name")
#write.matrix(merged, file = "big_matrix_svm-rfe.csv", sep = ",")
write.csv(result, file = "./results/double_cross_validation/svm/genes_DCV.csv")






#========================================================================================================

# # generate a candidates Signatures selecting genes that appears at least at X% of experements
# # freq >= 0.5 * length(repetitions) * k

# acc_by_signature <- list()
# signatures <- list()
# sig = 1
# signature_heuristic <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
# for (pc in signature_heuristic){
#   signature = list()

#   minimum_freq = pc * n_repetitions * nfold

#   for (geneidx in 1:length(global_genes_freq)){
#     if (global_genes_freq[geneidx] >= minimum_freq){
#       signature <- append(signature, geneidx)
#     }
#   }
#   signature <- unlist(signature)
#   # make a CV using the signature
#   folds <- stratified_balanced_cv(y=dataset.y,nfolds=5)
#   nfold <- length(folds)
#   acc_kfold = list()
#   for (i in 1:nfold){
#         # Define ith train and test set
#     dataset2.x <- dataset.x[-folds[[i]], ,drop=FALSE]
#     dataset2.y <- dataset.y[-folds[[i]]]
#     dataset2.testX <- dataset.x[folds[[i]], ,drop=FALSE]
#     dataset2.testY <- dataset.y[folds[[i]]]
#     dataset2.class_count <- table(dataset2.y) #classes count
#     dataset2.class_levels <- unique(dataset2.y)
#     dataset2.n_classes <- length(dataset2.class_levels)
#     dataset2.n_samples <- nrow(dataset2.x)
#     dataset2.n_genes <- ncol(dataset2.x)

#     tuned = NULL
#     if (TUNE == TRUE){
#       # Tune variable is used when CV and when classify the independent test of each loop
#       tuned <- tune.svm(x=dataset2.x, y=dataset2.y, gamma = 2^(-4:2), cost = 2^(1:4))
#     }else{
#       tuned$best.parameters[1] = 10^(-4)
#       tuned$best.parameters[2] = 10
#     }
#     svmModel = svm(dataset2.x[, signature], dataset2.y, cost = tuned$best.parameters[2], gamma = tuned$best.parameters[1], scale = T, type="C-classification", kernel="linear")
#     pred <- predict(svmModel, dataset2.testX[, signature])
#     acc_kfold<-append(acc_kfold, length(which(as.logical(pred == dataset2.testY)))/length(dataset2.testY))
#   }

#   acc_by_signature[sig] <- mean(unlist(acc_kfold))
#   signatures[sig] <- toString(colnames(dataset.x)[signature])
#   sig = sig + 1
# }

# # write signatures and CV results in File
# # ALERT: this CV results is highly biased since the frequencies of genes and the K-fold that test the signatures are computed using the same dataset
# result <- cbind(data.frame(signature_heuristic), data.frame(unlist(acc_by_signature)), data.frame(unlist(signatures)))
# colnames(result) <- c("presence_in_folds_and_repetitions","acc","signature")
# #write.matrix(merged, file = "big_matrix_svm-rfe.csv", sep = ",")
# write.csv(result, file = "./results/ranking_using_simple_crossvalidation/svm-rfe/potential_signatures_cv_svm-rfe.csv")


# tuned = NULL
# if (TUNE == TRUE){
#   # Tune variable is used when CV and when classify the independent test of each loop
#   tuned <- tune.svm(x=dataset.x, y=dataset.y, gamma = 2^(-4:2), cost = 2^(1:4))
# }else{
#   tuned$best.parameters[1] = 10^(-4)
#   tuned$best.parameters[2] = 10
# }

# # save the rank considering the entire dataset
# rank_data <- svmrfeFeatureRankingForMulticlass(dataset.x, dataset.y, tuned)
# rank = rank_data$featureRankedList
# rank_matrix <- cbind(data.frame(1:length(rank)), data.frame(colnames(dataset.x)[rank]))
# colnames(rank_matrix) <- c("rank","protein")
# write.csv(rank_matrix, file = "./results/ranking_using_simple_crossvalidation/svm-rfe/rank_by_svm-rfe_using_the_full_dataset.csv")
# #the end