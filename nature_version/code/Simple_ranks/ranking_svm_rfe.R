
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

leave_one_out_cv <- function(y, nfolds){
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
        svmModel = svm(x[, survivingFeaturesIndexes], y, cost = 10, cachesize=500,  scale=F, type="C-classification", kernel="radial" ) #linear, polynomial, radial basis, sigmoid

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
        svmModel = svm(x[, survivingFeaturesIndexes], y, tuned$best.parameters[2], gamma = tuned$best.parameters[1], scale=T, type="C-classification", kernel="linear" ) #linear, polynomial, radial basis, sigmoid

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
input_file_name <- "./dataset/independent_train.txt"
db <- read.table(input_file_name, header=FALSE, sep="\t")
ttest <- TRUE

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

kindexes = list()
probs = c()
if (ttest){
  #Kruskal filter
  for (i in 1:nrow(dataset.x)){
      classes = unique(dataset.y)

      index_x = which(dataset.y == classes[1], arr.ind=TRUE)
      values_x = dataset.x[i,index_x]

      index_y = which(dataset.y == classes[2], arr.ind=TRUE)
      values_y = dataset.x[i,index_y]
      cat(values_x)
      cat("\n")
      cat(values_y)
      cat("\n")
      cat(index_x)
      cat("\n")
      cat(index_y)
      cat("\n")

      ktest <- t.test(values_x, values_y, var.equal=TRUE, paired=FALSE)
      probs <- c(probs,as.numeric(ktest$p.value))
      cat(as.numeric(ktest[3][1]))
      cat("\n")
      if (as.numeric(ktest[3][1]) < 0.10){
        kindexes <- append(kindexes,i)
      }
  }


result <- cbind(data.frame(1:length(rownames(dataset.x))),data.frame(rownames(dataset.x)),probs)
colnames(result) <- c("index","name","p-value")
write.csv(result, file = "./results/double_cross_validation/ttest.csv")

  kindexes <- as.numeric(kindexes)

  dataset.x <- dataset.x[kindexes,,drop=FALSE]  
  thresholds_number <- nrow(dataset.x)
  dataset.genesnames <- rownames(dataset.x)

}


#=========================== =============================== ==============================
#=========================== =============================== ==============================

# rank the genes using SVM-RFE
rank_data <- svmrfeFeatureRanking(t(dataset.x), dataset.y)
rank = rank_data

#final genes according to Double-Cross
dc_selected_genes <- as.numeric(as.vector(rank))

result <- cbind(data.frame(dc_selected_genes),data.frame(1:length(dc_selected_genes)),data.frame(rownames(dataset.x)[dc_selected_genes]))
colnames(result) <- c("index","rank?","name")
write.csv(result, file = "./results/double_cross_validation/svm/simple_rank.csv")


