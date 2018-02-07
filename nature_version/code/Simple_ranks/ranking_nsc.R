
# Author: Henry Heberle
# contact: henry at icmc usp br

#Installing required packages

list.of.packages <- c("combinat", "doSNOW","caret","snow", "pamr", "rpart", "parallel", "MASS", "e1071")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

if(length(new.packages)) install.packages(new.packages,repos="http://brieger.esalq.usp.br/CRAN/")

library("rpart")
library("MASS")
#library("class")
library("e1071")
#library("dismo")
library("pamr")
library("caret")
library("parallel")
require(foreach)
require(doSNOW)
require(combinat)


#========================== =============================== ==============================
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



getLowerErrorMaxThresholdMaxProbability <- function(errors, thresholds, probabilities){
  min_error = min(errors)
  higher_threshold = min(thresholds) 

  index_error = which(errors == min_error, arr.ind=TRUE)
  
  selected_probs = probabilities[index_error]

  max_probability = max(selected_probs)

  index_prob = which(probabilities == max_probability, arr.ind = TRUE)

  index_prob = intersect(index_error,index_prob)

  selected_thres = thresholds[index_prob]
 
  max_threshold =  max(selected_thres)  

  result = list(higher_threshold = max_threshold, error = min_error, probability = max_probability)
  
  return(result)
}

#=========================== =============================== ==============================
#=========================== =============================== ==============================



input_file_name <-  "./dataset/independent_train.txt"
db <- read.table(input_file_name, header=FALSE, sep="\t")
#db <- t(db[1:100,])

class_delta <- 1
ttest <- TRUE

dataset.x <- as.matrix(db[3:(nrow(db)),2:ncol(db)])
class(dataset.x) <- "numeric"
rownames(dataset.x)<-db[3:nrow(db),1]
thresholds_number <- nrow(dataset.x)
dataset.y <- as.factor(as.matrix(db[2,2:ncol(db)]))
cat(dataset.y)
dataset.genesnames <- rownames(dataset.x)

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

# testing the threshold found Double-Cross with INDEPENDENT test
nsc_data<- list(x=dataset.x, y=factor(dataset.y), genenames=dataset.genesnames, geneids=1:length(dataset.genesnames))

# select genes using NSC
nsc_train <- pamr.train(nsc_data)
nsc_scales <- pamr.adaptthresh(nsc_train)
nsc_train <- pamr.train(nsc_data, threshold.scale=nsc_scales, n.threshold=thresholds_number)
nsc_cv <- pamr.cv(nsc_train,nsc_data)

nsc_result <- pamr.listgenes(nsc_train, nsc_data, 0.0000000, fitcv=nsc_cv)

#final genes according to Double-Cross
ranked_genes <- as.numeric(as.vector(nsc_result[,1]))

result <- cbind(data.frame(ranked_genes),data.frame(rownames(dataset.x)[ranked_genes],nsc_result))
colnames(result) <- c("index","rank?","name")
write.csv(result, file = "./results/double_cross_validation/nsc/simple_rank.csv")




