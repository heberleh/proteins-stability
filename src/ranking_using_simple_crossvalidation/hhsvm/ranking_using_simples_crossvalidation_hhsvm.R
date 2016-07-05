###################### load R code #########################

# Author: Henry Heberle
# contact: henry at icmc usp br

#Installing required packages

list.of.packages <- c("combinat", "doSNOW","snow", "pamr", "rpart", "parallel", "MASS", "e1071")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

if(length(new.packages)) install.packages(new.packages,repos="http://brieger.esalq.usp.br/CRAN/")
source("hhsvm.r")
library("rpart")
library("MASS")
#library("class")
library("e1071")
#library("dismo")
library("pamr")
library("caret")
library("parallel")
require(foreach)
require(MASS)
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

signature <- function(object, cutoff=1e-10){
  coef <- object$beta
  features <- drop(rep(1, nrow(coef)) %*% abs(coef))
  sig <- features > cutoff
  return(sig)
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
input_file_name <- "../../../dataset/current/train.txt"
db <- read.table(input_file_name, header=TRUE, sep="\t")

# HHSVM parameters
lam2 = 10
delta = 3

class_delta <- 1

n_repetitions <- 3

TUNE <- FALSE                # PRE-TUNE PARAMETERS OF CLASSIFICATION MODELS?
#=========================== =============================== ==============================
#=========================== =============================== ==============================
#=========================== =============================== ==============================



#=========================== =============================== ==============================
#=========================== ===== PRE-PROCESSING DATA ===== ==============================
db2 <-t(db)
dataset.x <- as.matrix(db2[3:nrow(db2),2:(ncol(db2))])
class(dataset.x) <- "numeric"
colnames(dataset.x)<-db2[2,2:ncol(db2)]

# first line of original matrix is already col/rownames
#print(db2[3:nrow(db2),1]) -> classes
#print(db2[3:nrow(db2),2]) -> values
dataset.y <- as.factor(db2[3:nrow(db2),1])
new_levels = vector()
levels_ids = list()
id = 0
for (level in levels(dataset.y)){
    levels_ids[level] <- id
    id = id + 1
}
for (c in dataset.y){
    new_levels <- append(new_levels,levels_ids[c])
}
dataset.y = as.factor(unlist(new_levels))
#=========================== =============================== ==============================
#=========================== =============================== ==============================




#=========================== =============================== ==============================
#=========================== ============ START ============ ==============================

folds <- stratified_balanced_cv(y=dataset.y,nfolds=5)
nfold <- length(folds)
cat("Number of folds", nfold,"\n")
#writeLines(paste("Number of folds: ",nfold), fileConn)

global_genes_freq <- rep(0, ncol(dataset.x))
avg_acc_by_rep <- rep(0,n_repetitions)
for (rep in 1:n_repetitions){
  folds <- stratified_balanced_cv(y=dataset.y,nfolds=5)
  nfold <- length(folds)


  # START K-Fold
  accs_kfold = rep(0,nfold)
  for (i in 1:nfold){
    # Define ith train and test set
    print(i)
    print(i)

    i = 1
    dataset2.x <- dataset.x[-folds[[i]], ,drop=FALSE]
    dataset2.y <- dataset.y[-folds[[i]]]
    dataset2.testX <- dataset.x[folds[[i]], ,drop=FALSE]
    dataset2.testY <- as.factor(dataset.y[folds[[i]]])
    dataset2.class_count <- table(dataset2.y) #classes count
    dataset2.class_levels <- unique(dataset2.y)
    dataset2.n_classes <- length(dataset2.class_levels)
    dataset2.n_samples <- nrow(dataset2.x)
    dataset2.n_genes <- ncol(dataset2.x)


# FALHOU... NÃO FUNCIONA PARA MULTICLASS


    y <- as.numeric(as.vector(cbind(rep(-1,12),rep(1,12))))
    y <- as.numeric(as.vector(dataset2.y))



    train <- DrHSVM(dataset2.x,y, lam2, delta=delta)
    pred <- DrHSVM.predict(train, dataset2.testX, dataset2.testY)    # training error

    accs_kfold[i] <- 1 - pred$error

    signature = signature(train)
    # If a gene is selected, than increment its global frequency
    for (j in 1:length(signature)):
        if (signature[j] ==  TRUE){
            global_genes_freq[j] +=1
        }
    }
  }
  # END - K-fold

  avg_acc_by_rep[rep] = mean(accs_kfold)
}

avg_acc = mean(avg_acc_by_rep)
sd_acc = sd(avg_acc_by_rep)

result <- cbind(data.frame(avg_acc),data.frame(sd_acc))
colnames(result) <- c("avg acc","std acc")
write.csv(result, file = "./results/ranking_using_simple_crossvalidation/nsc/cv_result_avg_repetitions_nsc.csv")


# sort genes by freq.
rank_by_freq <- rev(sort(global_genes_freq, index.return = TRUE)$ix)

acc_by_n_freq_rank = rep(-1,length(rank_by_freq))
n_values = list()
#acc_by_n_freq_rank <- append(acc_by_n_freq_rank, 0)
#n_values <- append(n_values,1)
for(nfeatures in 2:length(rank_by_freq)){
  folds <- stratified_balanced_cv(y=dataset.y,nfolds=5)
  nfold <- length(folds)
  acc_kfold = list()
  for (i in 1:nfold){
        # Define ith train and test set
        i=1
    dataset2.x <- dataset.x[, -folds[[i]],drop=FALSE]
    dataset2.y <- dataset.y[-folds[[i]]]
    dataset2.testX <- dataset.x[, folds[[i]],drop=FALSE]
    dataset2.testY <- dataset.y[folds[[i]]]
    dataset2.class_count <- table(dataset2.y) #classes count
    dataset2.class_levels <- unique(dataset2.y)
    dataset2.n_classes <- length(dataset2.class_levels)
    dataset2.n_samples <- nrow(dataset2.x)
    dataset2.n_genes <- ncol(dataset2.x)

    nsc_data <- list(x=dataset2.x[rank_by_freq[1:nfeatures], ], y=factor(dataset2.y),genenames=dataset.genesnames[rank_by_freq[1:nfeatures]],geneids=rank_by_freq[1:nfeatures])

    #  train NSC
    nsc_train <- pamr.train(nsc_data)

    #teste
    pred = pamr.predict(nsc_train, dataset2.testX[rank_by_freq[1:nfeatures], ], 0.000, type=c("class")) #consider all selected proteins (n features of this loop)
    acc = length(which(as.logical(pred == dataset2.testY)))/length(dataset2.testY)
    acc_kfold <- append(acc_kfold, acc)
  }
  acc_by_n_freq_rank[nfeatures] <-  mean(unlist(acc_kfold))
  n_values <- append(n_values,nfeatures)
}

# write global rank and associated mean Acc by N value in rank
print(length(rank_by_freq))
print(length(acc_by_n_freq_rank))
result <- cbind(data.frame(1:length(rank_by_freq)),data.frame(global_genes_freq[rank_by_freq]),data.frame(colnames(dataset.x)[rank_by_freq]),data.frame(unlist(acc_by_n_freq_rank)))
colnames(result) <- c("rank","freq","gene","avg_cv_acc_after_nsc")
#write.matrix(merged, file = "big_matrix_svm-rfe.csv", sep = ",")
write.csv(result, file = "./results/ranking_using_simple_crossvalidation/nsc/genes_freq_matrix_cv_whithout_repetition.csv")




# plot the cv result of rank by freq
pdf("./results/ranking_using_simple_crossvalidation/nsc/freq_rank_cv_nsc.pdf")
plot(1:length(rank_by_freq),unlist(acc_by_n_freq_rank))
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

  if (length(signature)>1){
    # make a CV using the signature
    folds <- stratified_balanced_cv(y=dataset.y,nfolds=5)
    nfold <- length(folds)
    acc_kfold = list()
    for (i in 1:nfold){
          # Define ith train and test set
      dataset2.x <- dataset.x[, -folds[[i]],drop=FALSE]
      dataset2.y <- dataset.y[-folds[[i]]]
      dataset2.testX <- dataset.x[, folds[[i]],drop=FALSE]
      dataset2.testY <- dataset.y[folds[[i]]]
      dataset2.class_count <- table(dataset2.y) #classes count
      dataset2.class_levels <- unique(dataset2.y)
      dataset2.n_classes <- length(dataset2.class_levels)
      dataset2.n_samples <- nrow(dataset2.x)
      dataset2.n_genes <- ncol(dataset2.x)

      nsc_data <- list(x=dataset2.x[signature, ], y=dataset2.y,genenames=dataset.genesnames[signature],geneids=signature)

      #  train NSC
      nsc_train <- pamr.train(nsc_data)

      #teste
      pred = pamr.predict(nsc_train, dataset2.testX[signature, ], 0.000, type=c("class")) #consider all selected proteins (n features of this loop)
      acc = length(which(as.logical(pred == dataset2.testY)))/length(dataset2.testY)
      acc_kfold <- append(acc_kfold, acc)
      acc_by_signature[sig] <- mean(unlist(acc_kfold))


    }
  }else{
    acc_by_signature[sig] <- -1
  }
  signatures[sig] <- toString(rownames(dataset.x)[signature])
  sig = sig + 1
}

# write signatures and CV results in File
# ALERT: this CV results is highly biased since the frequencies of genes and the K-fold that test the signatures are computed using the same dataset
result <- cbind(data.frame(signature_heuristic), data.frame(unlist(acc_by_signature)), data.frame(unlist(signatures)))
colnames(result) <- c("presence_in_folds_and_repetitions","acc","signature")
#write.matrix(merged, file = "big_matrix_svm-rfe.csv", sep = ",")
write.csv(result, file = "./results/ranking_using_simple_crossvalidation/nsc/potential_signatures_cv_nsc.csv")


# save the rank considering the entire dataset
nsc_data <- list(x=dataset.x, y=factor(dataset.y), genenames=dataset.genesnames, geneids=dataset.genesnames)
nsc_train <- pamr.train(nsc_data)
nsc_scales <- pamr.adaptthresh(nsc_train)

nsc_train <- pamr.train(nsc_data, threshold.scale=nsc_scales,n.threshold=thresholds_number, scale.sd=TRUE)

complete_genes_list = pamr.listgenes(nsc_train, nsc_data, 0, genenames=TRUE)
write.csv(complete_genes_list,"./results/ranking_using_simple_crossvalidation/nsc/rank_by_nsc_using_the_full_dataset.csv")
#the end

################### DrHSVM Method  ###########################


x1 = mvrnorm(n=bigN,mu1,sigma)
x2 = mvrnorm(n=bigN+1,mu2,sigma)
y1 = rep(1,bigN)
y2 = rep(-1,bigN+1)
trX = rbind(x1,x2)
trY = c(y1,y2)

N = 2*bigN+1  # size of the training data set

g <- DrHSVM(trX,trY,lam2,delta=delta)
pre <- DrHSVM.predict(g,trX,trY)    # training error


matplot(s, cbind(g$beta,pre$err), type="n", cex.lab = 1.5,
      xlab=expression(paste("||",beta,"||",scriptscriptstyle(1))), ylab=expression(beta))
for(i in 1:q)
  lines(s, g$beta[,i], col=i+1, lty=1)
for(i in (q+1):p)
  lines(s, g$beta[,i], col=i+1, lty=2)

signature = signature(g)
selected_genes = dataset.genes[]

for i in 1:length(signature):
    if (signature[i] ==  TRUE){
        freq[i] +=1
    }
}