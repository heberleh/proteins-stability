
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
#db <- t(db[1:100,])

input_file_name_test <- "./dataset/test_6_samples_independent.txt"
db_test <- read.table(input_file_name_test, header=FALSE, sep="\t")

class_delta <- 1
kruskal_test <- FALSE# TRUE

TUNE <- FALSE                # PRE-TUNE PARAMETERS OF CLASSIFICATION MODELS?
#=========================== =============================== ==============================
#=========================== =============================== ==============================
#=========================== =============================== ==============================


#=========================== =============================== ==============================
#=========================== ===== PRE-PROCESSING DATA ===== ==============================
#lines are genes
#columns are samples


dataset.x <- as.matrix(db[3:(nrow(db)),2:ncol(db)])
class(dataset.x) <- "numeric"
rownames(dataset.x)<-db[3:nrow(db),1]
thresholds_number <- nrow(dataset.x)
dataset.y <- as.factor(as.matrix(db[2,2:ncol(db)]))
cat(dataset.y)
dataset.genesnames <- rownames(dataset.x)

dataset_test.x <- as.matrix(db_test[3:(nrow(db_test)),2:ncol(db_test)])
class(dataset_test.x) <- "numeric"
rownames(dataset_test.x)<-db_test[3:nrow(db_test),1]
dataset_test.y <- as.factor(as.matrix(db_test[2,2:ncol(db_test)]))
dataset_test.genesnames <- rownames(dataset_test.x)


kindexes = list()
if (kruskal_test){
  #Kruskal filter
  for (i in 1:nrow(dataset.x)){
      ktest <- kruskal.test(x=dataset.x[i,], g=dataset.y)
      if (as.numeric(ktest[3][1]) < 0.5){
        kindexes <- append(kindexes,i)
      }
  }
  kindexes <- as.numeric(kindexes)

  dataset.x <- dataset.x[kindexes,,drop=FALSE]  
  thresholds_number <- nrow(dataset.x)
  dataset.genesnames <- rownames(dataset.x)


  dataset_test.x <- dataset_test.x[kindexes,,drop=FALSE]  
  dataset_test.genesnames <- rownames(dataset_test.x)
}



#=========================== =============================== ==============================
#=========================== =============================== ==============================




#=========================== =============================== ==============================
#=========================== ============ START ============ ==============================

outer_k = 22
inner_k = 7
n_repetitions <- 10

cat("Number of genes",nrow(dataset.x))

global_genes_freq_inner <- rep(0, nrow(dataset.x))
global_genes_freq_outer <- rep(0, nrow(dataset.x))

accs_kfold_outer = list()
accs_kfold_outer_all_genes = list()
max_thres_outer = 0
max_acc_outer = 0
max_prob_outer = 0


folds = NULL
if (outer_k == length(dataset.y)){
  folds <- leave_one_out_cv(dataset.y,outer_k)
}else{
  folds <- stratified_balanced_cv(y=dataset.y, nfolds=outer_k)
}  
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


  folds_inner = NULL
  if (inner_k == length(dataset2.y)){
    folds_inner <- leave_one_out_cv(dataset2.y,inner_k)
  }else{
    folds_inner <- stratified_balanced_cv(y=dataset2.y, nfolds=inner_k)
  }
  nfold_inner <- length(folds_inner)
  cat("Number of inner folds", nfold_inner,"\n")

  # START K-Fold   [inner-loop]    
  selected_genes_inner = NULL
  max_acc_inner = 0
  max_thres_inner = 0
  max_prob_inner = 0

  for (rep in 1:n_repetitions){
    for (i_inner in 1:nfold_inner){
      # Define ith train and test set
      cat("Fold number", i_inner,"\n")
      dataset3.x <- dataset2.x[, -folds_inner[[i_inner]],drop=FALSE]
      dataset3.y <- dataset2.y[-folds_inner[[i_inner]]]
      dataset3.testX <- dataset2.x[, folds_inner[[i_inner]],drop=FALSE]
      dataset3.testY <- dataset2.y[folds_inner[[i_inner]]]
      dataset3.class_count <- table(dataset3.y) #classes count
      dataset3.class_levels <- unique(dataset3.y)
      dataset3.n_classes <- length(dataset3.class_levels)
      dataset3.n_samples <- nrow(dataset3.x)
      dataset3.n_genes <- ncol(dataset3.x)

      nsc_data <- list(x=dataset3.x, y=factor(dataset3.y), genenames=dataset.genesnames, geneids=1:length(dataset.genesnames))

      # select genes using NSC
      # nsc_train <- pamr.train(nsc_data, n.threshold=thresholds_number)
      # nsc_scales <- pamr.adaptthresh(nsc_train)
      # nsc_train <- pamr.train(nsc_data, threshold.scale=nsc_scales, n.threshold=thresholds_number)
      nsc_train <- pamr.train(nsc_data, n.threshold=thresholds_number)

      #teste
      nsc_prob <- pamr.predictmany(nsc_train, dataset3.testX)

      preds = nsc_prob$predclass
      thresholds = nsc_train$threshold
      probabilities = vector()
      nex=0
      errors=vector()

      for (iie in 1:ncol(preds)){        
        errors[iie] = length(which(!as.logical(preds[,iie] == dataset3.testY)))/length(dataset3.testY)
        prob_matrix = nsc_prob$prob[,,iie,drop=FALSE]
        total_probability = 0
        if(ncol(prob_matrix)==1){
            nex=1
            total_probability = total_probability + max(prob_matrix[,1,,drop=FALSE])
        }else{
          nex = nrow(prob_matrix)
          for (iiie in 1:nrow(prob_matrix)){
            total_probability = total_probability + max(prob_matrix[iiie,,,drop=FALSE])
          }
        }
        probabilities[iie] = total_probability/nex
      }

      cat("\n#-#-#Erros ", errors)
      cat("\n#-#-#Probs ", probabilities,"\n")

      result_fold <- getLowerErrorMaxThresholdMaxProbability(errors,thresholds,probabilities)

      threshold = result_fold$higher_threshold
      cat("\n### max threshold[1] = ", threshold,"\n")

      sorted_thresholds = rev(sort(thresholds))
      if (threshold == sorted_thresholds[1]){ #max threshold implies in zero genes (?)
        threshold = sorted_thresholds[2] # get second higher threshold
      }
      cat("\n### max threshold[2] = ", threshold,"\n")

      # Filter the selected genes  by min error, max threshold and max probability, in this order
      selected_genes <- as.numeric(as.vector(pamr.listgenes(nsc_train, nsc_data, threshold)[,1])) #first column is gene names


      accs_kfold_inner <- 1 - result_fold$error
      cat("\n### acc = ", accs_kfold_inner,"\n")
      cat("\n### max_acc_inner = ", accs_kfold_inner,"\n")
      if (accs_kfold_inner > max_acc_inner){
        selected_genes_inner = selected_genes
        max_acc_inner = accs_kfold_inner
        max_prob_inner = result_fold$probability
        max_thres_inner = result_fold$higher_threshold
      }else{
        if (accs_kfold_inner == max_acc_inner){
          if(result_fold$probability > max_prob_inner){
            selected_genes_inner = selected_genes
            # max_acc_inner = accs_kfold_inner
            max_prob_inner = result_fold$probability
            max_thres_inner = result_fold$higher_threshold            
          }else{
            if(result_fold$probability == max_prob_inner){
              if(result_fold$higher_threshold > max_thres_inner){
                selected_genes_inner = selected_genes
                # max_acc_inner = accs_kfold_inner
                # max_prob_inner = result_fold$probability
                max_thres_inner = result_fold$higher_threshold
              }else{
                  cat("######################################################################################################################################################################### PROBLEM: There are more than one list with the same threshold, prob and accuracy. \n")
              }
            }
          }
        }
      }

      # If a gene is selected, than increment its global frequency
      for (j in 1:length(selected_genes)){
        global_genes_freq_inner[selected_genes[j]] = global_genes_freq_inner[selected_genes[j]] + 1
      }    
    } # end inner k-fold
  } # end rep

  # testing the threshold found in the inner-loop
  nsc_data_outer <- list(x=dataset2.x, y=factor(dataset2.y), genenames=dataset.genesnames, geneids=1:length(dataset.genesnames))

  # select genes using NSC
  # nsc_train_outer <- pamr.train(nsc_data_outer, n.threshold=thresholds_number)
  # nsc_scales_outer <- pamr.adaptthresh(nsc_train_outer)
  # nsc_train_outer <- pamr.train(nsc_data_outer, threshold.scale=nsc_scales_outer,  n.threshold=thresholds_number)
  nsc_train_outer <- pamr.train(nsc_data_outer, threshold=max_thres_inner, n.threshold=thresholds_number)

  #adapt threshold so that it will not eliminate all genes

  cat("\n\n#-# Used threshold in fold ", i_outer, " is ", max_thres_inner,"\n\n")
  #teste
  pred = pamr.predict(nsc_train_outer, dataset2.testX, max_thres_inner, type=c("class"))
  pred2 = pamr.predict(nsc_train_outer, dataset2.testX, 0.0000, type=c("class"))
  cat("\n\n*******pred == gtruth \n\n",pred, dataset2.testY)
  acc = NULL
  acc2 = NULL
  if (outer_k == length(dataset.y)){
    if (as.logical(pred == dataset2.testY)){
      acc = 1    
    }else{
      acc = 0
    }
    if (as.logical(pred2 == dataset2.testY)){
      acc2 = 1    
    }else{
      acc2 = 0
    }
  }else{
    acc = length(which(as.logical(pred == dataset2.testY)))/length(dataset2.testY)
    acc2 = length(which(as.logical(pred2 == dataset2.testY)))/length(dataset2.testY)
  }  

  
      
  accs_kfold_outer <- append(accs_kfold_outer, acc)
  accs_kfold_outer_all_genes <- append(accs_kfold_outer_all_genes, acc2)

  if(acc > max_acc_outer){
    max_acc_outer = acc
    max_thres_outer = max_thres_inner
  }else{
    if (acc == max_acc_outer){
      if(max_thres_outer < max_thres_inner){
        max_thres_outer = max_thres_inner
      }
    }
  }

  cat("\nMax threshold",max_thres_inner ,"\n")
  thresholds = nsc_train_outer$threshold

  # sorted_thresholds = rev(sort(thresholds))
  # if (max_thres_inner >= sorted_thresholds[1]){ #max threshold implies in zero genes (?) (or one?)
  #   max_thres_inner = sorted_thresholds[2] # get second higher threshold
  #   cat("\nPicking second max threshold",max_thres_inner ,"\n")
  # }

  # If a gene is selected, than increment its global frequency
  selected_genes <- as.numeric(as.vector(pamr.listgenes(nsc_train_outer, nsc_data_outer, max_thres_inner)[,1]))
  cat("\n",length(selected_genes),"Genes were selected. \n")
  # for (j in 1:length(selected_genes)){
  #   cat("Selected: ",selected_genes[j],"\n")
  # }
  # cat("Max index of selected genes", max(selected_genes), "\n")
  for (j in 1:length(selected_genes)){
    global_genes_freq_outer[selected_genes[j]] = global_genes_freq_outer[selected_genes[j]] + 1
  } 

}# END - K-fold - outer

accs_kfold_outer =  as.numeric(accs_kfold_outer)
accs_kfold_outer_all_genes = as.numeric(accs_kfold_outer_all_genes)


for (j in 1:length(accs_kfold_outer)){
  cat("*** acc: ",accs_kfold_outer[j],"\n")
}

avg_acc = mean(accs_kfold_outer)
avg_acc_all_genes = mean(accs_kfold_outer_all_genes)
cat("\n# Accuracy is", avg_acc,"\n")

# using selected threshold in the outer loop... compute the accuracy using the independent test


result <- cbind(data.frame(1:length(global_genes_freq_outer)),data.frame(global_genes_freq_outer),data.frame(rownames(dataset.x)))
colnames(result) <- c("index","freq","gene")
write.csv(result, file = "./results/double_cross_validation/nsc/genes_freq_matrix_outer.csv")

cat("Avg Acc DCV", avg_acc,"\n")

result <- cbind(data.frame(avg_acc),data.frame(avg_acc_all_genes))
colnames(result) <- c("avg acc dcv", "avg acc all genes - same folds from dcv")
write.csv(result, file = "./results/double_cross_validation/nsc/avg_std_acc.csv")

# sort genes by freq.
rank_by_freq <- rev(sort(global_genes_freq_inner, index.return = TRUE)$ix)

acc_by_n_freq_rank = rep(-1,length(rank_by_freq))
n_values = list()
#acc_by_n_freq_rank <- append(acc_by_n_freq_rank, 0)
#n_values <- append(n_values,1)
for(nfeatures in 2:length(rank_by_freq)){
  folds <- stratified_balanced_cv(y=dataset.y,nfolds=4)
  nfold_outer <- length(folds)
  acc_kfold = list()
  for (i in 1:nfold_outer){
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
result <- cbind(data.frame(rank_by_freq),data.frame(1:length(rank_by_freq)),data.frame(global_genes_freq_inner[rank_by_freq]),data.frame(rownames(dataset.x)[rank_by_freq]),data.frame(unlist(acc_by_n_freq_rank)))
colnames(result) <- c("index","rank","freq","gene","avg_cv_acc_after_nsc")
#write.matrix(merged, file = "big_matrix_svm-rfe.csv", sep = ",")
write.csv(result, file = "./results/double_cross_validation/nsc/genes_freq_matrix_inner_CV.csv")

# plot the cv result of rank by freq
pdf("./results/double_cross_validation/nsc/freq_inner_rank_cv_nsc.pdf")
plot(1:length(rank_by_freq),unlist(acc_by_n_freq_rank))
dev.off()


# testing the threshold found Double-Cross with INDEPENDENT test
nsc_data<- list(x=dataset.x, y=factor(dataset.y), genenames=dataset.genesnames, geneids=1:length(dataset.genesnames))

# select genes using NSC
nsc_train <- pamr.train(nsc_data)
nsc_scales <- pamr.adaptthresh(nsc_train)
nsc_train <- pamr.train(nsc_data, threshold.scale=nsc_scales, n.threshold=thresholds_number)

#test
pred = pamr.predict(nsc_train, dataset_test.x, max_thres_outer, type=c("class"))
pred2 = pamr.predict(nsc_train, dataset_test.x, 0.0000, type=c("class"))
cat("\npred ", pred,"\n")
cat("\nexpec ", dataset_test.y,"\n")
acc = length(which(as.logical(pred == dataset_test.y)))/length(dataset_test.y)   
acc2 = length(which(as.logical(pred2 == dataset_test.y)))/length(dataset_test.y)   

result <- cbind(data.frame(c(acc)),data.frame(c(acc2)))
colnames(result) <-  c("max DCV threshold","Threshold zero")
cat("independent acc: ", acc)
#write.matrix(merged, file = "big_matrix_svm-rfe.csv", sep = ",")
write.csv(result, file = "./results/double_cross_validation/nsc/independent_test_threshold_from_DCV.csv")



#final genes according to Double-Cross
dc_selected_genes <- as.numeric(as.vector(pamr.listgenes(nsc_train, nsc_data, max_thres_outer)[,1]))

result <- cbind(data.frame(dc_selected_genes),data.frame(1:length(dc_selected_genes)),data.frame(rownames(dataset.x)[dc_selected_genes]))
colnames(result) <- c("index","rank?","name")
#write.matrix(merged, file = "big_matrix_svm-rfe.csv", sep = ",")
write.csv(result, file = "./results/double_cross_validation/nsc/genes_DCV.csv")



cat("\n\n\nnumber of genes considered after kruskal filter: ", nrow(dataset.x))