
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


getLowerErrorMaxThresholdMaxProbability <- function(errors, thresholds, probabilities){
  min_error = min(errors)
  higher_threshold = min(thresholds)
  index = match(c(higher_threshold),thresholds) #???

  index_error = which(errors == min_error, arr.ind=TRUE)

  #entre os de erro minimo, qual o de max threshold
  selected_thres = thresholds[index_error]

  max_threshold =  max(selected_thres)

  index_threshold = which(thresholds == max_threshold, arr.ind=TRUE)

  index_error_thres = intersect(index_threshold, index_error)

  selected_probs = probabilities[index_error_thres]

  max_probability = max(selected_probs)

  index_prob = which(probabilities == max_probability, arr.ind = TRUE)

  index = intersect(index_prob, index_error_thres)[1]

  result = list(higher_threshold = max_threshold, error = min_error, probability = max_probability, index=index)
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
db_test <- read.table(input_file_name, header=FALSE, sep="\t")

class_delta <- 1
n_repetitions <- 1

TUNE <- FALSE                # PRE-TUNE PARAMETERS OF CLASSIFICATION MODELS?
#=========================== =============================== ==============================
#=========================== =============================== ==============================
#=========================== =============================== ==============================

#fileConn<-file("log.txt")

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
max_thres_outer = 0
max_acc_outer = 0

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


    folds_inner <- stratified_balanced_cv(y=dataset2.y, nfolds=inner_k)
    nfold_inner <- length(folds_inner)
    cat("Number of inner folds", nfold_inner,"\n")

    # START K-Fold   [inner-loop]    
    selected_genes_inner = NULL
    max_acc_inner = 0
    max_thres_inner = 0
    max_prob_inner = 0

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
      nsc_train <- pamr.train(nsc_data)
      nsc_scales <- pamr.adaptthresh(nsc_train)
      nsc_train <- pamr.train(nsc_data, threshold.scale=nsc_scales, n.threshold=thresholds_number)

      #teste
      nsc_prob <- pamr.predictmany(nsc_train, dataset3.testX)

      preds = nsc_prob$predclass
      thresholds = nsc_train$threshold
      probabilities = vector()
      nex=0
      errors=vector()
      for (iie in 1:ncol(preds)){
        errors[iie] = length(which(!as.logical(preds[,iie] == dataset2.testY)))/length(dataset2.testY)
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

      result_fold <- getLowerErrorMaxThresholdMaxProbability(errors,thresholds,probabilities)

      threshold = result_fold$higher_threshold
      sorted_thresholds = rev(sort(thresholds))
      if (threshold == sorted_thresholds[1]){ #max threshold implies in zero genes (?)
        threshold = sorted_thresholds[2] # get second higher threshold
      }

      # Filter the selected genes  by min error, max threshold and max probability, in this order
      selected_genes <- as.numeric(as.vector(pamr.listgenes(nsc_train, nsc_data, threshold)[,1])) #first column is gene names


      accs_kfold_inner <- 1 - result_fold$error
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
              max_prob_inner = result_fold$probability
              # max_thres_inner = result_fold$higher_threshold
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

    # testing the threshold found in the inner-loop
    nsc_data_outer <- list(x=dataset2.x, y=factor(dataset2.y), genenames=dataset.genesnames, geneids=1:length(dataset.genesnames))

    # select genes using NSC
    nsc_train_outer <- pamr.train(nsc_data_outer)
    nsc_scales_outer <- pamr.adaptthresh(nsc_train_outer)
    nsc_train_outer <- pamr.train(nsc_data_outer, threshold.scale=nsc_scales_outer, n.threshold=thresholds_number)

    #adapt threshold so that it will not eliminate all genes

    #teste
    pred = pamr.predict(nsc_train_outer, dataset2.testX, max_thres_inner, type=c("class"))
    acc = length(which(as.logical(pred == dataset2.testY)))/length(dataset2.testY)    
    accs_kfold_outer <- append(accs_kfold_outer, acc)

    if(acc > max_acc_outer){
      max_acc_outer = acc
      max_thres_outer = max_thres_inner
    }

    cat("\nMax threshold",max_thres_inner ,"\n")
    thresholds = nsc_train_outer$threshold

    sorted_thresholds = rev(sort(thresholds))
    if (max_thres_inner >= sorted_thresholds[1]){ #max threshold implies in zero genes (?) (or one?)
      max_thres_inner = sorted_thresholds[2] # get second higher threshold
      cat("\nPicking second max threshold",max_thres_inner ,"\n")
    }

    # If a gene is selected, than increment its global frequency
    selected_genes <- as.numeric(as.vector(pamr.listgenes(nsc_train_outer, nsc_data_outer, max_thres_inner)[,1])) 
    cat("\nGenes were selected. \n")
    for (j in 1:length(selected_genes)){
      cat("Selected: ",selected_genes[j],"\n")
    }
    cat("Max index of selected genes", max(selected_genes), "\n")
    for (j in 1:length(selected_genes)){
      global_genes_freq_inner[selected_genes[j]] = global_genes_freq_inner[selected_genes[j]] + 1
    } 

  }

  accs_kfold_outer =  as.numeric(accs_kfold_outer)
  # END - K-fold - outer
  for (j in 1:length(accs_kfold_outer)){
    cat("acc: ",accs_kfold_outer[j],"\n")
  }

  avg_acc_by_rep[rep] = mean(accs_kfold_outer)
  cat("\n# Accuracy for rep.",rep," is", avg_acc_by_rep[rep],"\n")

  std_acc_by_rep[rep] = sd(accs_kfold_outer)
  cat("\n# Std for rep.",rep," is", std_acc_by_rep[rep],"\n")

} # end reps  k-folds

# using selected threshold in the outer loop... compute the accuracy using the independent test

#avg_acc = mean(avg_acc_by_rep)
#sd_acc = sd(avg_acc_by_rep)

avg_acc = avg_acc_by_rep[1]
sd_acc = std_acc_by_rep[1]

cat("Avg Acc DCV", avg_acc,"\n")
cat("Std Acc DCV", sd_acc,"\n")

result <- cbind(data.frame(avg_acc),data.frame(sd_acc))
colnames(result) <- c("avg acc","std acc")
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
pdf("./results/ranking_using_simple_crossvalidation/nsc/freq_rank_cv_nsc.pdf")
plot(1:length(rank_by_freq),unlist(acc_by_n_freq_rank))
dev.off()


# dado o threshold final do K-Fold, utilizá-lo no conjunto de 22 amostras


# testar o threshold em uma crosvalidação


# testar o threshold com o teste independente


# teremos: acc de cross, acc de double-cross, e acc de teste independente 
                #(teoricamente da maior pra menor nesta ordem...)


# lista de genes final -> threshold aplicado no conjunto de 22 amostras


# salvar tal lista de genes para posterior criação de "arestas" e cálculo de todas as combinações possíveis e acurácias usando árvores de decisão.










# # generate a candidates Signatures selecting genes that appears at least at X% of experements
# # freq >= 0.5 * length(repetitions) * k

# acc_by_signature <- list()
# signatures <- list()
# sig = 1
# signature_heuristic <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
# for (pc in signature_heuristic){
#   signature = list()

#   minimum_freq = pc * n_repetitions * nfold_outer

#   for (geneidx in 1:length(global_genes_freq)){
#     if (global_genes_freq[geneidx] >= minimum_freq){
#       signature <- append(signature, geneidx)
#     }
#   }
#   signature <- unlist(signature)

#   if (length(signature)>1){
#     # make a CV using the signature
#     folds <- stratified_balanced_cv(y=dataset.y,nfolds=5)
#     nfold_outer <- length(folds)
#     acc_kfold = list()
#     for (i in 1:nfold_outer){
#           # Define ith train and test set
#       dataset2.x <- dataset.x[, -folds[[i]],drop=FALSE]
#       dataset2.y <- dataset.y[-folds[[i]]]
#       dataset2.testX <- dataset.x[, folds[[i]],drop=FALSE]
#       dataset2.testY <- dataset.y[folds[[i]]]
#       dataset2.class_count <- table(dataset2.y) #classes count
#       dataset2.class_levels <- unique(dataset2.y)
#       dataset2.n_classes <- length(dataset2.class_levels)
#       dataset2.n_samples <- nrow(dataset2.x)
#       dataset2.n_genes <- ncol(dataset2.x)

#       nsc_data <- list(x=dataset2.x[signature, ], y=dataset2.y,genenames=dataset.genesnames[signature],geneids=signature)

#       #  train NSC
#       nsc_train <- pamr.train(nsc_data)

#       #teste
#       pred = pamr.predict(nsc_train, dataset2.testX[signature, ], 0.000, type=c("class")) #consider all selected proteins (n features of this loop)
#       acc = length(which(as.logical(pred == dataset2.testY)))/length(dataset2.testY)
#       acc_kfold <- append(acc_kfold, acc)
#       acc_by_signature[sig] <- mean(unlist(acc_kfold))


#     }
#   }else{
#     acc_by_signature[sig] <- -1
#   }
#   signatures[sig] <- toString(rownames(dataset.x)[signature])
#   sig = sig + 1
# }

# # write signatures and CV results in File
# # ALERT: this CV results is highly biased since the frequencies of genes and the K-fold that test the signatures are computed using the same dataset
# result <- cbind(data.frame(signature_heuristic), data.frame(unlist(acc_by_signature)), data.frame(unlist(signatures)))
# colnames(result) <- c("presence_in_folds_and_repetitions","acc","signature")
# #write.matrix(merged, file = "big_matrix_svm-rfe.csv", sep = ",")
# write.csv(result, file = "./results/ranking_using_simple_crossvalidation/nsc/potential_signatures_cv_nsc.csv")


# # save the rank considering the entire dataset
# nsc_data <- list(x=dataset.x, y=factor(dataset.y), genenames=dataset.genesnames, geneids=dataset.genesnames)
# nsc_train <- pamr.train(nsc_data)
# nsc_scales <- pamr.adaptthresh(nsc_train)

# nsc_train <- pamr.train(nsc_data, threshold.scale=nsc_scales,n.threshold=thresholds_number, scale.sd=TRUE)

# complete_genes_list = pamr.listgenes(nsc_train, nsc_data, 0, genenames=TRUE)
# write.csv(complete_genes_list,"./results/ranking_using_simple_crossvalidation/nsc/rank_by_nsc_using_the_full_dataset.csv")
# #the end