
# Author: Henry Heberle
# contact: henry at icmc usp br

#Installing required packages

# sudo apt-get install r-cran-rcpp r-cran-rcppgsl r-cran-rcpproll
# sudo apt-get install r-cran-fastmatch r-cran-mass r-cran-slam r-cran-quantreg r-cran-survival r-cran-nnetReading

# bla <- "https://cran.r-project.org/src/contrib/MXM_1.3.2.tar.gz"
# install.packages(bla,repos=NULL)
# bla3 <- "https://cran.r-project.org/src/contrib/RcppZiggurat_0.1.4.tar.gz"
# install.packages(bla3,repos=NULL)

# slam_url <- "https://cran.r-project.org/src/contrib/Archive/slam/slam_0.1-37.tar.gz"
# install.packages(slam_url,repos=NULL)

# bla2 <- "https://cran.r-project.org/src/contrib/Rfast_1.8.8.tar.gz"
# install.packages(bla2,repos=NULL)


# relations_url <- "http://cran.fhcrc.org/src/contrib/relations_0.6-7.tar.gz"
# install.packages(relations_url,repos=NULL)


list.of.packages <- c("relations","slam","hash","quantreg","survival","nnet","ordinal","combinat", "doSNOW","snow", "rpart", "parallel", "MASS", "e1071","MXM")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

if(length(new.packages)) install.packages(new.packages,repos="http://brieger.esalq.usp.br/CRAN/")

library("parallel")
library("relations")
library("hash")
library("slam")
library("quantreg")
library("survival")
library("nnet")
library("ordinal")
library("MXM")
require(foreach)
require(doSNOW)
require(combinat)

#=========================== =============================== ==============================
#=========================== =============================== ==============================
#=========================== =============================== ==============================
#=========================== SETUP YOUR INPUT AND PARAMETERS ==============================
#=========================== =============================== ==============================
#=========================== =============================== ==============================
#=========================== =============================== ==============================

# PARAMETERS:    train.txt
# SEE THE train.txt AND FOLLOW THE PATTERN
input_file_name <- "./dataset/proteins/independent_train.txt"
db <- read.table(input_file_name, header=FALSE, sep="\t")
ttest <- TRUE



TUNE <- TRUE   # tune max_k and threshold?
k = 8          # k-fold cv for tuning


robust = TRUE # run robust statistical test?


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

      #ktest <- t.test(values_x, values_y, var.equal=TRUE, paired=FALSE)
      ktest <- wilcox.test(values_x, values_y)
      probs <- c(probs,as.numeric(ktest$p.value))
      cat(as.numeric(ktest[3][1]))
      cat("\n")
      if (as.numeric(ktest[3][1]) < 0.10){
        kindexes <- append(kindexes,i)
      }
  }


# result <- cbind(data.frame(1:length(rownames(dataset.x))),data.frame(rownames(dataset.x)),probs)
# colnames(result) <- c("index","name","p-value")
# write.csv(result, file = "./results/double_cross_validation/ttest.csv")

  kindexes <- as.numeric(kindexes)

  dataset.x <- dataset.x[kindexes,,drop=FALSE]  
  thresholds_number <- nrow(dataset.x)
  dataset.genesnames <- rownames(dataset.x)

}

#=========================== =============================== ==============================
#=========================== =============================== ==============================

# Tune parameters

x <- t(dataset.x)
tuned <- cv.ses(dataset.y, x, kfolds=k, max_ks = c(5,4,3,2), task= "C", ncores=8)

cat(tuned$best_performance)
config <- tuned$best_configuration

cat("maxk: ")
cat(config$max_k)
cat("\n")
cat("significance level: ")
cat(config$a)
cat("\n")
max_k <- config$max_k
threshold<- config$a #0.1#

# Finding signatures
result <- SES(dataset.y, x, max_k=max_k, threshold=threshold)

result

# cat("signatures:\n")
# print(result@queues)
# cat("\ny:\n")
# cat(dataset.y)
# png("./results/plot_equivalent_signatures.png",width=950, height=550,res=100)
# plot(result,mode="all")
# dev.off()

cat(dataset.genesnames)

# #final genes according to Double-Cross
# dc_selected_genes <- as.numeric(as.vector(rank))

# result <- cbind(data.frame(dc_selected_genes),data.frame(1:length(dc_selected_genes)),data.frame(rownames(dataset.x)[dc_selected_genes]))
# colnames(result) <- c("index","rank?","name")
# write.csv(result, file = "./results/double_cross_validation/svm/simple_rank.csv")


