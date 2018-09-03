
# Author: Henry Heberle
# contact: henry at icmc usp br

#Installing required packages
#


# slam_url <- "https://cran.r-project.org/src/contrib/Archive/slam/slam_0.1-37.tar.gz"
# install.packages(slam_url,repos=NULL)

# relations_url <- "http://cran.fhcrc.org/src/contrib/relations_0.6-7.tar.gz"
# install.packages(relations_url,repos=NULL)

list.of.packages <- c('relations',"slam","hash","quantreg","survival","nnet","ordinal","MXM","combinat", "doSNOW","snow", "rpart", "parallel", "MASS", "e1071")
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
input_file_name <- "./dataset/train_6_samples_independent.txt"
#input_file_name <- "./dataset/train_6_samples_independent_without_4_proteins.txt"
db <- read.table(input_file_name, header=FALSE, sep="\t")
kruskal_test <- TRUE


TUNE <- TRUE   # tune max_k and threshold?
k = 7          # k-fold cv for tuning


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
if (kruskal_test){
  #Kruskal filter
  for (i in 1:nrow(dataset.x)){
      ktest <- kruskal.test(x=dataset.x[i,], g=dataset.y)
      probs <- c(probs,as.numeric(ktest[3][1]))
      cat(as.numeric(ktest[3][1]))
      cat("\n")
      if (as.numeric(ktest[3][1]) < 0.05){
        kindexes <- append(kindexes,i)
      }
  }

  result <- cbind(data.frame(1:length(rownames(dataset.x))),data.frame(rownames(dataset.x)),probs)
  colnames(result) <- c("index","name","p-value")
  write.csv(result, file = "./results/double_cross_validation/kruskal.csv")

  kindexes <- as.numeric(kindexes)
  dataset.x <- dataset.x[kindexes,,drop=FALSE]  
  thresholds_number <- nrow(dataset.x)
  dataset.genesnames <- rownames(dataset.x)
}



#=========================== =============================== ==============================
#=========================== =============================== ==============================

# Tune parameters
cat(dataset.x[1,1:5])
cat("\n")
cat(dataset.x[2,1:5])
x <- t(dataset.x)

max_k <- 3#config$max_k
threshold<- 0.1#config$a
if (TUNE){                                                    
  tuned <- cv.ses(dataset.y, x, kfolds=k, task= "C", max_ks = c(9,8,7,6,5,4,3,2), ncores=detectCores(logical = TRUE)-1) # max_ks = c(7,6,5,4,3,2),
  config <- tuned$best_configuration
  max_k <- config$max_k
  threshold<- config$a  
}

# Finding signatures
result <- SES(dataset.y, x, max_k=max_k, threshold=threshold, robust=robust)

summary(result)

cat("signatures:")
print(result@queues)


cat("y:")
cat(dataset.y)

png("./results/plot_equivalent_signatures.png",width=950, height=550,res=100)
plot(result,mode="all")
dev.off()

cat(1:length(dataset.genesnames))
cat(dataset.genesnames)


# #final genes according to Double-Cross
# dc_selected_genes <- as.numeric(as.vector(rank))

# result <- cbind(data.frame(dc_selected_genes),data.frame(1:length(dc_selected_genes)),data.frame(rownames(dataset.x)[dc_selected_genes]))
# colnames(result) <- c("index","rank?","name")
# write.csv(result, file = "./results/double_cross_validation/svm/simple_rank.csv")


