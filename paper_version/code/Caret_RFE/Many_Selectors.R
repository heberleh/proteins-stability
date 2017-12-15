
# Author: Henry Heberle
# contact: henry at icmc usp br

#Installing required packages

list.of.packages <- c("combinat", "doSNOW","caret","snow", "pamr", "rpart", "parallel", "MASS", "e1071","mlbench","Hmisc","randomForest","doMC")
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
library(caret)
library(mlbench)
library(Hmisc)
library(randomForest)
library(doMC)
require(foreach)
require(doSNOW)
require(combinat)

# for automatic Parallel computing inside CARET


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


#=========================== =============================== ==============================
#=========================== ===== PRE-PROCESSING DATA ===== ==============================
#lines are genes
#columns are samples
dataset.x <- as.matrix(db[3:(nrow(db)),2:ncol(db)])
class(dataset.x) <- "numeric"
rownames(dataset.x)<-db[3:nrow(db),1]
colnames(dataset.x)<-db[1,2:ncol(db)]
thresholds_number <- nrow(dataset.x)
dataset.y <-as.factor(as.matrix(db[2,2:ncol(db)]))
cat(dataset.y)
dataset.genesnames <- rownames(dataset.x)

dataset_test.x <- as.matrix(db_test[3:(nrow(db_test)),2:ncol(db_test)])
class(dataset_test.x) <- "numeric"
rownames(dataset_test.x)<-db_test[3:nrow(db_test),1]
colnames(dataset_test.x)<-db_test[1,2:ncol(db)]
dataset_test.y <- as.factor(as.matrix(db_test[2,2:ncol(db_test)]))
dataset_test.genesnames <- rownames(dataset_test.x)


#=========================== =============================== ==============================
#=========================== =============================== ==============================




#=========================== =============================== ==============================
#=========================== ============ START ============ ==============================
registerDoMC(cores = 4)
outer_k = 10
inner_k = 3

repetitions = 3
resampling_method = "repeatedcv"

x = t(dataset.x)
cat("factor? ",as.factor(as.numeric(dataset.y)),"\n")
y = as.factor(as.numeric(dataset.y))

cat("rows", nrow(x),"\n")
cat("cols", ncol(x),"\n")
cat("classes", length(y),"\n")
x_test = t(dataset_test.x)
y_test = as.factor(as.numeric(dataset_test.y))

set.seed(1)

subsets <- c(1:10)

# caretFuncs
# lmFuncs
# rfFuncs
# treebagFuncs
# ldaFuncs
# nbFuncs
# gamFuncs

#linear models
ctrl <- rfeControl(
                  #  functions = lmFuncs,
                  functions = treebagFuncs,
                   method = resampling_method,
                   repeats = repetitions,
                   verbose = FALSE,
                   returnResamp = "all"
                   )
              
lmProfile <- rfe(x, y,
                 sizes = subsets,
                 rfeControl = ctrl,
                #  method = "svmRadial",                
                #  testX = x_test,
                #  testY = y_test
                )


cat("Computed\n")

lmProfile

head(lmProfile$resample)


# pdf("./results/many_selectors/plot1.pdf")
# trellis.par.set(caretTheme())
# plot(lmProfile, type = c("g", "o"))
# dev.off()


# pdf("./results/many_selectors/plot2.pdf")
# plot1 <- xyplot(lmProfile, 
#                 type = c("g", "p", "smooth"), 
#                 ylab = "RMSE CV Estimates")     

# plot2 <- densityplot(lmProfile, 
#                      subset = subsets, 
#                      adjust = 1.25, 
#                      as.table = TRUE, 
#                      xlab = "RMSE CV Estimates", 
#                      pch = "|")

# print(plot1)
# print(plot2)
# dev.off()