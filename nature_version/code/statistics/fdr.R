
# Author: Henry Heberle
# contact: henry at icmc usp br

#Installing required packages
# list.of.packages <- c("combinat", "doSNOW","caret","snow", "pamr", "rpart", "parallel", "MASS", "e1071")
# new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

# if(length(new.packages)) install.packages(new.packages,repos="http://brieger.esalq.usp.br/CRAN/")

# library("rpart")
# library("MASS")
# #library("class")
# library("e1071")
# #library("dismo")
# library("pamr")
# library("caret")
# library("parallel")
# require(foreach)
# require(doSNOW)
# require(combinat)


input_file_name <-  "./dataset/proteins/independent_train.txt"
db <- read.table(input_file_name, header=FALSE, sep="\t")

pairwise <- FALSE

dataset.x <- as.matrix(db[3:(nrow(db)),2:ncol(db)])
class(dataset.x) <- "numeric"
rownames(dataset.x)<-db[3:nrow(db),1]
thresholds_number <- nrow(dataset.x)
dataset.y <- as.factor(as.matrix(db[2,2:ncol(db)]))
cat(dataset.y)
dataset.genesnames <- rownames(dataset.x)

cat("\n") 
for (test_name in c('T-Test','Wilcoxon rank-sum test',"Chi-squared")){
  cat(test_name)
  cat("\n") 

  kindexes = list()
  probs = c()
  ignore = FALSE
  for (i in 1:nrow(dataset.x)){      
      classes = unique(dataset.y)

      index_x = which(dataset.y == classes[1], arr.ind=TRUE)
      values_x = dataset.x[i,index_x]

      index_y = which(dataset.y == classes[2], arr.ind=TRUE)
      values_y = dataset.x[i,index_y]

      ktest <- NULL
      if (test_name=='T-Test'){
          if(pairwise){
            test_name = 'Paired T-Test'
            ktest <- t.test(values_x, values_y, paired=TRUE)
          }else{
            ktest <- t.test(values_x, values_y, paired=FALSE)
          }
      }else{
        if(test_name=='Wilcoxon rank-sum test'){          
          if(pairwise){
            test_name = 'Wilcoxon signed-rank test'
            ktest <- wilcox.test(values_x, values_y, paired=TRUE)
          }else{
            ktest <- wilcox.test(values_x, values_y, paired=FALSE)
          }
        }else{
          if(pairwise && test_name=="Chi-squared"){
            ktest <- chisq.test(values_x, values_y)
          }else{
            ignore = TRUE
          }
        }
      }
      probs <- c(probs,as.numeric(ktest$p.value))      
  }# end for each row
  
  # ignore this loop and jump to next
  if (ignore){
    next
  }


  
  # Plot p-value histogram
  #hist(probs, breaks=10, col="blue" xlab="p-value", main=paste(test_name, "Histogram", sep=" ")) 
  
  cat(ncol(dataset.x))
  cat("\n")  
  cat(nrow(dataset.x))
  cat("\n")
  cat(probs)
  cat("\n")
  
  result <- cbind(data.frame(1:length(rownames(dataset.x))), data.frame(rownames(dataset.x)), probs)
  colnames(result) <- c("index","name","Raw.p")
  cat(colnames(result))
  cat("\n")
  cat(ncol(result))
  cat("\n")
  result <- result[order(result$Raw.p),]

  result$Bonferroni =
        p.adjust(result$Raw.p,
                method = "bonferroni")

  result$BH =
        p.adjust(result$Raw.p,
                method = "BH")

  result$Holm =
        p.adjust(result$ Raw.p,
                method = "holm")

  result$Hochberg =
        p.adjust(result$ Raw.p,
                method = "hochberg")

  result$Hommel =
        p.adjust(result$ Raw.p,
                method = "hommel")

  result$BY =
        p.adjust(result$ Raw.p,
                method = "BY")


  write.csv(result, file = paste("./results/proteins/p values ",test_name,".csv", sep=""))

}




