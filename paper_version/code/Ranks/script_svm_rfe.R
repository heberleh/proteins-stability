library(rpart)
library(MASS)
library(class)
library(e1071)
library(dismo)

################################################
# Feature Ranking with SVM-RFE
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
svmrfeFeatureRankingForMulticlass = function(x,y){
    n = ncol(x)
    
    survivingFeaturesIndexes = seq(1:n)
    featureRankedList = vector(length=n)
    rankedFeatureIndex = n
    
    while(length(survivingFeaturesIndexes)>0){
        #train the support vector machine
        svmModel = svm(x[, survivingFeaturesIndexes], y, cost = 10, gamma=0.0001, cachesize=500,  scale=T, type="C-classification", kernel="linear" )
        
        #compute the weight vector
        multiclassWeights = svm.weights(svmModel)
		
		    #calcular o erro baseado nos pesos médios
		    #N=bestN()
        
        #compute ranking criteria
        multiclassWeights = multiclassWeights * multiclassWeights
        rankingCriteria = 0
        
        
        for(i in 1:ncol(multiclassWeights))rankingCriteria[i] = mean(multiclassWeights[,i])
        		
        
        if (rankedFeatureIndex == n-1){
          write.matrix(multiclassWeights, file = "multiclassWeights.csv", sep = ",") 
          auxm<-multiclassWeights
        }
        
        #rank the features
        (ranking = sort(rankingCriteria, index.return = TRUE)$ix)
        
		#ranking[i] é o index do index do atributo. em surviving pode ter restado o atributo de index 200 na posição 1. ou seja, surv[ranking[1]] = 200 -> atributo de index 200 da tabela original.
		
        #update feature ranked list
        (featureRankedList[rankedFeatureIndex] = survivingFeaturesIndexes[ranking[1]])
        rankedFeatureIndex = rankedFeatureIndex - 1
        
        #eliminate the feature with smallest ranking criterion
        (survivingFeaturesIndexes = survivingFeaturesIndexes[-ranking[1]])
        #cat(length(survivingFeaturesIndexes),"\n")
    }
	return (featureRankedList)
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

db <- read.table("./dataset/spectral_counts_no_zeros_input.txt", header=TRUE,sep="\t")
#data <- fulldata[,!(colnames(fulldata) %in% c("Gene_names","class"))]
#zeros <- db[ ,apply(db, 2, function(z) any(z==0))]
#cat("Foram removidas as proteínas:\n")
#print(zeros)
#db2 <- db[ ,apply(db, 2, function(z) !any(z==0))]

#diminuir a dimensionalidade apenas para testes rápidos do código
#db2 <- t(db[1:100,])
db2 <-t(db)
x <- as.matrix(db2[3:nrow(db2),2:(ncol(db2))])
class(x) <- "numeric"
#colnames(x)<-1:ncol(x)
colnames(x)<-db2[2,2:ncol(db2)]
#yvector<-as.vector(unlist(db2[3:ncol(db2),1]))
y <- as.factor(db2[3:nrow(db2),1])

#x = log(x)
#scale samples with mean zero and standard deviation one
#for(i in 1:nrow(x))x[i,] = (x[i,]-mean(x[i,]))/sd(x[i,])
#scale features with mean zero and standard deviation one
#for(i in 1:ncol(x))x[,i] = (x[,i]-mean(x[,i]))/sd(x[,i])
#x = 2*atan(x/2)

featureRankedList = svmrfeFeatureRankingForMulticlass(x,y)


#6-fold
nf <- {}
Accuracy <- {}
for(pow in 1550:1600){
  nfeatures = pow
  truePredictions = 0
  svmModel = svm(x[, featureRankedList[1:nfeatures]], y, cost = 10, gamma=0.0001, cachesize=500,  scale=T, type="C-classification", kernel="linear",cross=5) 
  nf<-rbind(nf,nfeatures)
  Accuracy<-rbind(Accuracy,mean(svmModel$accuracies))  
}


plot(nf,Accuracy)


names = colnames(subset(x,select=featureRankedList)) # ou... colnames(x[,featureRankedList])
x_ordered = cbind(names,t(x[,featureRankedList]))
merged = cbind(names,featureRankedList,1:length(names)) #merge(names,featureRankedList, by = "row.names", all = TRUE)
#merged = as.matrix(merged[-1])
colnames(merged) <- c("gene name","index","rank")

write.matrix(merged, file = "featureRankedList_SVMRFE_scale.csv", sep = ",")
write.matrix(x_ordered, file = "matrix_by_rank_scale.csv", sep = ",")


