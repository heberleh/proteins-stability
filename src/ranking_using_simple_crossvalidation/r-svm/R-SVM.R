### R-code for R-SVM
### use leave-one-out / Nfold or bootstrape to permute data for external CV
### build SVM model and use mean-balanced weight to sort genes on training set
### and recursive elimination of least important genes
### author: Dr. Xin Lu, Research Scientist
###   Biostatistics Department, Harvard School of Public Health

library(e1071)

## read in SVM formated data in filename
## the format following the defination of SVMTorch
## the first line contains 2 integer: nSample  nFeature+1 
## followed by a matrix, each row for one sample, with the last column being +/1 1 for class label
ReadSVMdata <- function(filename)
{
    dd <- read.table( filename, header=F, skip=1)
    x <- as.matrix( dd[, 1:(ncol(dd)-1)] ) 
    y <- factor( dd[, ncol(dd)] )
    
    ret <- list(x=x, y=y)
}

## create a decreasing ladder for recursive feature elimination
CreatLadder <- function( Ntotal, pRatio=0.75, Nmin=5 )
{
    x <- vector()
    x[1] <- Ntotal
    for( i in 1:100 )
    {
        pp <- round(x[i] * pRatio)
        if( pp == x[i] )
        {
            pp <- pp-1
        }          
        
        if( pp >= Nmin )
        {
            x[i+1] <- pp
        } else
        {
            break
        }
    }
    
    x
}

## R-SVM core code
## input:
##    x: row matrix of data
##    y: class label: 1 / -1 for 2 classes
##    CVtype: 
##        integer: N fold CV
##        "LOO":    leave-one-out CV
##        "bootstrape": bootstrape CV
##    CVnum:   number of CVs
##        LOO: defined as sample size
##        Nfold and bootstrape:  user defined, default as sample size
## output: a named list
##    Error: a vector of CV error on each level
##    SelFreq: a matrix for the frequency of each gene being selected in each level
##             with each column corresponds to a level of selection
##             and each row for a gene
##          The top important gene in each level are those high-freqent ones
RSVM <- function(x, y, ladder, CVtype, CVnum=0 )
{
    ## check if y is binary response
    Ytype <- names(table(y))
    if( length(Ytype) != 2) 
    {
        print("ERROR!! RSVM can only deal with 2-class problem")
        return(0)
    }

    ## class mean
    m1 <- apply(x[ which(y==Ytype[1]), ], 2, mean)
    m2 <- apply(x[ which(y==Ytype[2]), ], 2, mean)
    md <- m1-m2

    yy <- vector( length=length(y))
    yy[which(y==Ytype[1])] <- 1
    yy[which(y==Ytype[2])] <- -1        
    y <- yy

    ## check ladder
    if( min(diff(ladder)) >= 0 )
    {
        print("ERROR!! ladder must be monotonously decreasing")
        return(0);
    }
    
    if( ladder[1] != ncol(x) )
    {
        ladder <- c(ncol(x), ladder)
    }

    nSample <- nrow(x)
    nGene   <- ncol(x)
    SampInd <- seq(1, nSample)

    if( CVtype == "LOO" )
    {
        CVnum <- nSample
    } else
    {
        if( CVnum == 0 )
        {
            CVnum <- nSample
        }
    }
    
    ## vector for test error and number of tests
    ErrVec <- vector( length=length(ladder))
    names(ErrVec) <- paste("Lev_", ladder, sep="")
    nTests <- 0

    SelFreq <- matrix( 0, nrow=nGene, ncol=length(ladder))
    colnames(SelFreq) <- paste("Lev_", ladder, sep="")
    
    ## for each CV    
    for( i in 1:CVnum )
    {
    
        ## split data
        if( CVtype == "LOO" )
        {
            TestInd <- i
            TrainInd <- SampInd[ -TestInd]
        } else
        {
            if( CVtype == "bootstrape" )
            {
                TrainInd <- sample(SampInd, nSample, replace=T )
                TestInd <- SampInd[ which(!(SampInd %in% TrainInd ))]
            } else
            {
                ## Nfold
                TrainInd <- sample(SampInd, nSample*(CVtype-1)/CVtype )
                TestInd <- SampInd[ which(!(SampInd %in% TrainInd ))]
            }
        }

        nTests <- nTests + length(TestInd)
        
        ## in each level, train a SVM model and record test error
        xTrain <- x[TrainInd, ]
        yTrain <- y[TrainInd]
       
        xTest  <- x[TestInd,]
        yTest  <- y[TestInd]

        ## index of the genes used in the 
        SelInd <- seq(1, nGene)
        for( gLevel in 1:length(ladder) )
        {
            ## record the genes selected in this ladder
            SelFreq[SelInd, gLevel] <- SelFreq[SelInd, gLevel] +1
            
            ## train SVM model and test error
             svmres <- svm(xTrain[, SelInd], yTrain, scale=F, type="C-classification", kernel="linear" )
             if( CVtype == "LOO" )
             {
                 svmpred <- predict(svmres, matrix(xTest[SelInd], nrow=1) )
             } else
             {
                 svmpred <- predict(svmres, xTest[, SelInd] )
             }
             ErrVec[gLevel] <- ErrVec[gLevel] + sum(svmpred != yTest )
             
            ## weight vector
             W <- t(svmres$coefs*yTrain[svmres$index]) %*% svmres$SV * md[SelInd]
             rkW <- rank(W)
             
             if( gLevel < length(ladder) )
             {
                SelInd <- SelInd[which(rkW > (ladder[gLevel] - ladder[gLevel+1]))]
             }
        }

    }

    ret <- list(ladder=ladder, Error=ErrVec/nTests, SelFreq=SelFreq)
    
}

SummaryRSVM <- function( RSVMres )
{
    ERInd <- max( which(RSVMres$Error == min(RSVMres$Error)) )
    MinLevel <- RSVMres$ladder[ERInd]
    FreqVec <- RSVMres$SelFreq[, ERInd]
    
    SelInd <- which( rank(FreqVec) >= (ladder[1]-MinLevel) )

#    print("MinCV error of", min(RSVMres$Error), "at", MinLevel, "genes" )
    
    ret <- list( MinER=min(RSVMres$Error), MinLevel=MinLevel, SelInd=SelInd)
}
