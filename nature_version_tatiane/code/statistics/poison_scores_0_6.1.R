
# Author: Henry Heberle
# contact: henry at icmc usp br

library("coin")

#require(ggplot2)
require(sandwich)
require(msm)

input_file_name <-  "./dataset/scores_0_6.csv"

db <- read.table(input_file_name, header=TRUE, sep=",",na.strings=c("", "NA"," "))

names <- colnames(db)

out <- NULL
n_features = ncol(db)


filename = "./results/poison_scores_0_6.txt"
filename2 = "./results/poison_p_value_table_scores_0_6.csv"

cat("Statistics", file=filename, sep="\n\n", append=FALSE)
cat(summary(db), file=filename, sep="\n\n", append=TRUE)
cat("\n\n", file=filename, sep="\n\n", append=TRUE)
p_matrix <- c()
#zero.method = c("Pratt", "Wilcoxon")
method = "Pratt"
k = 1
for (i in 1:(n_features/2)){
    index1 = k
    index2 = k+1
    k = k+2

    x = db[,index1]
    y = db[,index2]


    print(hist(x)$counts)
    print(hist(y)$counts)
    print("------")

    breaks = c(0,1,2,3,4,5,6)
    x = hist(x, breaks=breaks)$counts
    y = hist(y, breaks=breaks)$counts


    out <- capture.output(table(x,y))

    attributes_tested <- paste(names[index1], names[index2] , sep=" * ")
    cat(attributes_tested, out, file=filename, sep="\n", append=TRUE)

    out<-capture.output(x)    
    cat(out, file=filename, sep="\n", append=TRUE)
    out<-capture.output(y)
    cat(out, file=filename, sep="\n", append=TRUE)

    test5 <- sign_test(x ~ y, paired=TRUE, distribution="exact", alternative="greater", zero.method = method)
    out <- capture.output(test5)
    cat(out, file=filename, sep="\n", append=TRUE)
    print(test5)

    
           
    cat("----------------------------------", file=filename, sep="\n", append=TRUE)

}
warnings()
#cat("My title", out, file="summary_of_my_very_time_consuming_regression.txt", sep="n", append=TRUE)







# #





