
# Author: Henry Heberle
# contact: henry at icmc usp br

library("coin")

input_file_name <-  "./dataset/scores_1_7.csv"

db <- read.table(input_file_name, header=TRUE, sep=",",na.strings=c("", "NA"," "))

names <- colnames(db)

out <- NULL
n_features = ncol(db)


filename = "./results/wilcoxon_scores_1_7.txt"
filename2 = "./results/wilcoxon_p_value_table_scores_1_7.csv"

cat("Wilcoxon signed rank sum", file=filename, sep="\n\n", append=FALSE)

cat("Attribute 1, Attribute 2, Wil.sig.2sided.exact, Wil.sig.2.sided.Asymptotic, Wil.sig.less.exact, Wil.sig.less.Asymptotic, Wil.sig.greater.exact, Wil.sig.greater.Asymptotic \n", file=filename2, sep="", append=FALSE)

p_matrix <- c()
#zero.method = c("Pratt", "Wilcoxon")
method = "Pratt"
k = 1
for (i in 1:(n_features/2)){
    index1 = k
    index2 = k+1
    k = k+2

    db[]

    out <- capture.output(table(db[,index1], db[,index2]))
    attributes_tested <- paste(names[index1], names[index2] , sep=" * ")
    cat(attributes_tested, out, file=filename, sep="\n", append=TRUE)

    # alternative = c("two.sided", "less", "greater")
    test1 <- wilcoxsign_test(db[,index1] ~ db[,index2], paired=TRUE, distribution="exact", alternative="two.sided", zero.method = method)
    out <- capture.output(test1)
    cat(out, file=filename, sep="\n", append=TRUE)
    print(test1)

    test2 <- wilcoxsign_test(db[,index1] ~ db[,index2], paired=TRUE,  alternative="two.sided", zero.method = method)
    out <- capture.output(test2)
    cat(out, file=filename, sep="\n", append=TRUE)
    
    test3 <- wilcoxsign_test(db[,index1] ~ db[,index2], paired=TRUE, distribution="exact", alternative="less", zero.method = method)
    out <- capture.output(test3)
    cat(out, file=filename, sep="\n", append=TRUE)
    print(test3)

    test4 <- wilcoxsign_test(db[,index1] ~ db[,index2], paired=TRUE,  alternative="less", zero.method = method)
    out <- capture.output(test4)
    cat(out, file=filename, sep="\n", append=TRUE)

    test5 <- wilcoxsign_test(db[,index1] ~ db[,index2], paired=TRUE, distribution="exact", alternative="greater", zero.method = method)
    out <- capture.output(test5)
    cat(out, file=filename, sep="\n", append=TRUE)
    print(test5)

    test6 <- wilcoxsign_test(db[,index1] ~ db[,index2], paired=TRUE, alternative="greater", zero.method = method)
    out <- capture.output(test6)
    cat(out, file=filename, sep="\n", append=TRUE)
    
    cat("----------------------------------", file=filename, sep="\n", append=TRUE)
    p_values <- c(names[index1], names[index2], pvalue(test1), pvalue(test2), pvalue(test3), pvalue(test4), pvalue(test5), pvalue(test6))    
    cat(p_values, file=filename2, sep=",", append=TRUE) 
    cat("\n", file=filename2, sep="", append=TRUE)    
}
warnings()
#cat("My title", out, file="summary_of_my_very_time_consuming_regression.txt", sep="n", append=TRUE)







# #





