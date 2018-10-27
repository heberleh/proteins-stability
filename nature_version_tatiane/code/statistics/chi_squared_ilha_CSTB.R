
# Author: Henry Heberle
# contact: henry at icmc usp br

input_file_name <-  "./dataset/ILHA_CSTB.csv"

db <- read.table(input_file_name, header=TRUE, sep=",",na.strings=c("", "NA"))

names <- colnames(db)

out <- NULL
n_features = ncol(db)


filename = "./results/chi_squared_ILHA_CSTB.txt"
filename2 = "./results/chi_squared_p_value_table_ILHA_CSTB.csv"

cat("Chi-squared", file=filename, sep="\n\n", append=FALSE)

cat("Attribute 1, Attribute 2, Pearson's Chi-squared test, Pearson's Chi-squared test with Yates' continuity correction, Pearson's Chi-squared test with simulated p-value (based on 2000 replicates)\n", file=filename2, sep="", append=FALSE)

p_matrix <- c()

k = 1
for (i in 1:(n_features/2)){
    index1 = k
    index2 = k+1
    k = k+2

    db[]

    out <- capture.output(table(db[,index1], db[,index2]))
    attributes_tested <- paste(names[index1], names[index2] , sep=" * ")
    cat(attributes_tested, out, file=filename, sep="\n", append=TRUE)

    test1 <- chisq.test(db[,index1], db[,index2], correct=FALSE)
    out <- capture.output(test1)
    cat(out, file=filename, sep="\n", append=TRUE)    

    test2 <- chisq.test(db[,index1], db[,index2], correct=TRUE)
    out <- capture.output(test2)
    cat(out, file=filename, sep="\n", append=TRUE)

    test3 <- chisq.test(db[,index1], db[,index2],correct=FALSE,simulate.p.value=TRUE)
    out <- capture.output(test3)
    cat(out, file=filename, sep="\n", append=TRUE)
    
    cat("----------------------------------", file=filename, sep="\n", append=TRUE)
    p_values <- c(names[index1], names[index2], test1$p.value, test2$p.value, test3$p.value)    
    cat(p_values, file=filename2, sep=",", append=TRUE) 
    cat("\n", file=filename2, sep="", append=TRUE)    
}
warnings()
#cat("My title", out, file="summary_of_my_very_time_consuming_regression.txt", sep="n", append=TRUE)







# #





