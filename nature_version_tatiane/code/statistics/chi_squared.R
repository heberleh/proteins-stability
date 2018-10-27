
# Author: Henry Heberle
# contact: henry at icmc usp br

input_file_name <-  "./dataset/chi-squared.csv"

db <- read.table(input_file_name, header=TRUE, sep=",")

names <- colnames(db)
ind <- combn(NCOL(db),2)
out <- NULL

filename = "./results/chi_squared.txt"
filename2 = "./results/chi_squared_p_value_table.csv"
cat("Chi-squared", file=filename, sep="\n\n", append=FALSE)
cat("Attribute 1, Attribute 2, Pearson's Chi-squared test, Pearson's Chi-squared test with Yates' continuity correction, Pearson's Chi-squared test with simulated p-value (based on 2000 replicates)\n", file=filename2, sep="", append=FALSE)
p_matrix <- c()
lapply(1:NCOL(ind), function (i){
    out <- capture.output(table(db[,ind[1,i]], db[,ind[2,i]]))
    attributes_tested <- paste(names[ind[1,i]], names[ind[2,i]] , sep=" * ")
    cat(attributes_tested, out, file=filename, sep="\n", append=TRUE)

    test1 <- chisq.test(db[,ind[1,i]], db[,ind[2,i]],correct=FALSE)
    out <- capture.output(test1)
    cat(out, file=filename, sep="\n", append=TRUE)    

    test2 <- chisq.test(db[,ind[1,i]], db[,ind[2,i]],correct=TRUE)
    out <- capture.output(test2)
    cat(out, file=filename, sep="\n", append=TRUE)

    test3 <- chisq.test(db[,ind[1,i]], db[,ind[2,i]],correct=FALSE,simulate.p.value=TRUE)
    out <- capture.output(test3)
    cat(out, file=filename, sep="\n", append=TRUE)
    
    cat("----------------------------------", file=filename, sep="\n", append=TRUE)
    p_values <- c(names[ind[1,i]], names[ind[2,i]], test1$p.value, test2$p.value, test3$p.value)    
    cat(p_values, file=filename2, sep=",", append=TRUE) 
    cat("\n", file=filename2, sep="", append=TRUE)    
    
})

#cat("My title", out, file="summary_of_my_very_time_consuming_regression.txt", sep="n", append=TRUE)







#





