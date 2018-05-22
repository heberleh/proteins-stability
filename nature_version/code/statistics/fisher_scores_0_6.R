
# Author: Henry Heberle
# contact: henry at icmc usp br

library("coin")

input_file_name <-  "./dataset/scores_1_7.csv"

db <- read.table(input_file_name, header=TRUE, sep=",",na.strings=c("", "NA"," "))

names <- colnames(db)

out <- NULL
n_features = ncol(db)


filename = "./results/fisher_scores_1_7.txt"
filename2 = "./results/fisher_p_value_table_scores_1_7.csv"

cat("Wilcoxon signed rank sum", file=filename, sep="\n\n", append=FALSE)

cat("Attribute 1, Attribute 2,  Fisher.2.sided.MonteCarlo, Fisher.less.MonteCarlo, Fisher.greater.MonteCarlo \n", file=filename2, sep="", append=FALSE)

# testes
# wilcoxsign_test
# fisher.test(x, y = NULL, workspace = 200000, hybrid = FALSE,
            # hybridPars = c(expect = 5, percent = 80, Emin = 1),
            # control = list(), or = 1, alternative = "two.sided",
            # conf.int = TRUE, conf.level = 0.95,
            # simulate.p.value = FALSE, B = 2000)


p_matrix <- c()
#zero.method = c("Pratt", "Wilcoxon")
method = "Pratt"
k = 1
for (i in 1:(n_features/2)){
    index1 = k
    index2 = k+1
    k = k+2

    db[]

    x <- db[,index1]
    y <- db[,index2]

    out <- capture.output(table(x, y))
    attributes_tested <- paste(names[index1], names[index2] , sep=" * ")
    cat(attributes_tested, out, file=filename, sep="\n", append=TRUE)

    # # alternative = c("two.sided", "less", "greater")
    # test1 <- fisher.test(x, y,  alternative="two.sided", workspace=2e8)
    # out <- capture.output(test1)
    # cat(out, file=filename, sep="\n", append=TRUE)
    # print(test1)

    test2 <- fisher.test(x, y,   alternative="two.sided", simulate.p.value=TRUE,B=1e4 )
    out <- capture.output(test2)
    cat(out, file=filename, sep="\n", append=TRUE)
    print(test2)
    
    # test3 <- fisher.test(x, y,    alternative="less" )
    # out <- capture.output(test3)
    # cat(out, file=filename, sep="\n", append=TRUE)
    # print(test3)

    test4 <- fisher.test(x, y,   alternative="less", simulate.p.value=TRUE, B=1e4 )
    out <- capture.output(test4)
    cat(out, file=filename, sep="\n", append=TRUE)

    # test5 <- fisher.test(x, y,    alternative="greater" )
    # out <- capture.output(test5)
    # cat(out, file=filename, sep="\n", append=TRUE)
    # print(test5)

    test6 <- fisher.test(x, y,  alternative="greater", simulate.p.value=TRUE, B=1e4  )
    out <- capture.output(test6)
    cat(out, file=filename, sep="\n", append=TRUE)
    
    cat("----------------------------------", file=filename, sep="\n", append=TRUE)
    p_values <- c(names[index1], names[index2], test2$p.value,  test4$p.value, test6$p.value)    
    cat(p_values, file=filename2, sep=",", append=TRUE) 
    cat("\n", file=filename2, sep="", append=TRUE)    
}
warnings()
#cat("My title", out, file="summary_of_my_very_time_consuming_regression.txt", sep="n", append=TRUE)







# #





