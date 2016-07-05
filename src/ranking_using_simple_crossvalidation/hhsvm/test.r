require(MASS)

###################### load R code #########################
source("hhsvm.r")

#####################  Data #####################
bigN = 100
rho = 0

p = 20
q = 4

# figure out the correlation matrix
sigma = matrix(rep(0,p*p),nrow=p)
for (i in (1:p)){
   for (j in (1:p)){
      if (i==j)
          sigma[i,j]=1
      else if ((i<=q)&&(j<=q))
          sigma[i,j]=rho
   }
}

# mean vector
mu1 = rep(0,p)
mu1[1:q] = 1
mu2 = rep(0,p)
mu2[1:q] = -1


x1 = mvrnorm(n=bigN,mu1,sigma)
x2 = mvrnorm(n=bigN+1,mu2,sigma)
y1 = rep(1,bigN)
y2 = rep(-1,bigN+1)
trX = rbind(x1,x2)
trY = c(y1,y2)

N = 2*bigN+1  # size of the training data set


######################## DrHSVM Method  ###########################

lam2 = 10
delta = 3

g <- DrHSVM(trX,trY,lam2,delta=delta)
pre <- DrHSVM.predict(g,trX,trY)	# training error

np = dim(trX)
n = np[1]
p = np[2]
s <- apply(abs(g$beta), 1, sum)


matplot(s, cbind(g$beta,pre$err), type="n", cex.lab = 1.5,
      xlab=expression(paste("||",beta,"||",scriptscriptstyle(1))), ylab=expression(beta))
for(i in 1:q)
  lines(s, g$beta[,i], col=i+1, lty=1)
for(i in (q+1):p)
  lines(s, g$beta[,i], col=i+1, lty=2)

