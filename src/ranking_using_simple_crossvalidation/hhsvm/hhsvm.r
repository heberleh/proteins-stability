#######################################################################
### All rights are reserved by the authors.
### Authors: Li Wang, Ji Zhu, University of Michigan (jizhu@umich.edu)
### Date:    01/01/2007
#######################################################################
DrHSVM <- function(x, y, lambda, type = c("lasso", "lar"), trace = T, eps = 1e-10,
                   max.steps, delta = 2, scale = F)
{
### use truncated Huber square loss
### y \in {-1, 1}
### residual equals 1-yf
### scale: centers and standardizes predictors.
  call <- match.call()
  if(!missing(type))
    type <- as.character(substitute(type))
  type <- match.arg(type)
  
  if (lambda <0){
  	stop("lambda must be non-negative\n")
  }
  
  if(trace)
    switch(type,
           lasso = cat("LASSO sequence\n"),
           lars = cat("LAR sequence\n")
           )
          
  nm <- dim(x)
  n <- nm[1]
  m <- nm[2]
  maxvars <- m

  indm <- seq(m)
  indn <- seq(n)
  one <- rep(1, n)

  if (scale == T) {
### Center x and scale x, and save the means and sds
    meanx <- drop(one %*% x)/n
    x <- scale(x, meanx, F)   # centers x
    normx <- sqrt(drop(one %*% (x^2)))
    names(normx) <- NULL
    x <- scale(x, F, normx)   # scales x
  }
  z <- y*x
  
  ini <- DrHSVM.initial(x=x, y=y, delta=delta)
  residuals <- ini$residuals
  Cvec <- ini$Cvec
  actvar <- NULL
  actobs <- ini$actobs ## a T/F vector
  if(missing(max.steps))
    max.steps <- floor(sqrt(n)) * min(n,m) ###
  beta <- matrix(0, max.steps + 1, m)
  beta0 <- rep(0, max.steps + 1)
  beta0[1] <- ini$beta0 

### Necessary?
  loss <- rep(0, max.steps + 1)
  ind1 <- residuals >= delta
  ind2 <- residuals < delta & residuals >= 0
  loss[1] <- sum(residuals[ind2]^2) + sum(2*delta*residuals[ind1] - delta*delta)
  cor <- matrix(0, max.steps + 1, m)
  cor[1, ] <- Cvec
  cat("loss:", loss[1], "\n")
  cat("Cmax:", max(Cvec), "\n")
###
  
  drops <- F	# to do with "lasso"
  Sign <- NULL	# Keeps the sign of correlations
### Now the main loop over moves
###
  
  
  act <- "var"
  k <- 0
  while((k < max.steps)) {
    k <- k + 1
    inactive <- if(k == 1) indm else indm[ - actvar]
    C <- Cvec[inactive]
    Cmax <- max(abs(Cvec))
    if (act == "var") {
      if(!any(drops)) {
        new <- abs(C) >= Cmax - eps
        
        # For Debugging:
        #cat("Cmax= ",Cmax,"\n")
        #cat("inactive= ",inactive,"\n")
        #cat("new= ", new, "\n")
        #cat("Cvec= ", Cvec, "\n")
        #cat("Sign= ", Sign, "\n")
        #cat("beta= ", beta[k,], "\n")
                
        Sign <- c(Sign, sign(C[new]))
        C <- C[!new]	# for later
        new <- inactive[new]	# Get index numbers
        cat("Step", k, ":\tVariable", new, "\tadded\n")
        actvar <- c(actvar, new)			# array of indices of active predictors
      } 
    }else if (act == "obs0" || act == "obs2") {
      if (act == "obs0")
        new <- indn[ abs(residuals) < eps ]
      else
        new <- indn[ abs(delta-residuals) < eps ]
###
      tmpadd <- new[actobs[new] == F]
      tmpremove <- new[actobs[new] == T]
      if (length(tmpadd) != 0) {
        cat("Step", k, ":\tObservation", tmpadd, "\tadded(", act,
            ")\n")
        actobs[ tmpadd ] <- T
      }
      if (length(tmpremove) != 0) {
        cat("Step", k, ":\tObservation", tmpremove, "\tremoved(",
            act, ")\n")
        actobs[ tmpremove ] <- F
      }
    }else { stop("act has a fourth value.\n") }

    lenvar <- length(actvar)
    
    # There is no need to check singularity, since in DrHSVM the
    #   matrix is always non-singular.
    if (sum(actobs)>0){
    	tmp1 <- t(cbind(y[actobs], z[actobs, actvar, drop=F])) %*%
         	cbind(y[actobs], z[actobs, actvar, drop=F])
        
      	# tmp1 should be modified in DrHSVM
    	eye <- diag(lenvar+1)
    	eye[1,1] <- 0
    	tmp1 <- tmp1 + lambda*eye   
        
    	tmpqr <- qr(tmp1)
    	if (tmpqr$rank < lenvar + 1) {
        	cat("normal is true \n")
        	cat("singularity \n")
        	k <- k - 1
        	break
    	}else{
        	Gi1 <- qr.solve(tmpqr, c(0, Sign))
    	}
    }else{
		Gi1 <- rep(0,(lenvar+1))
		Gi1[1] = 0			# for d(beta0)/dlam1
		Gi1[2:(lenvar+1)] = Sign/lambda		# d(beta)/dlam1		Question: -Sign or Sign??
    }
    
### Compute how far to go
   
### How far to go if var change
###      A <- 1/sqrt(sum(Gi1 * Sign))
   	
      A <- 1
      w <- A * Gi1
      u <- drop(cbind(y, z[, actvar, drop = F]) %*% w)	# a vector with length n
	

      if(length(actvar) == maxvars) {
        gamvar <- Cmax/A
      }else if (sum(actobs)>0){
        a <- drop(u[actobs] %*% z[actobs, - actvar, drop=F])
        
        gam <- c((Cmax - C)/(A - a), (Cmax + C)/(A + a))		# Cmax=> lam1	C=> corr for inactive predictors
        gamvar <- min(gam[gam > eps], Cmax/A, na.rm=T)	
      	
        # For Debugging:
      	#cat("gam = ",gam,"\n")
      	#cat("gamvar= ",gamvar,"\n")
      
      }else{
		# no actobs points

		gam <- c((Cmax - C)/A , (Cmax + C)/A )		# Cmax=> lam1	C=> corr for inactive predictors
       	gamvar <- min(gam[gam > eps], Cmax/A, na.rm=T)	
	}
### How far to go if obs change

      gam0 <- residuals/u
      gam2 <- (residuals-delta)/u
      if (max(gam0) > eps)
        gamobs0 <- min(gam0[gam0 > eps], na.rm=T)
      else
        gamobs0 <- Inf
      if (max(gam2) > eps)
        gamobs2 <- min(gam2[gam2 > eps], na.rm=T)
      else
        gamobs2 <- Inf
### Compare which comes first
      if (gamvar < min(gamobs0, gamobs2)) {
        gamhat <- gamvar
        act <- "var"
      }
      else if (gamobs0 < min(gamobs2, gamvar)) {
        gamhat <- gamobs0
        act <- "obs0"
      }
      else if (gamobs2 < min(gamobs0, gamvar)) {
        gamhat <- gamobs2
        act <- "obs2"
      }
      else {
        cat("var, obs0, obs2 tie in normal\n")
        k <- k - 1
        break
      }
   
    #}
   
    if(type == "lasso" && k > 1) {
      tmpgam <- -beta[k, actvar]/Gi1[-1]
      if (max(tmpgam) > eps) {
        gam <- min(tmpgam[tmpgam > eps], na.rm=T)
        if (gam < gamhat) {
          gamhat <- gam
          act <- "var"
          drops <- tmpgam == gam
        }
        else
          drops <- F
      }
      else
        drops <- F
    }
    
    beta[k + 1,  ] <- beta[k,  ]
    beta[k + 1, actvar] <- beta[k + 1, actvar] + gamhat * w[-1]
    beta0[k + 1] <- beta0[k] + gamhat * w[1]
  	
    residuals <- residuals - gamhat * u
   
    # Cvec needs to be modified for DrHSVM
    if (sum(actobs)>0){
    	Cvec <- Cvec - gamhat * drop(u[actobs] %*% z[actobs, , drop=F]) 
    }else{
	Cvec <- Cvec	# inactive predictors' correlation do not change, if no points are in actobs	
    }
	
    Cvec[actvar] <- Cvec[actvar] - lambda*gamhat*w[-1]


### Necessary?
    ind1 <- residuals >= delta
    ind2 <- residuals < delta & residuals >= 0
    loss[k + 1] <- sum(residuals[ind2]^2) + sum(2*delta*residuals[ind1] - delta*delta)
    cor[k + 1, ] <- Cvec
    cat("loss:", loss[k + 1], "\n")
    cat("Cmax:", max(Cvec), "\n")
###
    
    if(type == "lasso" && any(drops)) {
      cat("Step", k, ":\tVariable", actvar[drops], "\tdropped\n")
      actvar <- actvar[!drops]
      Sign <- Sign[!drops]
    }

    if (act == "var" && gamhat == Cmax/A) {
      cat("Cmax/A reached\n")
      break
    }
    if (max(residuals) < eps) {
      cat("No point on left\n")
      break
    }
  }
  beta <- beta[seq(k + 1), ]
  beta0 <- beta0[seq(k + 1)]

### Necessary?
  loss <- loss[seq(k + 1)]
  cor <- cor[seq(k + 1), ]
  
  if (scale == T) {
    beta0 <- beta0 - beta %*% (meanx/normx)
    beta <- scale(beta, F, normx)
  }

  GACV <- DrHSVM.GACV(beta0,beta,x,y,delta,lambda,eps)

  lambda1 = apply(abs(cor),1,max)	# cor can also be returned.

  return(list(call = call, beta = beta, beta0 = beta0, residuals =
              residuals, Cvec = Cvec, loss = loss, lambda1 = lambda1, GACV = GACV))
}

# since beta = 0, the optimization target is the same as before.
# initial function does not need to be modified.
DrHSVM.initial <- function(x, y, delta) {
  f <- function(a, y) {
    res <- 1 - y*a
    ind1 <- res >= delta
    ind2 <- res < delta & res >=0
    sum(res[ind2]^2) + sum(2*delta*res[ind1] - delta*delta)
  }
  
  xmin <- optimize(f, c(-max(1, abs(delta-1)), max(1, abs(delta-1))),
                   maximum=F, y=y)
  beta0 <- xmin$minimum

  	if ((delta <=1)& (sum(y)==0)){
		beta0 <- 0		# it is also a optimum
	}

  residuals <- 1 - y * beta0
  actobs <- rep(F, length(y))
  actobs[residuals >= 0 & residuals < delta] <- T
  ind1 <- residuals >= delta
  if ( abs(beta0) == 1 || abs(beta0) == abs(1-delta) ) {
    cat("beta0 =", beta0, "\n")
    stop("Initial point at the boundary")
  }

  if (sum(actobs) == length(y)){
    Cvec <- t(residuals) %*% (y * x)
  }else if (sum(actobs) == 0) {    
	 Cvec <- delta * t(y[ind1]) %*% x[ind1, , drop=F]
  } else {
    Cvec <- t(residuals[actobs]) %*% (y[actobs] * x[actobs, , drop=F]) +
      delta * t(y[ind1]) %*% x[ind1, , drop=F]
  }

  cat("Initial:", sum(actobs), "actobs\n")
  cat("Initial: beta0:", beta0, "\n")
  
  return(list(beta0=beta0, Cvec=Cvec, residuals=residuals, actobs=actobs))
}


DrHSVM.predict <- function(object, newx, newy=NULL, eps = 1e-10) {
### Get rid of many zero coefficients
  coef1 <- object$beta
  c1 <- drop(rep(1, nrow(coef1)) %*% abs(coef1))
  nonzeros <- c1 > eps
  coef1 <- cbind(object$beta0, coef1[, nonzeros])
  newx <- cbind(1, newx[, nonzeros])
  fit <- newx %*% t(coef1)
  predict <- sign(fit)
  if (is.null(newy))
    return(list(coef = coef1, fit = fit, err=NULL))
  err <- apply(apply(predict, 2, FUN="!=", newy), 2, sum)/length(newy)
  return(list(fit = fit, err = err))
}

DrHSVM.GACV <- function(beta0,beta,x,y,delta,lam2,eps=1e-10){
	np <- dim(x)
	n <- np[1]
	p <- np[2]	

	fitM <-  cbind(1,x)%*%t(cbind(beta0,beta))	# col=number of steps, row=number of sample
	yfitM <- diag(y) %*% fitM	
	
	ns <- dim(yfitM)
	s <- ns[2]	# number of steps

	yM <- matrix(rep(y,s),nrow=n)	# a matrix with columns all equal to y


	lossM <- loss(yfitM,delta)
	der_lossM <- dloss.df(yfitM,y,delta)
	uM <- func.u(fitM,delta)
	der_uM <- du.df(fitM,delta)
 
	DM <- yfitM
	DM[] <- 0

	##################################### Method 1 for Approximation
	sumX <- apply(x,1,sum)	
	sumX <- sumX*sumX
	xM <- matrix(rep(sumX,s),nrow=n)
	
	H_fy <- yfitM
	H_fy[] <- 0

	index <- (yfitM>(1-delta))&(yfitM<=1)
	H_fy[index] <- 4*yfitM[index]-2
	index <- (yfitM<=(1-delta))
	H_fy[index] <- -2*delta

	H_ff <- yfitM
	H_ff[] <- 0

	index <- (yfitM>1)
	H_ff[index] <- lam2/xM[index]

	index <- (yfitM>(1-delta))&(yfitM<=1)
	H_ff[index] <- 2+lam2/xM[index]

	index <- (yfitM<=(1-delta))
	H_ff[index] <- lam2/xM[index]

	H <- -(1/H_ff)*H_fy

	DM <- -der_lossM*H*(yM-uM)/(1-der_uM*H)

	##################################### Method 2 for Approximation
	#index <- (yfitM>1)
	#DM[index] = 0

	#index <- (yfitM>(1-delta))&(yfitM<=1)
	# version 1
	##DM[index]= -der_lossM[index]* (-2*yfitM[index]+1) * (yM[index]-uM[index]) / (1-der_uM[index]*(-2*yfitM[index]+1))
	
	# version 2
	#DM[index]= -der_lossM[index]*1*(yM[index]-uM[index]) / (1-der_uM[index]*1)
	
	# version 3
	##DM[index]= -der_lossM[index]*(-2*yfitM[index]+1)*(yM[index]-uM[index]) / (1-der_uM[index]*1)
	
	#index <- (yfitM<=(1-delta))
	#DM[index] = der_lossM[index]*(yM[index]-uM[index])/der_uM[index]

	#cat("third case:",DM[index],"\n")

	
	#DM[DM<0] = 0	# This should not be included in the algorithm
				# However positive value of DM should be better approximation
	##############################################################
	DM[DM<0] = 0		

	V <- DM + lossM	
	
	GACV <- apply(V,2,sum)/n
	
	return(GACV)	
}

loss <- function(yfitM, delta){
	res <- yfitM
	res[yfitM>1]=0
	res[yfitM<=(1-delta)] =-2*delta*(res[yfitM<=(1-delta)])-delta*delta+2*delta
	res[(yfitM>(1-delta))&(yfitM<=1)] = (res[(yfitM>(1-delta))&(yfitM<=1)]-1)^2 	

	return(res)
}
# derivative of loss with respect to f (fitted value)
dloss.df <- function(yfitM,y,delta){
	ns <- dim(yfitM)
	n <- ns[1]	# number of samples
	s <- ns[2]	# number of steps

	yM <- matrix(rep(y,s),nrow=n)	# a matrix with columns all equal to y

	res <- yfitM
	res[yfitM>1]=0
	res[yfitM<=(1-delta)]=-2*delta*yM[yfitM<=(1-delta)]
	index <- (yfitM<=1)&(yfitM>(1-delta))
	res[index]= 2*(yfitM[index]-1)*yM[index]
	
	return(res)
}

func.u <- function(fitM,delta){
	res <- fitM
	if ((delta>0)&(delta<1)){
		index = fitM < (-1)		
		res[index] = -2*delta
	
		index = (fitM>=(-1))&(fitM<(delta-1))
		res[index] = -2*delta+2*(fitM[index]+1)

		index = (fitM>=(delta-1))&(fitM<(1-delta))
		res[index] = 0

		index = (fitM>=(1-delta))&(fitM<1)
		res[index] = 2*(fitM[index]-1+delta)		

		index = (fitM >=1)
		res[index] = 2*delta

		return(res)

	}else if ((delta<2)&(delta>=1)){
		index = fitM <(-1)
		res[index] = -2*delta		

		index = (fitM>=-1)&(fitM<(1-delta))
		res[index] = -2*delta+2*(fitM[index]+1)

		index = (fitM>=(1-delta))&(fitM<(delta-1))
		res[index] = 4-4*delta+4*(fitM[index]-1+delta)

		index = (fitM>=(delta-1))&(fitM<1)
		res[index] = 4*delta-4 + 2*(fitM[index]-delta+1)	

		index = (fitM>=1)
		res[index] = 2*delta

		return(res)

	}else if (delta>=2){
		index = fitM<(1-delta)
		res[index] = -2*delta

		index = (fitM>=(1-delta))&(fitM<-1)
		res[index] = -2*delta+2*(fitM[index]-1+delta)

		index = (fitM>=(-1))&(fitM<1)
		res[index] = -4+4*(fitM[index]+1)		

		index = (fitM>=1)&(fitM<(delta-1))
		res[index] = 4+2*(fitM[index]-1)

		index = fitM >= (delta-1)
		res[index] = 2*delta		
		
		return(res)
	}else{
		cat("Error: delta < 0\n")
		return(NULL)
	}


}

# derivative of u with respect to f
du.df <- function(fitM,delta){
	res <- fitM
	if ((delta>0)&(delta<1)){
		index = (fitM<(-1) )|(fitM>=1)
		res[index]=0

		index = (fitM>=(delta-1))&(fitM<(1-delta))
		res[index] = 0

		index = (fitM>=-1)&(fitM<(delta-1))
		res[index] = 2

		index = (fitM>=(1-delta))&(fitM<1)
		res[index] = 2

		return(res)
	}else if ((delta<2)&(delta>=1)){		
		index = (fitM<(-1) )|(fitM>=1)
		res[index]=0

		index = (fitM>=(-1))&(fitM<(1-delta))
		res[index]=2

		index = (fitM>=(1-delta))&(fitM<(delta-1))
		res[index]=4

		index = (fitM>=(delta-1))&(fitM<1)
		res[index]=2
		
		return(res)
	}else if (delta>=2){
		index = (fitM<(1-delta))|(fitM>=(delta-1))
		res[index]=0
	
		index = (fitM>=(1-delta))&(fitM<(-1))
		res[index]=2

		index = (fitM>=(-1))&(fitM<1)
		res[index]=4

		index = (fitM>=1)&(fitM<(delta-1))
		res[index]=2		

		return(res)
	}else{
		cat("Error: delta < 0\n")
		return(NULL)
	}
}



