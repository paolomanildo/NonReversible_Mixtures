### TEMPERING
tryCatch(library(nrMCtempmix), error = function(x) Rcpp::sourceCpp("nrMCtempmix.cpp"))

#function to simulate some dataset
sim_data <- function(n = 1000,
                     w = c(0.1,0.7,0.2),
                     mu = c(-2,1,2),
                     sigma2 = 1,
                     seed = 123){
  set.seed(seed)
  z <- sample(length(w),n,TRUE,prob = w)
  rnorm(n,mu[z],sqrt(sigma2))
}

#function that estimate the comunication barrier
estimate_barrier <- function(y,
                             K = 3,
                             alpha = rep(1,K),
                             sigma2 = 1,
                             mu0 = 0,
                             lambda0 = 0.1,
                             warm_up = 0,
                             N = 1000,
                             thin = 1,
                             thin_scan = 0,
                             gibbs = FALSE,
                             reversible = FALSE,
                             inverse_temperatures = NULL,
                             TT = 30,
                             max_iters = 100,
                             verbose = TRUE,
                             tol = 1e-5){
    
  #start with a geometric grid
  if(is.null(inverse_temperatures)){
    inverse_temperatures <- c(2^-seq(0,20, l = TT),0)
  }
  
  #initialize the output
  out <- list()
  
  #loop until maximum iterations or convergence reached
  rel_err <- Inf
  it <- 1
  while(rel_err > tol && it < max_iters){
    
    #set the old temperature equal to the current one
    old_inv_temp <- inverse_temperatures
    
    #fit the model
    sinka <- capture.output(fit <- nrMCtempmix(y, K = K, alpha = alpha, sigma2 = sigma2,
                                              mu0 = mu0, lambda0 = lambda0,
                                              inverse_temperatures = inverse_temperatures,
                                              warm_up = warm_up, N = N, thin = thin,
                                              thin_scan = thin_scan,
                                              gibbs = gibbs, reversible = reversible))
    #pairwise rejection expected probs
    rs <- rev(1 - as.numeric(fit$swaps))
    
    #add a nugget term if needed
    rs[rs == 0] <- 1e-6
    rs[rs == 1] <- 1-1e-6
    
    #cumulate them
    Lambda <- c(0,cumsum(rs))
    betas <- rev(inverse_temperatures)
    
    #fit a monotone spline
    schedule <- splinefun(
      x = Lambda / max(Lambda),
      y = betas,
      method="monoH.FC"
    )
    
    #update the interval
    inverse_temperatures <- sort(schedule(seq(0,1,l = length(betas))),
                                 decreasing = TRUE)
    
    #check convergence
    rel_err <- sum((inverse_temperatures - old_inv_temp)^2) / 
      sum(old_inv_temp^2)
    
    #save the current results
    out[[it]] <- list(rs = rs,
                      Lambda = Lambda,
                      betas = betas,
                      schedule = schedule,
                      rel_err = rel_err)
    
    if(verbose){
      cat("\nIteration:",it,"relative error:",rel_err)
    }
    
    #increase the counter
    it <- it + 1
    
  }
  
  #get the optimal inverse temperature schedule
  inverse_temperatures <- sort(schedule(
    seq(0,1,l = ceiling(2*max(Lambda)) )
  ), decreasing = TRUE)
  
  #return the output
  list(inverse_temperatures = inverse_temperatures,
       Lambda = max(Lambda),
       iters = it,
       convergence = rel_err < tol,
       steps = out)
}

#convergence diagnostic plot
plot_barrier <- function(X){
  
  #get the graphical window appearence
  op <- par(no.readonly = TRUE)
  
  #canvas plot
  my_plot <- function(xlim,ylim,...){
    plot(xlim,ylim, type = "n", axes = FALSE, ...)
    usr <- par("usr")
    rect(usr[1],usr[3],usr[2],usr[4],
         col = adjustcolor("lightgray",0.5),
         border = NA)
    grid(col = "white")
    axis(1,lwd = NA, lwd.ticks = 1)
    axis(2,lwd = NA, lwd.ticks = 1)
  }
  
  par(mfrow = c(1,2), mar = c(4.1,4.1,.1,1.1))
  
  #communication barrier 
  xs <- seq_along(X)
  ys <- sapply(X,function(x) max(x$Lambda))
  my_plot(range(xs),range(ys),
          xlab = "Iterations",
          ylab = expression(Lambda))
  lines(xs,ys)
  
  #global communication barrier
  my_plot(c(0,1),c(0,1), xlab = expression(beta), ylab = expression(Lambda))
  x_grid <- seq(0,1,l = 512)
  colors <- sapply(colorRampPalette(c("yellow","red","black"))(length(X)),adjustcolor,0.9)
  for(i in seq_along(X)){
    lines(X[[i]]$schedule(x_grid),x_grid,col = colors[i])
  }
  
  par(op)
  
}

#simulate four dataset with different separation degrees
y1 <- sim_data(mu = c(-2,1,2))
y2 <- sim_data(mu = c(-4,1,4))
y3 <- sim_data(mu = c(-5,1,5))
y4 <- sim_data(mu = c(-8,1,8))

#find the optimal tempering schedule for each of them
set.seed(123)
inv_temps1 <- estimate_barrier(y1)
inv_temps2 <- estimate_barrier(y2)
inv_temps3 <- estimate_barrier(y3)
inv_temps4 <- estimate_barrier(y4)

#check the diagnostic
plot_barrier(inv_temps1$steps)
plot_barrier(inv_temps2$steps)
plot_barrier(inv_temps3$steps)
plot_barrier(inv_temps4$steps)

#save the temperature schedules
save(y1,y2,y3,y4,inv_temps1, inv_temps2, inv_temps3, inv_temps4, file = "schedules.RData")

