library(nrMCmix)

library(doParallel)
library(foreach)

#create the clusters
n_cores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
if(is.na(n_cores)) n_cores <- 1 

cl <- makeCluster(n_cores)
registerDoParallel(cl)

performance <- function(cl, n = 2000,K = 3,alpha = rep(0.1,K), threshold = 0.1,
                   N = 1000, B = 1000, mu0 = 0, lambda0 = 1, sigma2 = 1, xi = 0){
  
  #adjust alpha
  if(length(alpha) == 1){
    alpha <- rep(alpha,K)
  }
  
  #define the function for estimating the total variation distance
  dtv <- function(draws,a,b){
    
    #define the target density on the logit scale
    target <- Vectorize(function(x,a,b){
      exp(-lbeta(a,b) - b * x - (a+b) * log1p(exp(-x)))
    }, "x")
    
    #define the grid
    x_grid <- seq(-10,10,l = 512)
    dx <- diff(x_grid)
    
    #get the density estimate for this marginal distribution
    proposal <- density(qlogis(draws),from = -10, to = 10, n = length(x_grid))$y
    
    #save the absolute difference from the target
    tmp <- abs(proposal - sapply(x_grid,target,a = a, b = b))
    
    #return the integral using the trapezoidal rule
    0.5 * sum( dx * (head(tmp, -1) + tail(tmp, -1)) / 2 )
    
  }
  
  res <- foreach(b = seq_len(B), .packages = c("nrMCmix")) %dopar% {
    
    set.seed(b)
    
    y <- generate_data(
      n = n, K = K, alpha = alpha,
      mu0 = mu0, lambda0 = lambda0,
      sigma2 = sigma2
    )$y[,1]
    
    sinka <- capture.output(gauss <- nrMCmix(y, mu0 = mu0, lambda0 = lambda0,
                                            sigma2 = sigma2, K = K, alpha = alpha,
                                            warm_up = 0, N = 2*N, xi = xi))
    
    sinka <- capture.output(partition <- nrMCmix(y,kernel = "partition",
                                                K = K, alpha = alpha,
                                                warm_up = 0, N = 2*N, xi = xi))
    
    list(
      W_gauss = t(gauss$W),
      entropy_gauss = drop(gauss$entropy),
      W_partition = t(partition$W),
      entropy_partition = drop(partition$entropy)
    )
  }
  
  Ws_gauss <- simplify2array(lapply(res, `[[`, "W_gauss"))
  Ws_gauss <- aperm(Ws_gauss, c(3, 1, 2))
  
  Ws_partition <- simplify2array(lapply(res, `[[`, "W_partition"))
  Ws_partition <- aperm(Ws_partition, c(3, 1, 2))
  
  entropies_gauss <- do.call(rbind, lapply(res, `[[`, "entropy_gauss"))
  entropies_partition <- do.call(rbind, lapply(res, `[[`, "entropy_partition"))
  
  #use the first half iterations to compute the TV
  TV_gauss <- foreach(
    i = seq_len(N),
    .combine = c
  ) %dopar% {
    
    mean(sapply(seq_along(alpha), function(k)
      dtv(
        Ws_gauss[, k, i],
        alpha[k],
        sum(alpha[-k])
      )
    ))
    
  }
  
  TV_partition <- foreach(
    i = seq_len(N),
    .combine = c
  ) %dopar% {
    
    mean(sapply(seq_along(alpha), function(k)
      dtv(
        Ws_partition[, k, i],
        alpha[k],
        sum(alpha[-k])
      )
    ))
    
  }
  
  TVs <- cbind(TV_gauss, TV_partition)
  
  #get the number of iteration required to go below the threshold
  n_iters <- apply(TVs,2,function(x) which(x <= threshold)[1])
  
  #use the second half iterations to compute the ESS
  ESS_w <- c(
    
    mean(sapply(seq_along(alpha), function(k){
      mean(apply(Ws_gauss[,k,N + seq_len(N)],1,coda::effectiveSize))
    })),
    
    mean(sapply(seq_along(alpha), function(k){
      mean(apply(Ws_partition[,k,N + seq_len(N)],1,coda::effectiveSize))
    }))
    
  )
  
  ESS_entropy <- c(
    
    mean(apply(entropies_gauss,1,coda::effectiveSize)),
    
    mean(apply(entropies_partition,1,coda::effectiveSize))
    
  )
  
  #return all the quantities of interest
  rbind(n_iters,
        ESS_w,
        ESS_entropy)
  
}

xi_grid <- 10^seq(log10(0.01), log10(1000), length.out = 100)

results <- array(NA,dim = c(3,2,length(xi_grid)))
for(i in seq_along(xi_grid)){
  results[,,i] <- performance(xi = xi_grid[i])
  
  save(results, file = "results.RData")
}

stopCluster(cl)
